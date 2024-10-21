
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1488_1, arg1489_1, arg1490_1, arg1491_1, arg1492_1, arg1493_1, arg1494_1, arg1495_1, arg1496_1, arg1497_1, arg1498_1, arg1499_1, arg1500_1, arg1501_1, arg1502_1, arg1503_1, arg1504_1, arg1505_1, arg1506_1, arg1507_1, arg1508_1, arg1509_1, arg1510_1, arg1511_1, arg1512_1, arg1513_1, arg1514_1, arg1515_1, arg1516_1, arg1517_1, arg1518_1, arg1519_1, arg1520_1, arg1521_1, arg1522_1, arg1523_1, arg1524_1, arg1525_1, arg1526_1, arg1527_1, arg1528_1, arg1529_1, arg1530_1, arg1531_1, arg1532_1, arg1533_1, arg1534_1, arg1535_1, arg1536_1, arg1537_1, arg1538_1, arg1539_1, arg1540_1, arg1541_1, arg1542_1, arg1543_1, arg1544_1, arg1545_1, arg1546_1, arg1547_1, arg1548_1, arg1549_1, arg1550_1, arg1551_1, arg1552_1, arg1553_1, arg1554_1, arg1555_1, arg1556_1, arg1557_1, arg1558_1, arg1559_1, arg1560_1, arg1561_1, arg1562_1, arg1563_1, arg1564_1, arg1565_1, arg1566_1, arg1567_1, arg1568_1, arg1569_1, arg1570_1, arg1571_1, arg1572_1, arg1573_1, arg1574_1, arg1575_1, arg1576_1, arg1577_1, arg1578_1, arg1579_1, arg1580_1, arg1581_1, arg1582_1, arg1583_1, arg1584_1, arg1585_1, arg1586_1, arg1587_1, arg1588_1, arg1589_1, arg1590_1, arg1591_1, arg1592_1, arg1593_1, arg1594_1, arg1595_1, arg1596_1, arg1597_1, arg1598_1, arg1599_1, arg1600_1, arg1601_1, arg1602_1, arg1603_1, arg1604_1, arg1605_1, arg1606_1, arg1607_1, arg1608_1, arg1609_1, arg1610_1, arg1611_1, arg1612_1, arg1613_1, arg1614_1, arg1615_1, arg1616_1, arg1617_1, arg1618_1, arg1619_1, arg1620_1, arg1621_1, arg1622_1, arg1623_1, arg1624_1, arg1625_1, arg1626_1, arg1627_1, arg1628_1, arg1629_1, arg1630_1, arg1631_1):
        convolution_325 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_951 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_325 = torch.ops.aten.sqrt.default(add_951);  add_951 = None
        reciprocal_325 = torch.ops.aten.reciprocal.default(sqrt_325);  sqrt_325 = None
        mul_1099 = torch.ops.aten.mul.Tensor(reciprocal_325, 1);  reciprocal_325 = None
        unsqueeze_2631 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_2632 = torch.ops.aten.unsqueeze.default(unsqueeze_2631, -1);  unsqueeze_2631 = None
        unsqueeze_2633 = torch.ops.aten.unsqueeze.default(mul_1099, -1);  mul_1099 = None
        unsqueeze_2634 = torch.ops.aten.unsqueeze.default(unsqueeze_2633, -1);  unsqueeze_2633 = None
        sub_325 = torch.ops.aten.sub.Tensor(convolution_325, unsqueeze_2632);  convolution_325 = unsqueeze_2632 = None
        mul_1100 = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_2634);  sub_325 = unsqueeze_2634 = None
        unsqueeze_2635 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_2636 = torch.ops.aten.unsqueeze.default(unsqueeze_2635, -1);  unsqueeze_2635 = None
        mul_1101 = torch.ops.aten.mul.Tensor(mul_1100, unsqueeze_2636);  mul_1100 = unsqueeze_2636 = None
        unsqueeze_2637 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_2638 = torch.ops.aten.unsqueeze.default(unsqueeze_2637, -1);  unsqueeze_2637 = None
        add_952 = torch.ops.aten.add.Tensor(mul_1101, unsqueeze_2638);  mul_1101 = unsqueeze_2638 = None
        relu_284 = torch.ops.aten.relu.default(add_952);  add_952 = None
        convolution_326 = torch.ops.aten.convolution.default(relu_284, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_284 = arg6_1 = None
        add_953 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_326 = torch.ops.aten.sqrt.default(add_953);  add_953 = None
        reciprocal_326 = torch.ops.aten.reciprocal.default(sqrt_326);  sqrt_326 = None
        mul_1102 = torch.ops.aten.mul.Tensor(reciprocal_326, 1);  reciprocal_326 = None
        unsqueeze_2639 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_2640 = torch.ops.aten.unsqueeze.default(unsqueeze_2639, -1);  unsqueeze_2639 = None
        unsqueeze_2641 = torch.ops.aten.unsqueeze.default(mul_1102, -1);  mul_1102 = None
        unsqueeze_2642 = torch.ops.aten.unsqueeze.default(unsqueeze_2641, -1);  unsqueeze_2641 = None
        sub_326 = torch.ops.aten.sub.Tensor(convolution_326, unsqueeze_2640);  convolution_326 = unsqueeze_2640 = None
        mul_1103 = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_2642);  sub_326 = unsqueeze_2642 = None
        unsqueeze_2643 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_2644 = torch.ops.aten.unsqueeze.default(unsqueeze_2643, -1);  unsqueeze_2643 = None
        mul_1104 = torch.ops.aten.mul.Tensor(mul_1103, unsqueeze_2644);  mul_1103 = unsqueeze_2644 = None
        unsqueeze_2645 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_2646 = torch.ops.aten.unsqueeze.default(unsqueeze_2645, -1);  unsqueeze_2645 = None
        add_954 = torch.ops.aten.add.Tensor(mul_1104, unsqueeze_2646);  mul_1104 = unsqueeze_2646 = None
        relu_285 = torch.ops.aten.relu.default(add_954);  add_954 = None
        convolution_327 = torch.ops.aten.convolution.default(relu_285, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg11_1 = None
        add_955 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_327 = torch.ops.aten.sqrt.default(add_955);  add_955 = None
        reciprocal_327 = torch.ops.aten.reciprocal.default(sqrt_327);  sqrt_327 = None
        mul_1105 = torch.ops.aten.mul.Tensor(reciprocal_327, 1);  reciprocal_327 = None
        unsqueeze_2647 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_2648 = torch.ops.aten.unsqueeze.default(unsqueeze_2647, -1);  unsqueeze_2647 = None
        unsqueeze_2649 = torch.ops.aten.unsqueeze.default(mul_1105, -1);  mul_1105 = None
        unsqueeze_2650 = torch.ops.aten.unsqueeze.default(unsqueeze_2649, -1);  unsqueeze_2649 = None
        sub_327 = torch.ops.aten.sub.Tensor(convolution_327, unsqueeze_2648);  convolution_327 = unsqueeze_2648 = None
        mul_1106 = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_2650);  sub_327 = unsqueeze_2650 = None
        unsqueeze_2651 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_2652 = torch.ops.aten.unsqueeze.default(unsqueeze_2651, -1);  unsqueeze_2651 = None
        mul_1107 = torch.ops.aten.mul.Tensor(mul_1106, unsqueeze_2652);  mul_1106 = unsqueeze_2652 = None
        unsqueeze_2653 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_2654 = torch.ops.aten.unsqueeze.default(unsqueeze_2653, -1);  unsqueeze_2653 = None
        add_956 = torch.ops.aten.add.Tensor(mul_1107, unsqueeze_2654);  mul_1107 = unsqueeze_2654 = None
        relu_286 = torch.ops.aten.relu.default(add_956);  add_956 = None
        convolution_328 = torch.ops.aten.convolution.default(relu_286, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_286 = arg16_1 = None
        add_957 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_328 = torch.ops.aten.sqrt.default(add_957);  add_957 = None
        reciprocal_328 = torch.ops.aten.reciprocal.default(sqrt_328);  sqrt_328 = None
        mul_1108 = torch.ops.aten.mul.Tensor(reciprocal_328, 1);  reciprocal_328 = None
        unsqueeze_2655 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_2656 = torch.ops.aten.unsqueeze.default(unsqueeze_2655, -1);  unsqueeze_2655 = None
        unsqueeze_2657 = torch.ops.aten.unsqueeze.default(mul_1108, -1);  mul_1108 = None
        unsqueeze_2658 = torch.ops.aten.unsqueeze.default(unsqueeze_2657, -1);  unsqueeze_2657 = None
        sub_328 = torch.ops.aten.sub.Tensor(convolution_328, unsqueeze_2656);  convolution_328 = unsqueeze_2656 = None
        mul_1109 = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_2658);  sub_328 = unsqueeze_2658 = None
        unsqueeze_2659 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_2660 = torch.ops.aten.unsqueeze.default(unsqueeze_2659, -1);  unsqueeze_2659 = None
        mul_1110 = torch.ops.aten.mul.Tensor(mul_1109, unsqueeze_2660);  mul_1109 = unsqueeze_2660 = None
        unsqueeze_2661 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_2662 = torch.ops.aten.unsqueeze.default(unsqueeze_2661, -1);  unsqueeze_2661 = None
        add_958 = torch.ops.aten.add.Tensor(mul_1110, unsqueeze_2662);  mul_1110 = unsqueeze_2662 = None
        relu_287 = torch.ops.aten.relu.default(add_958);  add_958 = None
        convolution_329 = torch.ops.aten.convolution.default(relu_287, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_287 = arg21_1 = None
        add_959 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_329 = torch.ops.aten.sqrt.default(add_959);  add_959 = None
        reciprocal_329 = torch.ops.aten.reciprocal.default(sqrt_329);  sqrt_329 = None
        mul_1111 = torch.ops.aten.mul.Tensor(reciprocal_329, 1);  reciprocal_329 = None
        unsqueeze_2663 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_2664 = torch.ops.aten.unsqueeze.default(unsqueeze_2663, -1);  unsqueeze_2663 = None
        unsqueeze_2665 = torch.ops.aten.unsqueeze.default(mul_1111, -1);  mul_1111 = None
        unsqueeze_2666 = torch.ops.aten.unsqueeze.default(unsqueeze_2665, -1);  unsqueeze_2665 = None
        sub_329 = torch.ops.aten.sub.Tensor(convolution_329, unsqueeze_2664);  convolution_329 = unsqueeze_2664 = None
        mul_1112 = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_2666);  sub_329 = unsqueeze_2666 = None
        unsqueeze_2667 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_2668 = torch.ops.aten.unsqueeze.default(unsqueeze_2667, -1);  unsqueeze_2667 = None
        mul_1113 = torch.ops.aten.mul.Tensor(mul_1112, unsqueeze_2668);  mul_1112 = unsqueeze_2668 = None
        unsqueeze_2669 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_2670 = torch.ops.aten.unsqueeze.default(unsqueeze_2669, -1);  unsqueeze_2669 = None
        add_960 = torch.ops.aten.add.Tensor(mul_1113, unsqueeze_2670);  mul_1113 = unsqueeze_2670 = None
        convolution_330 = torch.ops.aten.convolution.default(relu_285, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_285 = arg26_1 = None
        add_961 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_330 = torch.ops.aten.sqrt.default(add_961);  add_961 = None
        reciprocal_330 = torch.ops.aten.reciprocal.default(sqrt_330);  sqrt_330 = None
        mul_1114 = torch.ops.aten.mul.Tensor(reciprocal_330, 1);  reciprocal_330 = None
        unsqueeze_2671 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_2672 = torch.ops.aten.unsqueeze.default(unsqueeze_2671, -1);  unsqueeze_2671 = None
        unsqueeze_2673 = torch.ops.aten.unsqueeze.default(mul_1114, -1);  mul_1114 = None
        unsqueeze_2674 = torch.ops.aten.unsqueeze.default(unsqueeze_2673, -1);  unsqueeze_2673 = None
        sub_330 = torch.ops.aten.sub.Tensor(convolution_330, unsqueeze_2672);  convolution_330 = unsqueeze_2672 = None
        mul_1115 = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_2674);  sub_330 = unsqueeze_2674 = None
        unsqueeze_2675 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_2676 = torch.ops.aten.unsqueeze.default(unsqueeze_2675, -1);  unsqueeze_2675 = None
        mul_1116 = torch.ops.aten.mul.Tensor(mul_1115, unsqueeze_2676);  mul_1115 = unsqueeze_2676 = None
        unsqueeze_2677 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_2678 = torch.ops.aten.unsqueeze.default(unsqueeze_2677, -1);  unsqueeze_2677 = None
        add_962 = torch.ops.aten.add.Tensor(mul_1116, unsqueeze_2678);  mul_1116 = unsqueeze_2678 = None
        add_963 = torch.ops.aten.add.Tensor(add_960, add_962);  add_960 = add_962 = None
        relu_288 = torch.ops.aten.relu.default(add_963);  add_963 = None
        convolution_331 = torch.ops.aten.convolution.default(relu_288, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg31_1 = None
        add_964 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_331 = torch.ops.aten.sqrt.default(add_964);  add_964 = None
        reciprocal_331 = torch.ops.aten.reciprocal.default(sqrt_331);  sqrt_331 = None
        mul_1117 = torch.ops.aten.mul.Tensor(reciprocal_331, 1);  reciprocal_331 = None
        unsqueeze_2679 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_2680 = torch.ops.aten.unsqueeze.default(unsqueeze_2679, -1);  unsqueeze_2679 = None
        unsqueeze_2681 = torch.ops.aten.unsqueeze.default(mul_1117, -1);  mul_1117 = None
        unsqueeze_2682 = torch.ops.aten.unsqueeze.default(unsqueeze_2681, -1);  unsqueeze_2681 = None
        sub_331 = torch.ops.aten.sub.Tensor(convolution_331, unsqueeze_2680);  convolution_331 = unsqueeze_2680 = None
        mul_1118 = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_2682);  sub_331 = unsqueeze_2682 = None
        unsqueeze_2683 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_2684 = torch.ops.aten.unsqueeze.default(unsqueeze_2683, -1);  unsqueeze_2683 = None
        mul_1119 = torch.ops.aten.mul.Tensor(mul_1118, unsqueeze_2684);  mul_1118 = unsqueeze_2684 = None
        unsqueeze_2685 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_2686 = torch.ops.aten.unsqueeze.default(unsqueeze_2685, -1);  unsqueeze_2685 = None
        add_965 = torch.ops.aten.add.Tensor(mul_1119, unsqueeze_2686);  mul_1119 = unsqueeze_2686 = None
        relu_289 = torch.ops.aten.relu.default(add_965);  add_965 = None
        convolution_332 = torch.ops.aten.convolution.default(relu_289, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_289 = arg36_1 = None
        add_966 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_332 = torch.ops.aten.sqrt.default(add_966);  add_966 = None
        reciprocal_332 = torch.ops.aten.reciprocal.default(sqrt_332);  sqrt_332 = None
        mul_1120 = torch.ops.aten.mul.Tensor(reciprocal_332, 1);  reciprocal_332 = None
        unsqueeze_2687 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_2688 = torch.ops.aten.unsqueeze.default(unsqueeze_2687, -1);  unsqueeze_2687 = None
        unsqueeze_2689 = torch.ops.aten.unsqueeze.default(mul_1120, -1);  mul_1120 = None
        unsqueeze_2690 = torch.ops.aten.unsqueeze.default(unsqueeze_2689, -1);  unsqueeze_2689 = None
        sub_332 = torch.ops.aten.sub.Tensor(convolution_332, unsqueeze_2688);  convolution_332 = unsqueeze_2688 = None
        mul_1121 = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_2690);  sub_332 = unsqueeze_2690 = None
        unsqueeze_2691 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_2692 = torch.ops.aten.unsqueeze.default(unsqueeze_2691, -1);  unsqueeze_2691 = None
        mul_1122 = torch.ops.aten.mul.Tensor(mul_1121, unsqueeze_2692);  mul_1121 = unsqueeze_2692 = None
        unsqueeze_2693 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_2694 = torch.ops.aten.unsqueeze.default(unsqueeze_2693, -1);  unsqueeze_2693 = None
        add_967 = torch.ops.aten.add.Tensor(mul_1122, unsqueeze_2694);  mul_1122 = unsqueeze_2694 = None
        relu_290 = torch.ops.aten.relu.default(add_967);  add_967 = None
        convolution_333 = torch.ops.aten.convolution.default(relu_290, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_290 = arg41_1 = None
        add_968 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_333 = torch.ops.aten.sqrt.default(add_968);  add_968 = None
        reciprocal_333 = torch.ops.aten.reciprocal.default(sqrt_333);  sqrt_333 = None
        mul_1123 = torch.ops.aten.mul.Tensor(reciprocal_333, 1);  reciprocal_333 = None
        unsqueeze_2695 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_2696 = torch.ops.aten.unsqueeze.default(unsqueeze_2695, -1);  unsqueeze_2695 = None
        unsqueeze_2697 = torch.ops.aten.unsqueeze.default(mul_1123, -1);  mul_1123 = None
        unsqueeze_2698 = torch.ops.aten.unsqueeze.default(unsqueeze_2697, -1);  unsqueeze_2697 = None
        sub_333 = torch.ops.aten.sub.Tensor(convolution_333, unsqueeze_2696);  convolution_333 = unsqueeze_2696 = None
        mul_1124 = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_2698);  sub_333 = unsqueeze_2698 = None
        unsqueeze_2699 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_2700 = torch.ops.aten.unsqueeze.default(unsqueeze_2699, -1);  unsqueeze_2699 = None
        mul_1125 = torch.ops.aten.mul.Tensor(mul_1124, unsqueeze_2700);  mul_1124 = unsqueeze_2700 = None
        unsqueeze_2701 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_2702 = torch.ops.aten.unsqueeze.default(unsqueeze_2701, -1);  unsqueeze_2701 = None
        add_969 = torch.ops.aten.add.Tensor(mul_1125, unsqueeze_2702);  mul_1125 = unsqueeze_2702 = None
        add_970 = torch.ops.aten.add.Tensor(add_969, relu_288);  add_969 = relu_288 = None
        relu_291 = torch.ops.aten.relu.default(add_970);  add_970 = None
        convolution_334 = torch.ops.aten.convolution.default(relu_291, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg46_1 = None
        add_971 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_334 = torch.ops.aten.sqrt.default(add_971);  add_971 = None
        reciprocal_334 = torch.ops.aten.reciprocal.default(sqrt_334);  sqrt_334 = None
        mul_1126 = torch.ops.aten.mul.Tensor(reciprocal_334, 1);  reciprocal_334 = None
        unsqueeze_2703 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_2704 = torch.ops.aten.unsqueeze.default(unsqueeze_2703, -1);  unsqueeze_2703 = None
        unsqueeze_2705 = torch.ops.aten.unsqueeze.default(mul_1126, -1);  mul_1126 = None
        unsqueeze_2706 = torch.ops.aten.unsqueeze.default(unsqueeze_2705, -1);  unsqueeze_2705 = None
        sub_334 = torch.ops.aten.sub.Tensor(convolution_334, unsqueeze_2704);  convolution_334 = unsqueeze_2704 = None
        mul_1127 = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_2706);  sub_334 = unsqueeze_2706 = None
        unsqueeze_2707 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_2708 = torch.ops.aten.unsqueeze.default(unsqueeze_2707, -1);  unsqueeze_2707 = None
        mul_1128 = torch.ops.aten.mul.Tensor(mul_1127, unsqueeze_2708);  mul_1127 = unsqueeze_2708 = None
        unsqueeze_2709 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_2710 = torch.ops.aten.unsqueeze.default(unsqueeze_2709, -1);  unsqueeze_2709 = None
        add_972 = torch.ops.aten.add.Tensor(mul_1128, unsqueeze_2710);  mul_1128 = unsqueeze_2710 = None
        relu_292 = torch.ops.aten.relu.default(add_972);  add_972 = None
        convolution_335 = torch.ops.aten.convolution.default(relu_292, arg51_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_292 = arg51_1 = None
        add_973 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_335 = torch.ops.aten.sqrt.default(add_973);  add_973 = None
        reciprocal_335 = torch.ops.aten.reciprocal.default(sqrt_335);  sqrt_335 = None
        mul_1129 = torch.ops.aten.mul.Tensor(reciprocal_335, 1);  reciprocal_335 = None
        unsqueeze_2711 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_2712 = torch.ops.aten.unsqueeze.default(unsqueeze_2711, -1);  unsqueeze_2711 = None
        unsqueeze_2713 = torch.ops.aten.unsqueeze.default(mul_1129, -1);  mul_1129 = None
        unsqueeze_2714 = torch.ops.aten.unsqueeze.default(unsqueeze_2713, -1);  unsqueeze_2713 = None
        sub_335 = torch.ops.aten.sub.Tensor(convolution_335, unsqueeze_2712);  convolution_335 = unsqueeze_2712 = None
        mul_1130 = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_2714);  sub_335 = unsqueeze_2714 = None
        unsqueeze_2715 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_2716 = torch.ops.aten.unsqueeze.default(unsqueeze_2715, -1);  unsqueeze_2715 = None
        mul_1131 = torch.ops.aten.mul.Tensor(mul_1130, unsqueeze_2716);  mul_1130 = unsqueeze_2716 = None
        unsqueeze_2717 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_2718 = torch.ops.aten.unsqueeze.default(unsqueeze_2717, -1);  unsqueeze_2717 = None
        add_974 = torch.ops.aten.add.Tensor(mul_1131, unsqueeze_2718);  mul_1131 = unsqueeze_2718 = None
        relu_293 = torch.ops.aten.relu.default(add_974);  add_974 = None
        convolution_336 = torch.ops.aten.convolution.default(relu_293, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_293 = arg56_1 = None
        add_975 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_336 = torch.ops.aten.sqrt.default(add_975);  add_975 = None
        reciprocal_336 = torch.ops.aten.reciprocal.default(sqrt_336);  sqrt_336 = None
        mul_1132 = torch.ops.aten.mul.Tensor(reciprocal_336, 1);  reciprocal_336 = None
        unsqueeze_2719 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_2720 = torch.ops.aten.unsqueeze.default(unsqueeze_2719, -1);  unsqueeze_2719 = None
        unsqueeze_2721 = torch.ops.aten.unsqueeze.default(mul_1132, -1);  mul_1132 = None
        unsqueeze_2722 = torch.ops.aten.unsqueeze.default(unsqueeze_2721, -1);  unsqueeze_2721 = None
        sub_336 = torch.ops.aten.sub.Tensor(convolution_336, unsqueeze_2720);  convolution_336 = unsqueeze_2720 = None
        mul_1133 = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_2722);  sub_336 = unsqueeze_2722 = None
        unsqueeze_2723 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_2724 = torch.ops.aten.unsqueeze.default(unsqueeze_2723, -1);  unsqueeze_2723 = None
        mul_1134 = torch.ops.aten.mul.Tensor(mul_1133, unsqueeze_2724);  mul_1133 = unsqueeze_2724 = None
        unsqueeze_2725 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_2726 = torch.ops.aten.unsqueeze.default(unsqueeze_2725, -1);  unsqueeze_2725 = None
        add_976 = torch.ops.aten.add.Tensor(mul_1134, unsqueeze_2726);  mul_1134 = unsqueeze_2726 = None
        add_977 = torch.ops.aten.add.Tensor(add_976, relu_291);  add_976 = relu_291 = None
        relu_294 = torch.ops.aten.relu.default(add_977);  add_977 = None
        convolution_337 = torch.ops.aten.convolution.default(relu_294, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg61_1 = None
        add_978 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_337 = torch.ops.aten.sqrt.default(add_978);  add_978 = None
        reciprocal_337 = torch.ops.aten.reciprocal.default(sqrt_337);  sqrt_337 = None
        mul_1135 = torch.ops.aten.mul.Tensor(reciprocal_337, 1);  reciprocal_337 = None
        unsqueeze_2727 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_2728 = torch.ops.aten.unsqueeze.default(unsqueeze_2727, -1);  unsqueeze_2727 = None
        unsqueeze_2729 = torch.ops.aten.unsqueeze.default(mul_1135, -1);  mul_1135 = None
        unsqueeze_2730 = torch.ops.aten.unsqueeze.default(unsqueeze_2729, -1);  unsqueeze_2729 = None
        sub_337 = torch.ops.aten.sub.Tensor(convolution_337, unsqueeze_2728);  convolution_337 = unsqueeze_2728 = None
        mul_1136 = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_2730);  sub_337 = unsqueeze_2730 = None
        unsqueeze_2731 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_2732 = torch.ops.aten.unsqueeze.default(unsqueeze_2731, -1);  unsqueeze_2731 = None
        mul_1137 = torch.ops.aten.mul.Tensor(mul_1136, unsqueeze_2732);  mul_1136 = unsqueeze_2732 = None
        unsqueeze_2733 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_2734 = torch.ops.aten.unsqueeze.default(unsqueeze_2733, -1);  unsqueeze_2733 = None
        add_979 = torch.ops.aten.add.Tensor(mul_1137, unsqueeze_2734);  mul_1137 = unsqueeze_2734 = None
        relu_295 = torch.ops.aten.relu.default(add_979);  add_979 = None
        convolution_338 = torch.ops.aten.convolution.default(relu_295, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_295 = arg66_1 = None
        add_980 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_338 = torch.ops.aten.sqrt.default(add_980);  add_980 = None
        reciprocal_338 = torch.ops.aten.reciprocal.default(sqrt_338);  sqrt_338 = None
        mul_1138 = torch.ops.aten.mul.Tensor(reciprocal_338, 1);  reciprocal_338 = None
        unsqueeze_2735 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_2736 = torch.ops.aten.unsqueeze.default(unsqueeze_2735, -1);  unsqueeze_2735 = None
        unsqueeze_2737 = torch.ops.aten.unsqueeze.default(mul_1138, -1);  mul_1138 = None
        unsqueeze_2738 = torch.ops.aten.unsqueeze.default(unsqueeze_2737, -1);  unsqueeze_2737 = None
        sub_338 = torch.ops.aten.sub.Tensor(convolution_338, unsqueeze_2736);  convolution_338 = unsqueeze_2736 = None
        mul_1139 = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_2738);  sub_338 = unsqueeze_2738 = None
        unsqueeze_2739 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_2740 = torch.ops.aten.unsqueeze.default(unsqueeze_2739, -1);  unsqueeze_2739 = None
        mul_1140 = torch.ops.aten.mul.Tensor(mul_1139, unsqueeze_2740);  mul_1139 = unsqueeze_2740 = None
        unsqueeze_2741 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_2742 = torch.ops.aten.unsqueeze.default(unsqueeze_2741, -1);  unsqueeze_2741 = None
        add_981 = torch.ops.aten.add.Tensor(mul_1140, unsqueeze_2742);  mul_1140 = unsqueeze_2742 = None
        relu_296 = torch.ops.aten.relu.default(add_981);  add_981 = None
        convolution_339 = torch.ops.aten.convolution.default(relu_296, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_296 = arg71_1 = None
        add_982 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_339 = torch.ops.aten.sqrt.default(add_982);  add_982 = None
        reciprocal_339 = torch.ops.aten.reciprocal.default(sqrt_339);  sqrt_339 = None
        mul_1141 = torch.ops.aten.mul.Tensor(reciprocal_339, 1);  reciprocal_339 = None
        unsqueeze_2743 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_2744 = torch.ops.aten.unsqueeze.default(unsqueeze_2743, -1);  unsqueeze_2743 = None
        unsqueeze_2745 = torch.ops.aten.unsqueeze.default(mul_1141, -1);  mul_1141 = None
        unsqueeze_2746 = torch.ops.aten.unsqueeze.default(unsqueeze_2745, -1);  unsqueeze_2745 = None
        sub_339 = torch.ops.aten.sub.Tensor(convolution_339, unsqueeze_2744);  convolution_339 = unsqueeze_2744 = None
        mul_1142 = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_2746);  sub_339 = unsqueeze_2746 = None
        unsqueeze_2747 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_2748 = torch.ops.aten.unsqueeze.default(unsqueeze_2747, -1);  unsqueeze_2747 = None
        mul_1143 = torch.ops.aten.mul.Tensor(mul_1142, unsqueeze_2748);  mul_1142 = unsqueeze_2748 = None
        unsqueeze_2749 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_2750 = torch.ops.aten.unsqueeze.default(unsqueeze_2749, -1);  unsqueeze_2749 = None
        add_983 = torch.ops.aten.add.Tensor(mul_1143, unsqueeze_2750);  mul_1143 = unsqueeze_2750 = None
        add_984 = torch.ops.aten.add.Tensor(add_983, relu_294);  add_983 = relu_294 = None
        relu_297 = torch.ops.aten.relu.default(add_984);  add_984 = None
        convolution_340 = torch.ops.aten.convolution.default(relu_297, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg76_1 = None
        add_985 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_340 = torch.ops.aten.sqrt.default(add_985);  add_985 = None
        reciprocal_340 = torch.ops.aten.reciprocal.default(sqrt_340);  sqrt_340 = None
        mul_1144 = torch.ops.aten.mul.Tensor(reciprocal_340, 1);  reciprocal_340 = None
        unsqueeze_2751 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_2752 = torch.ops.aten.unsqueeze.default(unsqueeze_2751, -1);  unsqueeze_2751 = None
        unsqueeze_2753 = torch.ops.aten.unsqueeze.default(mul_1144, -1);  mul_1144 = None
        unsqueeze_2754 = torch.ops.aten.unsqueeze.default(unsqueeze_2753, -1);  unsqueeze_2753 = None
        sub_340 = torch.ops.aten.sub.Tensor(convolution_340, unsqueeze_2752);  convolution_340 = unsqueeze_2752 = None
        mul_1145 = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_2754);  sub_340 = unsqueeze_2754 = None
        unsqueeze_2755 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_2756 = torch.ops.aten.unsqueeze.default(unsqueeze_2755, -1);  unsqueeze_2755 = None
        mul_1146 = torch.ops.aten.mul.Tensor(mul_1145, unsqueeze_2756);  mul_1145 = unsqueeze_2756 = None
        unsqueeze_2757 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_2758 = torch.ops.aten.unsqueeze.default(unsqueeze_2757, -1);  unsqueeze_2757 = None
        add_986 = torch.ops.aten.add.Tensor(mul_1146, unsqueeze_2758);  mul_1146 = unsqueeze_2758 = None
        relu_298 = torch.ops.aten.relu.default(add_986);  add_986 = None
        convolution_341 = torch.ops.aten.convolution.default(relu_297, arg81_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_297 = arg81_1 = None
        add_987 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_341 = torch.ops.aten.sqrt.default(add_987);  add_987 = None
        reciprocal_341 = torch.ops.aten.reciprocal.default(sqrt_341);  sqrt_341 = None
        mul_1147 = torch.ops.aten.mul.Tensor(reciprocal_341, 1);  reciprocal_341 = None
        unsqueeze_2759 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_2760 = torch.ops.aten.unsqueeze.default(unsqueeze_2759, -1);  unsqueeze_2759 = None
        unsqueeze_2761 = torch.ops.aten.unsqueeze.default(mul_1147, -1);  mul_1147 = None
        unsqueeze_2762 = torch.ops.aten.unsqueeze.default(unsqueeze_2761, -1);  unsqueeze_2761 = None
        sub_341 = torch.ops.aten.sub.Tensor(convolution_341, unsqueeze_2760);  convolution_341 = unsqueeze_2760 = None
        mul_1148 = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_2762);  sub_341 = unsqueeze_2762 = None
        unsqueeze_2763 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_2764 = torch.ops.aten.unsqueeze.default(unsqueeze_2763, -1);  unsqueeze_2763 = None
        mul_1149 = torch.ops.aten.mul.Tensor(mul_1148, unsqueeze_2764);  mul_1148 = unsqueeze_2764 = None
        unsqueeze_2765 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_2766 = torch.ops.aten.unsqueeze.default(unsqueeze_2765, -1);  unsqueeze_2765 = None
        add_988 = torch.ops.aten.add.Tensor(mul_1149, unsqueeze_2766);  mul_1149 = unsqueeze_2766 = None
        relu_299 = torch.ops.aten.relu.default(add_988);  add_988 = None
        convolution_342 = torch.ops.aten.convolution.default(relu_298, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg86_1 = None
        add_989 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_342 = torch.ops.aten.sqrt.default(add_989);  add_989 = None
        reciprocal_342 = torch.ops.aten.reciprocal.default(sqrt_342);  sqrt_342 = None
        mul_1150 = torch.ops.aten.mul.Tensor(reciprocal_342, 1);  reciprocal_342 = None
        unsqueeze_2767 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_2768 = torch.ops.aten.unsqueeze.default(unsqueeze_2767, -1);  unsqueeze_2767 = None
        unsqueeze_2769 = torch.ops.aten.unsqueeze.default(mul_1150, -1);  mul_1150 = None
        unsqueeze_2770 = torch.ops.aten.unsqueeze.default(unsqueeze_2769, -1);  unsqueeze_2769 = None
        sub_342 = torch.ops.aten.sub.Tensor(convolution_342, unsqueeze_2768);  convolution_342 = unsqueeze_2768 = None
        mul_1151 = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_2770);  sub_342 = unsqueeze_2770 = None
        unsqueeze_2771 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_2772 = torch.ops.aten.unsqueeze.default(unsqueeze_2771, -1);  unsqueeze_2771 = None
        mul_1152 = torch.ops.aten.mul.Tensor(mul_1151, unsqueeze_2772);  mul_1151 = unsqueeze_2772 = None
        unsqueeze_2773 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_2774 = torch.ops.aten.unsqueeze.default(unsqueeze_2773, -1);  unsqueeze_2773 = None
        add_990 = torch.ops.aten.add.Tensor(mul_1152, unsqueeze_2774);  mul_1152 = unsqueeze_2774 = None
        relu_300 = torch.ops.aten.relu.default(add_990);  add_990 = None
        convolution_343 = torch.ops.aten.convolution.default(relu_300, arg91_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_300 = arg91_1 = None
        add_991 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_343 = torch.ops.aten.sqrt.default(add_991);  add_991 = None
        reciprocal_343 = torch.ops.aten.reciprocal.default(sqrt_343);  sqrt_343 = None
        mul_1153 = torch.ops.aten.mul.Tensor(reciprocal_343, 1);  reciprocal_343 = None
        unsqueeze_2775 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_2776 = torch.ops.aten.unsqueeze.default(unsqueeze_2775, -1);  unsqueeze_2775 = None
        unsqueeze_2777 = torch.ops.aten.unsqueeze.default(mul_1153, -1);  mul_1153 = None
        unsqueeze_2778 = torch.ops.aten.unsqueeze.default(unsqueeze_2777, -1);  unsqueeze_2777 = None
        sub_343 = torch.ops.aten.sub.Tensor(convolution_343, unsqueeze_2776);  convolution_343 = unsqueeze_2776 = None
        mul_1154 = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_2778);  sub_343 = unsqueeze_2778 = None
        unsqueeze_2779 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_2780 = torch.ops.aten.unsqueeze.default(unsqueeze_2779, -1);  unsqueeze_2779 = None
        mul_1155 = torch.ops.aten.mul.Tensor(mul_1154, unsqueeze_2780);  mul_1154 = unsqueeze_2780 = None
        unsqueeze_2781 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_2782 = torch.ops.aten.unsqueeze.default(unsqueeze_2781, -1);  unsqueeze_2781 = None
        add_992 = torch.ops.aten.add.Tensor(mul_1155, unsqueeze_2782);  mul_1155 = unsqueeze_2782 = None
        add_993 = torch.ops.aten.add.Tensor(add_992, relu_298);  add_992 = relu_298 = None
        relu_301 = torch.ops.aten.relu.default(add_993);  add_993 = None
        convolution_344 = torch.ops.aten.convolution.default(relu_301, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg96_1 = None
        add_994 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_344 = torch.ops.aten.sqrt.default(add_994);  add_994 = None
        reciprocal_344 = torch.ops.aten.reciprocal.default(sqrt_344);  sqrt_344 = None
        mul_1156 = torch.ops.aten.mul.Tensor(reciprocal_344, 1);  reciprocal_344 = None
        unsqueeze_2783 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_2784 = torch.ops.aten.unsqueeze.default(unsqueeze_2783, -1);  unsqueeze_2783 = None
        unsqueeze_2785 = torch.ops.aten.unsqueeze.default(mul_1156, -1);  mul_1156 = None
        unsqueeze_2786 = torch.ops.aten.unsqueeze.default(unsqueeze_2785, -1);  unsqueeze_2785 = None
        sub_344 = torch.ops.aten.sub.Tensor(convolution_344, unsqueeze_2784);  convolution_344 = unsqueeze_2784 = None
        mul_1157 = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_2786);  sub_344 = unsqueeze_2786 = None
        unsqueeze_2787 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_2788 = torch.ops.aten.unsqueeze.default(unsqueeze_2787, -1);  unsqueeze_2787 = None
        mul_1158 = torch.ops.aten.mul.Tensor(mul_1157, unsqueeze_2788);  mul_1157 = unsqueeze_2788 = None
        unsqueeze_2789 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_2790 = torch.ops.aten.unsqueeze.default(unsqueeze_2789, -1);  unsqueeze_2789 = None
        add_995 = torch.ops.aten.add.Tensor(mul_1158, unsqueeze_2790);  mul_1158 = unsqueeze_2790 = None
        relu_302 = torch.ops.aten.relu.default(add_995);  add_995 = None
        convolution_345 = torch.ops.aten.convolution.default(relu_302, arg101_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_302 = arg101_1 = None
        add_996 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_345 = torch.ops.aten.sqrt.default(add_996);  add_996 = None
        reciprocal_345 = torch.ops.aten.reciprocal.default(sqrt_345);  sqrt_345 = None
        mul_1159 = torch.ops.aten.mul.Tensor(reciprocal_345, 1);  reciprocal_345 = None
        unsqueeze_2791 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_2792 = torch.ops.aten.unsqueeze.default(unsqueeze_2791, -1);  unsqueeze_2791 = None
        unsqueeze_2793 = torch.ops.aten.unsqueeze.default(mul_1159, -1);  mul_1159 = None
        unsqueeze_2794 = torch.ops.aten.unsqueeze.default(unsqueeze_2793, -1);  unsqueeze_2793 = None
        sub_345 = torch.ops.aten.sub.Tensor(convolution_345, unsqueeze_2792);  convolution_345 = unsqueeze_2792 = None
        mul_1160 = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_2794);  sub_345 = unsqueeze_2794 = None
        unsqueeze_2795 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_2796 = torch.ops.aten.unsqueeze.default(unsqueeze_2795, -1);  unsqueeze_2795 = None
        mul_1161 = torch.ops.aten.mul.Tensor(mul_1160, unsqueeze_2796);  mul_1160 = unsqueeze_2796 = None
        unsqueeze_2797 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_2798 = torch.ops.aten.unsqueeze.default(unsqueeze_2797, -1);  unsqueeze_2797 = None
        add_997 = torch.ops.aten.add.Tensor(mul_1161, unsqueeze_2798);  mul_1161 = unsqueeze_2798 = None
        add_998 = torch.ops.aten.add.Tensor(add_997, relu_301);  add_997 = relu_301 = None
        relu_303 = torch.ops.aten.relu.default(add_998);  add_998 = None
        convolution_346 = torch.ops.aten.convolution.default(relu_303, arg106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg106_1 = None
        add_999 = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_346 = torch.ops.aten.sqrt.default(add_999);  add_999 = None
        reciprocal_346 = torch.ops.aten.reciprocal.default(sqrt_346);  sqrt_346 = None
        mul_1162 = torch.ops.aten.mul.Tensor(reciprocal_346, 1);  reciprocal_346 = None
        unsqueeze_2799 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_2800 = torch.ops.aten.unsqueeze.default(unsqueeze_2799, -1);  unsqueeze_2799 = None
        unsqueeze_2801 = torch.ops.aten.unsqueeze.default(mul_1162, -1);  mul_1162 = None
        unsqueeze_2802 = torch.ops.aten.unsqueeze.default(unsqueeze_2801, -1);  unsqueeze_2801 = None
        sub_346 = torch.ops.aten.sub.Tensor(convolution_346, unsqueeze_2800);  convolution_346 = unsqueeze_2800 = None
        mul_1163 = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_2802);  sub_346 = unsqueeze_2802 = None
        unsqueeze_2803 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_2804 = torch.ops.aten.unsqueeze.default(unsqueeze_2803, -1);  unsqueeze_2803 = None
        mul_1164 = torch.ops.aten.mul.Tensor(mul_1163, unsqueeze_2804);  mul_1163 = unsqueeze_2804 = None
        unsqueeze_2805 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_2806 = torch.ops.aten.unsqueeze.default(unsqueeze_2805, -1);  unsqueeze_2805 = None
        add_1000 = torch.ops.aten.add.Tensor(mul_1164, unsqueeze_2806);  mul_1164 = unsqueeze_2806 = None
        relu_304 = torch.ops.aten.relu.default(add_1000);  add_1000 = None
        convolution_347 = torch.ops.aten.convolution.default(relu_304, arg111_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_304 = arg111_1 = None
        add_1001 = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_347 = torch.ops.aten.sqrt.default(add_1001);  add_1001 = None
        reciprocal_347 = torch.ops.aten.reciprocal.default(sqrt_347);  sqrt_347 = None
        mul_1165 = torch.ops.aten.mul.Tensor(reciprocal_347, 1);  reciprocal_347 = None
        unsqueeze_2807 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_2808 = torch.ops.aten.unsqueeze.default(unsqueeze_2807, -1);  unsqueeze_2807 = None
        unsqueeze_2809 = torch.ops.aten.unsqueeze.default(mul_1165, -1);  mul_1165 = None
        unsqueeze_2810 = torch.ops.aten.unsqueeze.default(unsqueeze_2809, -1);  unsqueeze_2809 = None
        sub_347 = torch.ops.aten.sub.Tensor(convolution_347, unsqueeze_2808);  convolution_347 = unsqueeze_2808 = None
        mul_1166 = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_2810);  sub_347 = unsqueeze_2810 = None
        unsqueeze_2811 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_2812 = torch.ops.aten.unsqueeze.default(unsqueeze_2811, -1);  unsqueeze_2811 = None
        mul_1167 = torch.ops.aten.mul.Tensor(mul_1166, unsqueeze_2812);  mul_1166 = unsqueeze_2812 = None
        unsqueeze_2813 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_2814 = torch.ops.aten.unsqueeze.default(unsqueeze_2813, -1);  unsqueeze_2813 = None
        add_1002 = torch.ops.aten.add.Tensor(mul_1167, unsqueeze_2814);  mul_1167 = unsqueeze_2814 = None
        add_1003 = torch.ops.aten.add.Tensor(add_1002, relu_303);  add_1002 = relu_303 = None
        relu_305 = torch.ops.aten.relu.default(add_1003);  add_1003 = None
        convolution_348 = torch.ops.aten.convolution.default(relu_305, arg116_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg116_1 = None
        add_1004 = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_348 = torch.ops.aten.sqrt.default(add_1004);  add_1004 = None
        reciprocal_348 = torch.ops.aten.reciprocal.default(sqrt_348);  sqrt_348 = None
        mul_1168 = torch.ops.aten.mul.Tensor(reciprocal_348, 1);  reciprocal_348 = None
        unsqueeze_2815 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_2816 = torch.ops.aten.unsqueeze.default(unsqueeze_2815, -1);  unsqueeze_2815 = None
        unsqueeze_2817 = torch.ops.aten.unsqueeze.default(mul_1168, -1);  mul_1168 = None
        unsqueeze_2818 = torch.ops.aten.unsqueeze.default(unsqueeze_2817, -1);  unsqueeze_2817 = None
        sub_348 = torch.ops.aten.sub.Tensor(convolution_348, unsqueeze_2816);  convolution_348 = unsqueeze_2816 = None
        mul_1169 = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_2818);  sub_348 = unsqueeze_2818 = None
        unsqueeze_2819 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_2820 = torch.ops.aten.unsqueeze.default(unsqueeze_2819, -1);  unsqueeze_2819 = None
        mul_1170 = torch.ops.aten.mul.Tensor(mul_1169, unsqueeze_2820);  mul_1169 = unsqueeze_2820 = None
        unsqueeze_2821 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_2822 = torch.ops.aten.unsqueeze.default(unsqueeze_2821, -1);  unsqueeze_2821 = None
        add_1005 = torch.ops.aten.add.Tensor(mul_1170, unsqueeze_2822);  mul_1170 = unsqueeze_2822 = None
        relu_306 = torch.ops.aten.relu.default(add_1005);  add_1005 = None
        convolution_349 = torch.ops.aten.convolution.default(relu_306, arg121_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_306 = arg121_1 = None
        add_1006 = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_349 = torch.ops.aten.sqrt.default(add_1006);  add_1006 = None
        reciprocal_349 = torch.ops.aten.reciprocal.default(sqrt_349);  sqrt_349 = None
        mul_1171 = torch.ops.aten.mul.Tensor(reciprocal_349, 1);  reciprocal_349 = None
        unsqueeze_2823 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_2824 = torch.ops.aten.unsqueeze.default(unsqueeze_2823, -1);  unsqueeze_2823 = None
        unsqueeze_2825 = torch.ops.aten.unsqueeze.default(mul_1171, -1);  mul_1171 = None
        unsqueeze_2826 = torch.ops.aten.unsqueeze.default(unsqueeze_2825, -1);  unsqueeze_2825 = None
        sub_349 = torch.ops.aten.sub.Tensor(convolution_349, unsqueeze_2824);  convolution_349 = unsqueeze_2824 = None
        mul_1172 = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_2826);  sub_349 = unsqueeze_2826 = None
        unsqueeze_2827 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_2828 = torch.ops.aten.unsqueeze.default(unsqueeze_2827, -1);  unsqueeze_2827 = None
        mul_1173 = torch.ops.aten.mul.Tensor(mul_1172, unsqueeze_2828);  mul_1172 = unsqueeze_2828 = None
        unsqueeze_2829 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_2830 = torch.ops.aten.unsqueeze.default(unsqueeze_2829, -1);  unsqueeze_2829 = None
        add_1007 = torch.ops.aten.add.Tensor(mul_1173, unsqueeze_2830);  mul_1173 = unsqueeze_2830 = None
        add_1008 = torch.ops.aten.add.Tensor(add_1007, relu_305);  add_1007 = relu_305 = None
        relu_307 = torch.ops.aten.relu.default(add_1008);  add_1008 = None
        convolution_350 = torch.ops.aten.convolution.default(relu_299, arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg126_1 = None
        add_1009 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_350 = torch.ops.aten.sqrt.default(add_1009);  add_1009 = None
        reciprocal_350 = torch.ops.aten.reciprocal.default(sqrt_350);  sqrt_350 = None
        mul_1174 = torch.ops.aten.mul.Tensor(reciprocal_350, 1);  reciprocal_350 = None
        unsqueeze_2831 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_2832 = torch.ops.aten.unsqueeze.default(unsqueeze_2831, -1);  unsqueeze_2831 = None
        unsqueeze_2833 = torch.ops.aten.unsqueeze.default(mul_1174, -1);  mul_1174 = None
        unsqueeze_2834 = torch.ops.aten.unsqueeze.default(unsqueeze_2833, -1);  unsqueeze_2833 = None
        sub_350 = torch.ops.aten.sub.Tensor(convolution_350, unsqueeze_2832);  convolution_350 = unsqueeze_2832 = None
        mul_1175 = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_2834);  sub_350 = unsqueeze_2834 = None
        unsqueeze_2835 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_2836 = torch.ops.aten.unsqueeze.default(unsqueeze_2835, -1);  unsqueeze_2835 = None
        mul_1176 = torch.ops.aten.mul.Tensor(mul_1175, unsqueeze_2836);  mul_1175 = unsqueeze_2836 = None
        unsqueeze_2837 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_2838 = torch.ops.aten.unsqueeze.default(unsqueeze_2837, -1);  unsqueeze_2837 = None
        add_1010 = torch.ops.aten.add.Tensor(mul_1176, unsqueeze_2838);  mul_1176 = unsqueeze_2838 = None
        relu_308 = torch.ops.aten.relu.default(add_1010);  add_1010 = None
        convolution_351 = torch.ops.aten.convolution.default(relu_308, arg131_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_308 = arg131_1 = None
        add_1011 = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_351 = torch.ops.aten.sqrt.default(add_1011);  add_1011 = None
        reciprocal_351 = torch.ops.aten.reciprocal.default(sqrt_351);  sqrt_351 = None
        mul_1177 = torch.ops.aten.mul.Tensor(reciprocal_351, 1);  reciprocal_351 = None
        unsqueeze_2839 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_2840 = torch.ops.aten.unsqueeze.default(unsqueeze_2839, -1);  unsqueeze_2839 = None
        unsqueeze_2841 = torch.ops.aten.unsqueeze.default(mul_1177, -1);  mul_1177 = None
        unsqueeze_2842 = torch.ops.aten.unsqueeze.default(unsqueeze_2841, -1);  unsqueeze_2841 = None
        sub_351 = torch.ops.aten.sub.Tensor(convolution_351, unsqueeze_2840);  convolution_351 = unsqueeze_2840 = None
        mul_1178 = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_2842);  sub_351 = unsqueeze_2842 = None
        unsqueeze_2843 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_2844 = torch.ops.aten.unsqueeze.default(unsqueeze_2843, -1);  unsqueeze_2843 = None
        mul_1179 = torch.ops.aten.mul.Tensor(mul_1178, unsqueeze_2844);  mul_1178 = unsqueeze_2844 = None
        unsqueeze_2845 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_2846 = torch.ops.aten.unsqueeze.default(unsqueeze_2845, -1);  unsqueeze_2845 = None
        add_1012 = torch.ops.aten.add.Tensor(mul_1179, unsqueeze_2846);  mul_1179 = unsqueeze_2846 = None
        add_1013 = torch.ops.aten.add.Tensor(add_1012, relu_299);  add_1012 = relu_299 = None
        relu_309 = torch.ops.aten.relu.default(add_1013);  add_1013 = None
        convolution_352 = torch.ops.aten.convolution.default(relu_309, arg136_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg136_1 = None
        add_1014 = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_352 = torch.ops.aten.sqrt.default(add_1014);  add_1014 = None
        reciprocal_352 = torch.ops.aten.reciprocal.default(sqrt_352);  sqrt_352 = None
        mul_1180 = torch.ops.aten.mul.Tensor(reciprocal_352, 1);  reciprocal_352 = None
        unsqueeze_2847 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_2848 = torch.ops.aten.unsqueeze.default(unsqueeze_2847, -1);  unsqueeze_2847 = None
        unsqueeze_2849 = torch.ops.aten.unsqueeze.default(mul_1180, -1);  mul_1180 = None
        unsqueeze_2850 = torch.ops.aten.unsqueeze.default(unsqueeze_2849, -1);  unsqueeze_2849 = None
        sub_352 = torch.ops.aten.sub.Tensor(convolution_352, unsqueeze_2848);  convolution_352 = unsqueeze_2848 = None
        mul_1181 = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_2850);  sub_352 = unsqueeze_2850 = None
        unsqueeze_2851 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_2852 = torch.ops.aten.unsqueeze.default(unsqueeze_2851, -1);  unsqueeze_2851 = None
        mul_1182 = torch.ops.aten.mul.Tensor(mul_1181, unsqueeze_2852);  mul_1181 = unsqueeze_2852 = None
        unsqueeze_2853 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_2854 = torch.ops.aten.unsqueeze.default(unsqueeze_2853, -1);  unsqueeze_2853 = None
        add_1015 = torch.ops.aten.add.Tensor(mul_1182, unsqueeze_2854);  mul_1182 = unsqueeze_2854 = None
        relu_310 = torch.ops.aten.relu.default(add_1015);  add_1015 = None
        convolution_353 = torch.ops.aten.convolution.default(relu_310, arg141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_310 = arg141_1 = None
        add_1016 = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_353 = torch.ops.aten.sqrt.default(add_1016);  add_1016 = None
        reciprocal_353 = torch.ops.aten.reciprocal.default(sqrt_353);  sqrt_353 = None
        mul_1183 = torch.ops.aten.mul.Tensor(reciprocal_353, 1);  reciprocal_353 = None
        unsqueeze_2855 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_2856 = torch.ops.aten.unsqueeze.default(unsqueeze_2855, -1);  unsqueeze_2855 = None
        unsqueeze_2857 = torch.ops.aten.unsqueeze.default(mul_1183, -1);  mul_1183 = None
        unsqueeze_2858 = torch.ops.aten.unsqueeze.default(unsqueeze_2857, -1);  unsqueeze_2857 = None
        sub_353 = torch.ops.aten.sub.Tensor(convolution_353, unsqueeze_2856);  convolution_353 = unsqueeze_2856 = None
        mul_1184 = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_2858);  sub_353 = unsqueeze_2858 = None
        unsqueeze_2859 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_2860 = torch.ops.aten.unsqueeze.default(unsqueeze_2859, -1);  unsqueeze_2859 = None
        mul_1185 = torch.ops.aten.mul.Tensor(mul_1184, unsqueeze_2860);  mul_1184 = unsqueeze_2860 = None
        unsqueeze_2861 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_2862 = torch.ops.aten.unsqueeze.default(unsqueeze_2861, -1);  unsqueeze_2861 = None
        add_1017 = torch.ops.aten.add.Tensor(mul_1185, unsqueeze_2862);  mul_1185 = unsqueeze_2862 = None
        add_1018 = torch.ops.aten.add.Tensor(add_1017, relu_309);  add_1017 = relu_309 = None
        relu_311 = torch.ops.aten.relu.default(add_1018);  add_1018 = None
        convolution_354 = torch.ops.aten.convolution.default(relu_311, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg146_1 = None
        add_1019 = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_354 = torch.ops.aten.sqrt.default(add_1019);  add_1019 = None
        reciprocal_354 = torch.ops.aten.reciprocal.default(sqrt_354);  sqrt_354 = None
        mul_1186 = torch.ops.aten.mul.Tensor(reciprocal_354, 1);  reciprocal_354 = None
        unsqueeze_2863 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_2864 = torch.ops.aten.unsqueeze.default(unsqueeze_2863, -1);  unsqueeze_2863 = None
        unsqueeze_2865 = torch.ops.aten.unsqueeze.default(mul_1186, -1);  mul_1186 = None
        unsqueeze_2866 = torch.ops.aten.unsqueeze.default(unsqueeze_2865, -1);  unsqueeze_2865 = None
        sub_354 = torch.ops.aten.sub.Tensor(convolution_354, unsqueeze_2864);  convolution_354 = unsqueeze_2864 = None
        mul_1187 = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_2866);  sub_354 = unsqueeze_2866 = None
        unsqueeze_2867 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_2868 = torch.ops.aten.unsqueeze.default(unsqueeze_2867, -1);  unsqueeze_2867 = None
        mul_1188 = torch.ops.aten.mul.Tensor(mul_1187, unsqueeze_2868);  mul_1187 = unsqueeze_2868 = None
        unsqueeze_2869 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_2870 = torch.ops.aten.unsqueeze.default(unsqueeze_2869, -1);  unsqueeze_2869 = None
        add_1020 = torch.ops.aten.add.Tensor(mul_1188, unsqueeze_2870);  mul_1188 = unsqueeze_2870 = None
        relu_312 = torch.ops.aten.relu.default(add_1020);  add_1020 = None
        convolution_355 = torch.ops.aten.convolution.default(relu_312, arg151_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_312 = arg151_1 = None
        add_1021 = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_355 = torch.ops.aten.sqrt.default(add_1021);  add_1021 = None
        reciprocal_355 = torch.ops.aten.reciprocal.default(sqrt_355);  sqrt_355 = None
        mul_1189 = torch.ops.aten.mul.Tensor(reciprocal_355, 1);  reciprocal_355 = None
        unsqueeze_2871 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_2872 = torch.ops.aten.unsqueeze.default(unsqueeze_2871, -1);  unsqueeze_2871 = None
        unsqueeze_2873 = torch.ops.aten.unsqueeze.default(mul_1189, -1);  mul_1189 = None
        unsqueeze_2874 = torch.ops.aten.unsqueeze.default(unsqueeze_2873, -1);  unsqueeze_2873 = None
        sub_355 = torch.ops.aten.sub.Tensor(convolution_355, unsqueeze_2872);  convolution_355 = unsqueeze_2872 = None
        mul_1190 = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_2874);  sub_355 = unsqueeze_2874 = None
        unsqueeze_2875 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_2876 = torch.ops.aten.unsqueeze.default(unsqueeze_2875, -1);  unsqueeze_2875 = None
        mul_1191 = torch.ops.aten.mul.Tensor(mul_1190, unsqueeze_2876);  mul_1190 = unsqueeze_2876 = None
        unsqueeze_2877 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_2878 = torch.ops.aten.unsqueeze.default(unsqueeze_2877, -1);  unsqueeze_2877 = None
        add_1022 = torch.ops.aten.add.Tensor(mul_1191, unsqueeze_2878);  mul_1191 = unsqueeze_2878 = None
        add_1023 = torch.ops.aten.add.Tensor(add_1022, relu_311);  add_1022 = relu_311 = None
        relu_313 = torch.ops.aten.relu.default(add_1023);  add_1023 = None
        convolution_356 = torch.ops.aten.convolution.default(relu_313, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg156_1 = None
        add_1024 = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_356 = torch.ops.aten.sqrt.default(add_1024);  add_1024 = None
        reciprocal_356 = torch.ops.aten.reciprocal.default(sqrt_356);  sqrt_356 = None
        mul_1192 = torch.ops.aten.mul.Tensor(reciprocal_356, 1);  reciprocal_356 = None
        unsqueeze_2879 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_2880 = torch.ops.aten.unsqueeze.default(unsqueeze_2879, -1);  unsqueeze_2879 = None
        unsqueeze_2881 = torch.ops.aten.unsqueeze.default(mul_1192, -1);  mul_1192 = None
        unsqueeze_2882 = torch.ops.aten.unsqueeze.default(unsqueeze_2881, -1);  unsqueeze_2881 = None
        sub_356 = torch.ops.aten.sub.Tensor(convolution_356, unsqueeze_2880);  convolution_356 = unsqueeze_2880 = None
        mul_1193 = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_2882);  sub_356 = unsqueeze_2882 = None
        unsqueeze_2883 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_2884 = torch.ops.aten.unsqueeze.default(unsqueeze_2883, -1);  unsqueeze_2883 = None
        mul_1194 = torch.ops.aten.mul.Tensor(mul_1193, unsqueeze_2884);  mul_1193 = unsqueeze_2884 = None
        unsqueeze_2885 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_2886 = torch.ops.aten.unsqueeze.default(unsqueeze_2885, -1);  unsqueeze_2885 = None
        add_1025 = torch.ops.aten.add.Tensor(mul_1194, unsqueeze_2886);  mul_1194 = unsqueeze_2886 = None
        relu_314 = torch.ops.aten.relu.default(add_1025);  add_1025 = None
        convolution_357 = torch.ops.aten.convolution.default(relu_314, arg161_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_314 = arg161_1 = None
        add_1026 = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_357 = torch.ops.aten.sqrt.default(add_1026);  add_1026 = None
        reciprocal_357 = torch.ops.aten.reciprocal.default(sqrt_357);  sqrt_357 = None
        mul_1195 = torch.ops.aten.mul.Tensor(reciprocal_357, 1);  reciprocal_357 = None
        unsqueeze_2887 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_2888 = torch.ops.aten.unsqueeze.default(unsqueeze_2887, -1);  unsqueeze_2887 = None
        unsqueeze_2889 = torch.ops.aten.unsqueeze.default(mul_1195, -1);  mul_1195 = None
        unsqueeze_2890 = torch.ops.aten.unsqueeze.default(unsqueeze_2889, -1);  unsqueeze_2889 = None
        sub_357 = torch.ops.aten.sub.Tensor(convolution_357, unsqueeze_2888);  convolution_357 = unsqueeze_2888 = None
        mul_1196 = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_2890);  sub_357 = unsqueeze_2890 = None
        unsqueeze_2891 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_2892 = torch.ops.aten.unsqueeze.default(unsqueeze_2891, -1);  unsqueeze_2891 = None
        mul_1197 = torch.ops.aten.mul.Tensor(mul_1196, unsqueeze_2892);  mul_1196 = unsqueeze_2892 = None
        unsqueeze_2893 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_2894 = torch.ops.aten.unsqueeze.default(unsqueeze_2893, -1);  unsqueeze_2893 = None
        add_1027 = torch.ops.aten.add.Tensor(mul_1197, unsqueeze_2894);  mul_1197 = unsqueeze_2894 = None
        add_1028 = torch.ops.aten.add.Tensor(add_1027, relu_313);  add_1027 = relu_313 = None
        relu_315 = torch.ops.aten.relu.default(add_1028);  add_1028 = None
        convolution_358 = torch.ops.aten.convolution.default(relu_315, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg166_1 = None
        add_1029 = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_358 = torch.ops.aten.sqrt.default(add_1029);  add_1029 = None
        reciprocal_358 = torch.ops.aten.reciprocal.default(sqrt_358);  sqrt_358 = None
        mul_1198 = torch.ops.aten.mul.Tensor(reciprocal_358, 1);  reciprocal_358 = None
        unsqueeze_2895 = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_2896 = torch.ops.aten.unsqueeze.default(unsqueeze_2895, -1);  unsqueeze_2895 = None
        unsqueeze_2897 = torch.ops.aten.unsqueeze.default(mul_1198, -1);  mul_1198 = None
        unsqueeze_2898 = torch.ops.aten.unsqueeze.default(unsqueeze_2897, -1);  unsqueeze_2897 = None
        sub_358 = torch.ops.aten.sub.Tensor(convolution_358, unsqueeze_2896);  convolution_358 = unsqueeze_2896 = None
        mul_1199 = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_2898);  sub_358 = unsqueeze_2898 = None
        unsqueeze_2899 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_2900 = torch.ops.aten.unsqueeze.default(unsqueeze_2899, -1);  unsqueeze_2899 = None
        mul_1200 = torch.ops.aten.mul.Tensor(mul_1199, unsqueeze_2900);  mul_1199 = unsqueeze_2900 = None
        unsqueeze_2901 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_2902 = torch.ops.aten.unsqueeze.default(unsqueeze_2901, -1);  unsqueeze_2901 = None
        add_1030 = torch.ops.aten.add.Tensor(mul_1200, unsqueeze_2902);  mul_1200 = unsqueeze_2902 = None
        iota_62 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1201 = torch.ops.aten.mul.Tensor(iota_62, 1);  iota_62 = None
        add_1031 = torch.ops.aten.add.Tensor(mul_1201, 0);  mul_1201 = None
        convert_element_type_842 = torch.ops.prims.convert_element_type.default(add_1031, torch.float32);  add_1031 = None
        add_1032 = torch.ops.aten.add.Tensor(convert_element_type_842, 0.0);  convert_element_type_842 = None
        mul_1202 = torch.ops.aten.mul.Tensor(add_1032, 0.5);  add_1032 = None
        convert_element_type_843 = torch.ops.prims.convert_element_type.default(mul_1202, torch.int64);  mul_1202 = None
        unsqueeze_2903 = torch.ops.aten.unsqueeze.default(convert_element_type_843, -1);  convert_element_type_843 = None
        iota_63 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1203 = torch.ops.aten.mul.Tensor(iota_63, 1);  iota_63 = None
        add_1033 = torch.ops.aten.add.Tensor(mul_1203, 0);  mul_1203 = None
        convert_element_type_844 = torch.ops.prims.convert_element_type.default(add_1033, torch.float32);  add_1033 = None
        add_1034 = torch.ops.aten.add.Tensor(convert_element_type_844, 0.0);  convert_element_type_844 = None
        mul_1204 = torch.ops.aten.mul.Tensor(add_1034, 0.5);  add_1034 = None
        convert_element_type_845 = torch.ops.prims.convert_element_type.default(mul_1204, torch.int64);  mul_1204 = None
        _unsafe_index_31 = torch.ops.aten._unsafe_index.Tensor(add_1030, [None, None, unsqueeze_2903, convert_element_type_845]);  add_1030 = unsqueeze_2903 = convert_element_type_845 = None
        add_1035 = torch.ops.aten.add.Tensor(relu_307, _unsafe_index_31);  _unsafe_index_31 = None
        relu_316 = torch.ops.aten.relu.default(add_1035);  add_1035 = None
        convolution_359 = torch.ops.aten.convolution.default(relu_307, arg171_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_307 = arg171_1 = None
        add_1036 = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_359 = torch.ops.aten.sqrt.default(add_1036);  add_1036 = None
        reciprocal_359 = torch.ops.aten.reciprocal.default(sqrt_359);  sqrt_359 = None
        mul_1205 = torch.ops.aten.mul.Tensor(reciprocal_359, 1);  reciprocal_359 = None
        unsqueeze_2904 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_2905 = torch.ops.aten.unsqueeze.default(unsqueeze_2904, -1);  unsqueeze_2904 = None
        unsqueeze_2906 = torch.ops.aten.unsqueeze.default(mul_1205, -1);  mul_1205 = None
        unsqueeze_2907 = torch.ops.aten.unsqueeze.default(unsqueeze_2906, -1);  unsqueeze_2906 = None
        sub_359 = torch.ops.aten.sub.Tensor(convolution_359, unsqueeze_2905);  convolution_359 = unsqueeze_2905 = None
        mul_1206 = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_2907);  sub_359 = unsqueeze_2907 = None
        unsqueeze_2908 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_2909 = torch.ops.aten.unsqueeze.default(unsqueeze_2908, -1);  unsqueeze_2908 = None
        mul_1207 = torch.ops.aten.mul.Tensor(mul_1206, unsqueeze_2909);  mul_1206 = unsqueeze_2909 = None
        unsqueeze_2910 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_2911 = torch.ops.aten.unsqueeze.default(unsqueeze_2910, -1);  unsqueeze_2910 = None
        add_1037 = torch.ops.aten.add.Tensor(mul_1207, unsqueeze_2911);  mul_1207 = unsqueeze_2911 = None
        add_1038 = torch.ops.aten.add.Tensor(add_1037, relu_315);  add_1037 = relu_315 = None
        relu_317 = torch.ops.aten.relu.default(add_1038);  add_1038 = None
        convolution_360 = torch.ops.aten.convolution.default(relu_317, arg176_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg176_1 = None
        add_1039 = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_360 = torch.ops.aten.sqrt.default(add_1039);  add_1039 = None
        reciprocal_360 = torch.ops.aten.reciprocal.default(sqrt_360);  sqrt_360 = None
        mul_1208 = torch.ops.aten.mul.Tensor(reciprocal_360, 1);  reciprocal_360 = None
        unsqueeze_2912 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_2913 = torch.ops.aten.unsqueeze.default(unsqueeze_2912, -1);  unsqueeze_2912 = None
        unsqueeze_2914 = torch.ops.aten.unsqueeze.default(mul_1208, -1);  mul_1208 = None
        unsqueeze_2915 = torch.ops.aten.unsqueeze.default(unsqueeze_2914, -1);  unsqueeze_2914 = None
        sub_360 = torch.ops.aten.sub.Tensor(convolution_360, unsqueeze_2913);  convolution_360 = unsqueeze_2913 = None
        mul_1209 = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_2915);  sub_360 = unsqueeze_2915 = None
        unsqueeze_2916 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_2917 = torch.ops.aten.unsqueeze.default(unsqueeze_2916, -1);  unsqueeze_2916 = None
        mul_1210 = torch.ops.aten.mul.Tensor(mul_1209, unsqueeze_2917);  mul_1209 = unsqueeze_2917 = None
        unsqueeze_2918 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_2919 = torch.ops.aten.unsqueeze.default(unsqueeze_2918, -1);  unsqueeze_2918 = None
        add_1040 = torch.ops.aten.add.Tensor(mul_1210, unsqueeze_2919);  mul_1210 = unsqueeze_2919 = None
        relu_318 = torch.ops.aten.relu.default(add_1040);  add_1040 = None
        convolution_361 = torch.ops.aten.convolution.default(relu_316, arg181_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg181_1 = None
        add_1041 = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_361 = torch.ops.aten.sqrt.default(add_1041);  add_1041 = None
        reciprocal_361 = torch.ops.aten.reciprocal.default(sqrt_361);  sqrt_361 = None
        mul_1211 = torch.ops.aten.mul.Tensor(reciprocal_361, 1);  reciprocal_361 = None
        unsqueeze_2920 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_2921 = torch.ops.aten.unsqueeze.default(unsqueeze_2920, -1);  unsqueeze_2920 = None
        unsqueeze_2922 = torch.ops.aten.unsqueeze.default(mul_1211, -1);  mul_1211 = None
        unsqueeze_2923 = torch.ops.aten.unsqueeze.default(unsqueeze_2922, -1);  unsqueeze_2922 = None
        sub_361 = torch.ops.aten.sub.Tensor(convolution_361, unsqueeze_2921);  convolution_361 = unsqueeze_2921 = None
        mul_1212 = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_2923);  sub_361 = unsqueeze_2923 = None
        unsqueeze_2924 = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_2925 = torch.ops.aten.unsqueeze.default(unsqueeze_2924, -1);  unsqueeze_2924 = None
        mul_1213 = torch.ops.aten.mul.Tensor(mul_1212, unsqueeze_2925);  mul_1212 = unsqueeze_2925 = None
        unsqueeze_2926 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_2927 = torch.ops.aten.unsqueeze.default(unsqueeze_2926, -1);  unsqueeze_2926 = None
        add_1042 = torch.ops.aten.add.Tensor(mul_1213, unsqueeze_2927);  mul_1213 = unsqueeze_2927 = None
        relu_319 = torch.ops.aten.relu.default(add_1042);  add_1042 = None
        convolution_362 = torch.ops.aten.convolution.default(relu_319, arg186_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_319 = arg186_1 = None
        add_1043 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_362 = torch.ops.aten.sqrt.default(add_1043);  add_1043 = None
        reciprocal_362 = torch.ops.aten.reciprocal.default(sqrt_362);  sqrt_362 = None
        mul_1214 = torch.ops.aten.mul.Tensor(reciprocal_362, 1);  reciprocal_362 = None
        unsqueeze_2928 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_2929 = torch.ops.aten.unsqueeze.default(unsqueeze_2928, -1);  unsqueeze_2928 = None
        unsqueeze_2930 = torch.ops.aten.unsqueeze.default(mul_1214, -1);  mul_1214 = None
        unsqueeze_2931 = torch.ops.aten.unsqueeze.default(unsqueeze_2930, -1);  unsqueeze_2930 = None
        sub_362 = torch.ops.aten.sub.Tensor(convolution_362, unsqueeze_2929);  convolution_362 = unsqueeze_2929 = None
        mul_1215 = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_2931);  sub_362 = unsqueeze_2931 = None
        unsqueeze_2932 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_2933 = torch.ops.aten.unsqueeze.default(unsqueeze_2932, -1);  unsqueeze_2932 = None
        mul_1216 = torch.ops.aten.mul.Tensor(mul_1215, unsqueeze_2933);  mul_1215 = unsqueeze_2933 = None
        unsqueeze_2934 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_2935 = torch.ops.aten.unsqueeze.default(unsqueeze_2934, -1);  unsqueeze_2934 = None
        add_1044 = torch.ops.aten.add.Tensor(mul_1216, unsqueeze_2935);  mul_1216 = unsqueeze_2935 = None
        add_1045 = torch.ops.aten.add.Tensor(add_1044, relu_316);  add_1044 = relu_316 = None
        relu_320 = torch.ops.aten.relu.default(add_1045);  add_1045 = None
        convolution_363 = torch.ops.aten.convolution.default(relu_320, arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg191_1 = None
        add_1046 = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_363 = torch.ops.aten.sqrt.default(add_1046);  add_1046 = None
        reciprocal_363 = torch.ops.aten.reciprocal.default(sqrt_363);  sqrt_363 = None
        mul_1217 = torch.ops.aten.mul.Tensor(reciprocal_363, 1);  reciprocal_363 = None
        unsqueeze_2936 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_2937 = torch.ops.aten.unsqueeze.default(unsqueeze_2936, -1);  unsqueeze_2936 = None
        unsqueeze_2938 = torch.ops.aten.unsqueeze.default(mul_1217, -1);  mul_1217 = None
        unsqueeze_2939 = torch.ops.aten.unsqueeze.default(unsqueeze_2938, -1);  unsqueeze_2938 = None
        sub_363 = torch.ops.aten.sub.Tensor(convolution_363, unsqueeze_2937);  convolution_363 = unsqueeze_2937 = None
        mul_1218 = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_2939);  sub_363 = unsqueeze_2939 = None
        unsqueeze_2940 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_2941 = torch.ops.aten.unsqueeze.default(unsqueeze_2940, -1);  unsqueeze_2940 = None
        mul_1219 = torch.ops.aten.mul.Tensor(mul_1218, unsqueeze_2941);  mul_1218 = unsqueeze_2941 = None
        unsqueeze_2942 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_2943 = torch.ops.aten.unsqueeze.default(unsqueeze_2942, -1);  unsqueeze_2942 = None
        add_1047 = torch.ops.aten.add.Tensor(mul_1219, unsqueeze_2943);  mul_1219 = unsqueeze_2943 = None
        relu_321 = torch.ops.aten.relu.default(add_1047);  add_1047 = None
        convolution_364 = torch.ops.aten.convolution.default(relu_321, arg196_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_321 = arg196_1 = None
        add_1048 = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_364 = torch.ops.aten.sqrt.default(add_1048);  add_1048 = None
        reciprocal_364 = torch.ops.aten.reciprocal.default(sqrt_364);  sqrt_364 = None
        mul_1220 = torch.ops.aten.mul.Tensor(reciprocal_364, 1);  reciprocal_364 = None
        unsqueeze_2944 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_2945 = torch.ops.aten.unsqueeze.default(unsqueeze_2944, -1);  unsqueeze_2944 = None
        unsqueeze_2946 = torch.ops.aten.unsqueeze.default(mul_1220, -1);  mul_1220 = None
        unsqueeze_2947 = torch.ops.aten.unsqueeze.default(unsqueeze_2946, -1);  unsqueeze_2946 = None
        sub_364 = torch.ops.aten.sub.Tensor(convolution_364, unsqueeze_2945);  convolution_364 = unsqueeze_2945 = None
        mul_1221 = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_2947);  sub_364 = unsqueeze_2947 = None
        unsqueeze_2948 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_2949 = torch.ops.aten.unsqueeze.default(unsqueeze_2948, -1);  unsqueeze_2948 = None
        mul_1222 = torch.ops.aten.mul.Tensor(mul_1221, unsqueeze_2949);  mul_1221 = unsqueeze_2949 = None
        unsqueeze_2950 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_2951 = torch.ops.aten.unsqueeze.default(unsqueeze_2950, -1);  unsqueeze_2950 = None
        add_1049 = torch.ops.aten.add.Tensor(mul_1222, unsqueeze_2951);  mul_1222 = unsqueeze_2951 = None
        add_1050 = torch.ops.aten.add.Tensor(add_1049, relu_320);  add_1049 = relu_320 = None
        relu_322 = torch.ops.aten.relu.default(add_1050);  add_1050 = None
        convolution_365 = torch.ops.aten.convolution.default(relu_322, arg201_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg201_1 = None
        add_1051 = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_365 = torch.ops.aten.sqrt.default(add_1051);  add_1051 = None
        reciprocal_365 = torch.ops.aten.reciprocal.default(sqrt_365);  sqrt_365 = None
        mul_1223 = torch.ops.aten.mul.Tensor(reciprocal_365, 1);  reciprocal_365 = None
        unsqueeze_2952 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_2953 = torch.ops.aten.unsqueeze.default(unsqueeze_2952, -1);  unsqueeze_2952 = None
        unsqueeze_2954 = torch.ops.aten.unsqueeze.default(mul_1223, -1);  mul_1223 = None
        unsqueeze_2955 = torch.ops.aten.unsqueeze.default(unsqueeze_2954, -1);  unsqueeze_2954 = None
        sub_365 = torch.ops.aten.sub.Tensor(convolution_365, unsqueeze_2953);  convolution_365 = unsqueeze_2953 = None
        mul_1224 = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_2955);  sub_365 = unsqueeze_2955 = None
        unsqueeze_2956 = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_2957 = torch.ops.aten.unsqueeze.default(unsqueeze_2956, -1);  unsqueeze_2956 = None
        mul_1225 = torch.ops.aten.mul.Tensor(mul_1224, unsqueeze_2957);  mul_1224 = unsqueeze_2957 = None
        unsqueeze_2958 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_2959 = torch.ops.aten.unsqueeze.default(unsqueeze_2958, -1);  unsqueeze_2958 = None
        add_1052 = torch.ops.aten.add.Tensor(mul_1225, unsqueeze_2959);  mul_1225 = unsqueeze_2959 = None
        relu_323 = torch.ops.aten.relu.default(add_1052);  add_1052 = None
        convolution_366 = torch.ops.aten.convolution.default(relu_323, arg206_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_323 = arg206_1 = None
        add_1053 = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_366 = torch.ops.aten.sqrt.default(add_1053);  add_1053 = None
        reciprocal_366 = torch.ops.aten.reciprocal.default(sqrt_366);  sqrt_366 = None
        mul_1226 = torch.ops.aten.mul.Tensor(reciprocal_366, 1);  reciprocal_366 = None
        unsqueeze_2960 = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_2961 = torch.ops.aten.unsqueeze.default(unsqueeze_2960, -1);  unsqueeze_2960 = None
        unsqueeze_2962 = torch.ops.aten.unsqueeze.default(mul_1226, -1);  mul_1226 = None
        unsqueeze_2963 = torch.ops.aten.unsqueeze.default(unsqueeze_2962, -1);  unsqueeze_2962 = None
        sub_366 = torch.ops.aten.sub.Tensor(convolution_366, unsqueeze_2961);  convolution_366 = unsqueeze_2961 = None
        mul_1227 = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_2963);  sub_366 = unsqueeze_2963 = None
        unsqueeze_2964 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_2965 = torch.ops.aten.unsqueeze.default(unsqueeze_2964, -1);  unsqueeze_2964 = None
        mul_1228 = torch.ops.aten.mul.Tensor(mul_1227, unsqueeze_2965);  mul_1227 = unsqueeze_2965 = None
        unsqueeze_2966 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_2967 = torch.ops.aten.unsqueeze.default(unsqueeze_2966, -1);  unsqueeze_2966 = None
        add_1054 = torch.ops.aten.add.Tensor(mul_1228, unsqueeze_2967);  mul_1228 = unsqueeze_2967 = None
        add_1055 = torch.ops.aten.add.Tensor(add_1054, relu_322);  add_1054 = relu_322 = None
        relu_324 = torch.ops.aten.relu.default(add_1055);  add_1055 = None
        convolution_367 = torch.ops.aten.convolution.default(relu_324, arg211_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg211_1 = None
        add_1056 = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_367 = torch.ops.aten.sqrt.default(add_1056);  add_1056 = None
        reciprocal_367 = torch.ops.aten.reciprocal.default(sqrt_367);  sqrt_367 = None
        mul_1229 = torch.ops.aten.mul.Tensor(reciprocal_367, 1);  reciprocal_367 = None
        unsqueeze_2968 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_2969 = torch.ops.aten.unsqueeze.default(unsqueeze_2968, -1);  unsqueeze_2968 = None
        unsqueeze_2970 = torch.ops.aten.unsqueeze.default(mul_1229, -1);  mul_1229 = None
        unsqueeze_2971 = torch.ops.aten.unsqueeze.default(unsqueeze_2970, -1);  unsqueeze_2970 = None
        sub_367 = torch.ops.aten.sub.Tensor(convolution_367, unsqueeze_2969);  convolution_367 = unsqueeze_2969 = None
        mul_1230 = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_2971);  sub_367 = unsqueeze_2971 = None
        unsqueeze_2972 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_2973 = torch.ops.aten.unsqueeze.default(unsqueeze_2972, -1);  unsqueeze_2972 = None
        mul_1231 = torch.ops.aten.mul.Tensor(mul_1230, unsqueeze_2973);  mul_1230 = unsqueeze_2973 = None
        unsqueeze_2974 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_2975 = torch.ops.aten.unsqueeze.default(unsqueeze_2974, -1);  unsqueeze_2974 = None
        add_1057 = torch.ops.aten.add.Tensor(mul_1231, unsqueeze_2975);  mul_1231 = unsqueeze_2975 = None
        relu_325 = torch.ops.aten.relu.default(add_1057);  add_1057 = None
        convolution_368 = torch.ops.aten.convolution.default(relu_325, arg216_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_325 = arg216_1 = None
        add_1058 = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_368 = torch.ops.aten.sqrt.default(add_1058);  add_1058 = None
        reciprocal_368 = torch.ops.aten.reciprocal.default(sqrt_368);  sqrt_368 = None
        mul_1232 = torch.ops.aten.mul.Tensor(reciprocal_368, 1);  reciprocal_368 = None
        unsqueeze_2976 = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_2977 = torch.ops.aten.unsqueeze.default(unsqueeze_2976, -1);  unsqueeze_2976 = None
        unsqueeze_2978 = torch.ops.aten.unsqueeze.default(mul_1232, -1);  mul_1232 = None
        unsqueeze_2979 = torch.ops.aten.unsqueeze.default(unsqueeze_2978, -1);  unsqueeze_2978 = None
        sub_368 = torch.ops.aten.sub.Tensor(convolution_368, unsqueeze_2977);  convolution_368 = unsqueeze_2977 = None
        mul_1233 = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_2979);  sub_368 = unsqueeze_2979 = None
        unsqueeze_2980 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_2981 = torch.ops.aten.unsqueeze.default(unsqueeze_2980, -1);  unsqueeze_2980 = None
        mul_1234 = torch.ops.aten.mul.Tensor(mul_1233, unsqueeze_2981);  mul_1233 = unsqueeze_2981 = None
        unsqueeze_2982 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_2983 = torch.ops.aten.unsqueeze.default(unsqueeze_2982, -1);  unsqueeze_2982 = None
        add_1059 = torch.ops.aten.add.Tensor(mul_1234, unsqueeze_2983);  mul_1234 = unsqueeze_2983 = None
        add_1060 = torch.ops.aten.add.Tensor(add_1059, relu_324);  add_1059 = relu_324 = None
        relu_326 = torch.ops.aten.relu.default(add_1060);  add_1060 = None
        convolution_369 = torch.ops.aten.convolution.default(relu_317, arg221_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg221_1 = None
        add_1061 = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_369 = torch.ops.aten.sqrt.default(add_1061);  add_1061 = None
        reciprocal_369 = torch.ops.aten.reciprocal.default(sqrt_369);  sqrt_369 = None
        mul_1235 = torch.ops.aten.mul.Tensor(reciprocal_369, 1);  reciprocal_369 = None
        unsqueeze_2984 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_2985 = torch.ops.aten.unsqueeze.default(unsqueeze_2984, -1);  unsqueeze_2984 = None
        unsqueeze_2986 = torch.ops.aten.unsqueeze.default(mul_1235, -1);  mul_1235 = None
        unsqueeze_2987 = torch.ops.aten.unsqueeze.default(unsqueeze_2986, -1);  unsqueeze_2986 = None
        sub_369 = torch.ops.aten.sub.Tensor(convolution_369, unsqueeze_2985);  convolution_369 = unsqueeze_2985 = None
        mul_1236 = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_2987);  sub_369 = unsqueeze_2987 = None
        unsqueeze_2988 = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_2989 = torch.ops.aten.unsqueeze.default(unsqueeze_2988, -1);  unsqueeze_2988 = None
        mul_1237 = torch.ops.aten.mul.Tensor(mul_1236, unsqueeze_2989);  mul_1236 = unsqueeze_2989 = None
        unsqueeze_2990 = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_2991 = torch.ops.aten.unsqueeze.default(unsqueeze_2990, -1);  unsqueeze_2990 = None
        add_1062 = torch.ops.aten.add.Tensor(mul_1237, unsqueeze_2991);  mul_1237 = unsqueeze_2991 = None
        relu_327 = torch.ops.aten.relu.default(add_1062);  add_1062 = None
        convolution_370 = torch.ops.aten.convolution.default(relu_327, arg226_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_327 = arg226_1 = None
        add_1063 = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_370 = torch.ops.aten.sqrt.default(add_1063);  add_1063 = None
        reciprocal_370 = torch.ops.aten.reciprocal.default(sqrt_370);  sqrt_370 = None
        mul_1238 = torch.ops.aten.mul.Tensor(reciprocal_370, 1);  reciprocal_370 = None
        unsqueeze_2992 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_2993 = torch.ops.aten.unsqueeze.default(unsqueeze_2992, -1);  unsqueeze_2992 = None
        unsqueeze_2994 = torch.ops.aten.unsqueeze.default(mul_1238, -1);  mul_1238 = None
        unsqueeze_2995 = torch.ops.aten.unsqueeze.default(unsqueeze_2994, -1);  unsqueeze_2994 = None
        sub_370 = torch.ops.aten.sub.Tensor(convolution_370, unsqueeze_2993);  convolution_370 = unsqueeze_2993 = None
        mul_1239 = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_2995);  sub_370 = unsqueeze_2995 = None
        unsqueeze_2996 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_2997 = torch.ops.aten.unsqueeze.default(unsqueeze_2996, -1);  unsqueeze_2996 = None
        mul_1240 = torch.ops.aten.mul.Tensor(mul_1239, unsqueeze_2997);  mul_1239 = unsqueeze_2997 = None
        unsqueeze_2998 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_2999 = torch.ops.aten.unsqueeze.default(unsqueeze_2998, -1);  unsqueeze_2998 = None
        add_1064 = torch.ops.aten.add.Tensor(mul_1240, unsqueeze_2999);  mul_1240 = unsqueeze_2999 = None
        add_1065 = torch.ops.aten.add.Tensor(add_1064, relu_317);  add_1064 = relu_317 = None
        relu_328 = torch.ops.aten.relu.default(add_1065);  add_1065 = None
        convolution_371 = torch.ops.aten.convolution.default(relu_328, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg231_1 = None
        add_1066 = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_371 = torch.ops.aten.sqrt.default(add_1066);  add_1066 = None
        reciprocal_371 = torch.ops.aten.reciprocal.default(sqrt_371);  sqrt_371 = None
        mul_1241 = torch.ops.aten.mul.Tensor(reciprocal_371, 1);  reciprocal_371 = None
        unsqueeze_3000 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_3001 = torch.ops.aten.unsqueeze.default(unsqueeze_3000, -1);  unsqueeze_3000 = None
        unsqueeze_3002 = torch.ops.aten.unsqueeze.default(mul_1241, -1);  mul_1241 = None
        unsqueeze_3003 = torch.ops.aten.unsqueeze.default(unsqueeze_3002, -1);  unsqueeze_3002 = None
        sub_371 = torch.ops.aten.sub.Tensor(convolution_371, unsqueeze_3001);  convolution_371 = unsqueeze_3001 = None
        mul_1242 = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_3003);  sub_371 = unsqueeze_3003 = None
        unsqueeze_3004 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_3005 = torch.ops.aten.unsqueeze.default(unsqueeze_3004, -1);  unsqueeze_3004 = None
        mul_1243 = torch.ops.aten.mul.Tensor(mul_1242, unsqueeze_3005);  mul_1242 = unsqueeze_3005 = None
        unsqueeze_3006 = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_3007 = torch.ops.aten.unsqueeze.default(unsqueeze_3006, -1);  unsqueeze_3006 = None
        add_1067 = torch.ops.aten.add.Tensor(mul_1243, unsqueeze_3007);  mul_1243 = unsqueeze_3007 = None
        relu_329 = torch.ops.aten.relu.default(add_1067);  add_1067 = None
        convolution_372 = torch.ops.aten.convolution.default(relu_329, arg236_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_329 = arg236_1 = None
        add_1068 = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_372 = torch.ops.aten.sqrt.default(add_1068);  add_1068 = None
        reciprocal_372 = torch.ops.aten.reciprocal.default(sqrt_372);  sqrt_372 = None
        mul_1244 = torch.ops.aten.mul.Tensor(reciprocal_372, 1);  reciprocal_372 = None
        unsqueeze_3008 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_3009 = torch.ops.aten.unsqueeze.default(unsqueeze_3008, -1);  unsqueeze_3008 = None
        unsqueeze_3010 = torch.ops.aten.unsqueeze.default(mul_1244, -1);  mul_1244 = None
        unsqueeze_3011 = torch.ops.aten.unsqueeze.default(unsqueeze_3010, -1);  unsqueeze_3010 = None
        sub_372 = torch.ops.aten.sub.Tensor(convolution_372, unsqueeze_3009);  convolution_372 = unsqueeze_3009 = None
        mul_1245 = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_3011);  sub_372 = unsqueeze_3011 = None
        unsqueeze_3012 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_3013 = torch.ops.aten.unsqueeze.default(unsqueeze_3012, -1);  unsqueeze_3012 = None
        mul_1246 = torch.ops.aten.mul.Tensor(mul_1245, unsqueeze_3013);  mul_1245 = unsqueeze_3013 = None
        unsqueeze_3014 = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_3015 = torch.ops.aten.unsqueeze.default(unsqueeze_3014, -1);  unsqueeze_3014 = None
        add_1069 = torch.ops.aten.add.Tensor(mul_1246, unsqueeze_3015);  mul_1246 = unsqueeze_3015 = None
        add_1070 = torch.ops.aten.add.Tensor(add_1069, relu_328);  add_1069 = relu_328 = None
        relu_330 = torch.ops.aten.relu.default(add_1070);  add_1070 = None
        convolution_373 = torch.ops.aten.convolution.default(relu_330, arg241_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg241_1 = None
        add_1071 = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_373 = torch.ops.aten.sqrt.default(add_1071);  add_1071 = None
        reciprocal_373 = torch.ops.aten.reciprocal.default(sqrt_373);  sqrt_373 = None
        mul_1247 = torch.ops.aten.mul.Tensor(reciprocal_373, 1);  reciprocal_373 = None
        unsqueeze_3016 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_3017 = torch.ops.aten.unsqueeze.default(unsqueeze_3016, -1);  unsqueeze_3016 = None
        unsqueeze_3018 = torch.ops.aten.unsqueeze.default(mul_1247, -1);  mul_1247 = None
        unsqueeze_3019 = torch.ops.aten.unsqueeze.default(unsqueeze_3018, -1);  unsqueeze_3018 = None
        sub_373 = torch.ops.aten.sub.Tensor(convolution_373, unsqueeze_3017);  convolution_373 = unsqueeze_3017 = None
        mul_1248 = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_3019);  sub_373 = unsqueeze_3019 = None
        unsqueeze_3020 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_3021 = torch.ops.aten.unsqueeze.default(unsqueeze_3020, -1);  unsqueeze_3020 = None
        mul_1249 = torch.ops.aten.mul.Tensor(mul_1248, unsqueeze_3021);  mul_1248 = unsqueeze_3021 = None
        unsqueeze_3022 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_3023 = torch.ops.aten.unsqueeze.default(unsqueeze_3022, -1);  unsqueeze_3022 = None
        add_1072 = torch.ops.aten.add.Tensor(mul_1249, unsqueeze_3023);  mul_1249 = unsqueeze_3023 = None
        relu_331 = torch.ops.aten.relu.default(add_1072);  add_1072 = None
        convolution_374 = torch.ops.aten.convolution.default(relu_331, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_331 = arg246_1 = None
        add_1073 = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_374 = torch.ops.aten.sqrt.default(add_1073);  add_1073 = None
        reciprocal_374 = torch.ops.aten.reciprocal.default(sqrt_374);  sqrt_374 = None
        mul_1250 = torch.ops.aten.mul.Tensor(reciprocal_374, 1);  reciprocal_374 = None
        unsqueeze_3024 = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_3025 = torch.ops.aten.unsqueeze.default(unsqueeze_3024, -1);  unsqueeze_3024 = None
        unsqueeze_3026 = torch.ops.aten.unsqueeze.default(mul_1250, -1);  mul_1250 = None
        unsqueeze_3027 = torch.ops.aten.unsqueeze.default(unsqueeze_3026, -1);  unsqueeze_3026 = None
        sub_374 = torch.ops.aten.sub.Tensor(convolution_374, unsqueeze_3025);  convolution_374 = unsqueeze_3025 = None
        mul_1251 = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_3027);  sub_374 = unsqueeze_3027 = None
        unsqueeze_3028 = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_3029 = torch.ops.aten.unsqueeze.default(unsqueeze_3028, -1);  unsqueeze_3028 = None
        mul_1252 = torch.ops.aten.mul.Tensor(mul_1251, unsqueeze_3029);  mul_1251 = unsqueeze_3029 = None
        unsqueeze_3030 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_3031 = torch.ops.aten.unsqueeze.default(unsqueeze_3030, -1);  unsqueeze_3030 = None
        add_1074 = torch.ops.aten.add.Tensor(mul_1252, unsqueeze_3031);  mul_1252 = unsqueeze_3031 = None
        add_1075 = torch.ops.aten.add.Tensor(add_1074, relu_330);  add_1074 = relu_330 = None
        relu_332 = torch.ops.aten.relu.default(add_1075);  add_1075 = None
        convolution_375 = torch.ops.aten.convolution.default(relu_332, arg251_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg251_1 = None
        add_1076 = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_375 = torch.ops.aten.sqrt.default(add_1076);  add_1076 = None
        reciprocal_375 = torch.ops.aten.reciprocal.default(sqrt_375);  sqrt_375 = None
        mul_1253 = torch.ops.aten.mul.Tensor(reciprocal_375, 1);  reciprocal_375 = None
        unsqueeze_3032 = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_3033 = torch.ops.aten.unsqueeze.default(unsqueeze_3032, -1);  unsqueeze_3032 = None
        unsqueeze_3034 = torch.ops.aten.unsqueeze.default(mul_1253, -1);  mul_1253 = None
        unsqueeze_3035 = torch.ops.aten.unsqueeze.default(unsqueeze_3034, -1);  unsqueeze_3034 = None
        sub_375 = torch.ops.aten.sub.Tensor(convolution_375, unsqueeze_3033);  convolution_375 = unsqueeze_3033 = None
        mul_1254 = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_3035);  sub_375 = unsqueeze_3035 = None
        unsqueeze_3036 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_3037 = torch.ops.aten.unsqueeze.default(unsqueeze_3036, -1);  unsqueeze_3036 = None
        mul_1255 = torch.ops.aten.mul.Tensor(mul_1254, unsqueeze_3037);  mul_1254 = unsqueeze_3037 = None
        unsqueeze_3038 = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_3039 = torch.ops.aten.unsqueeze.default(unsqueeze_3038, -1);  unsqueeze_3038 = None
        add_1077 = torch.ops.aten.add.Tensor(mul_1255, unsqueeze_3039);  mul_1255 = unsqueeze_3039 = None
        relu_333 = torch.ops.aten.relu.default(add_1077);  add_1077 = None
        convolution_376 = torch.ops.aten.convolution.default(relu_333, arg256_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_333 = arg256_1 = None
        add_1078 = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_376 = torch.ops.aten.sqrt.default(add_1078);  add_1078 = None
        reciprocal_376 = torch.ops.aten.reciprocal.default(sqrt_376);  sqrt_376 = None
        mul_1256 = torch.ops.aten.mul.Tensor(reciprocal_376, 1);  reciprocal_376 = None
        unsqueeze_3040 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_3041 = torch.ops.aten.unsqueeze.default(unsqueeze_3040, -1);  unsqueeze_3040 = None
        unsqueeze_3042 = torch.ops.aten.unsqueeze.default(mul_1256, -1);  mul_1256 = None
        unsqueeze_3043 = torch.ops.aten.unsqueeze.default(unsqueeze_3042, -1);  unsqueeze_3042 = None
        sub_376 = torch.ops.aten.sub.Tensor(convolution_376, unsqueeze_3041);  convolution_376 = unsqueeze_3041 = None
        mul_1257 = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_3043);  sub_376 = unsqueeze_3043 = None
        unsqueeze_3044 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_3045 = torch.ops.aten.unsqueeze.default(unsqueeze_3044, -1);  unsqueeze_3044 = None
        mul_1258 = torch.ops.aten.mul.Tensor(mul_1257, unsqueeze_3045);  mul_1257 = unsqueeze_3045 = None
        unsqueeze_3046 = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_3047 = torch.ops.aten.unsqueeze.default(unsqueeze_3046, -1);  unsqueeze_3046 = None
        add_1079 = torch.ops.aten.add.Tensor(mul_1258, unsqueeze_3047);  mul_1258 = unsqueeze_3047 = None
        add_1080 = torch.ops.aten.add.Tensor(add_1079, relu_332);  add_1079 = relu_332 = None
        relu_334 = torch.ops.aten.relu.default(add_1080);  add_1080 = None
        convolution_377 = torch.ops.aten.convolution.default(relu_318, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg261_1 = None
        add_1081 = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_377 = torch.ops.aten.sqrt.default(add_1081);  add_1081 = None
        reciprocal_377 = torch.ops.aten.reciprocal.default(sqrt_377);  sqrt_377 = None
        mul_1259 = torch.ops.aten.mul.Tensor(reciprocal_377, 1);  reciprocal_377 = None
        unsqueeze_3048 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_3049 = torch.ops.aten.unsqueeze.default(unsqueeze_3048, -1);  unsqueeze_3048 = None
        unsqueeze_3050 = torch.ops.aten.unsqueeze.default(mul_1259, -1);  mul_1259 = None
        unsqueeze_3051 = torch.ops.aten.unsqueeze.default(unsqueeze_3050, -1);  unsqueeze_3050 = None
        sub_377 = torch.ops.aten.sub.Tensor(convolution_377, unsqueeze_3049);  convolution_377 = unsqueeze_3049 = None
        mul_1260 = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_3051);  sub_377 = unsqueeze_3051 = None
        unsqueeze_3052 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_3053 = torch.ops.aten.unsqueeze.default(unsqueeze_3052, -1);  unsqueeze_3052 = None
        mul_1261 = torch.ops.aten.mul.Tensor(mul_1260, unsqueeze_3053);  mul_1260 = unsqueeze_3053 = None
        unsqueeze_3054 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_3055 = torch.ops.aten.unsqueeze.default(unsqueeze_3054, -1);  unsqueeze_3054 = None
        add_1082 = torch.ops.aten.add.Tensor(mul_1261, unsqueeze_3055);  mul_1261 = unsqueeze_3055 = None
        relu_335 = torch.ops.aten.relu.default(add_1082);  add_1082 = None
        convolution_378 = torch.ops.aten.convolution.default(relu_335, arg266_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_335 = arg266_1 = None
        add_1083 = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
        sqrt_378 = torch.ops.aten.sqrt.default(add_1083);  add_1083 = None
        reciprocal_378 = torch.ops.aten.reciprocal.default(sqrt_378);  sqrt_378 = None
        mul_1262 = torch.ops.aten.mul.Tensor(reciprocal_378, 1);  reciprocal_378 = None
        unsqueeze_3056 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_3057 = torch.ops.aten.unsqueeze.default(unsqueeze_3056, -1);  unsqueeze_3056 = None
        unsqueeze_3058 = torch.ops.aten.unsqueeze.default(mul_1262, -1);  mul_1262 = None
        unsqueeze_3059 = torch.ops.aten.unsqueeze.default(unsqueeze_3058, -1);  unsqueeze_3058 = None
        sub_378 = torch.ops.aten.sub.Tensor(convolution_378, unsqueeze_3057);  convolution_378 = unsqueeze_3057 = None
        mul_1263 = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_3059);  sub_378 = unsqueeze_3059 = None
        unsqueeze_3060 = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_3061 = torch.ops.aten.unsqueeze.default(unsqueeze_3060, -1);  unsqueeze_3060 = None
        mul_1264 = torch.ops.aten.mul.Tensor(mul_1263, unsqueeze_3061);  mul_1263 = unsqueeze_3061 = None
        unsqueeze_3062 = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_3063 = torch.ops.aten.unsqueeze.default(unsqueeze_3062, -1);  unsqueeze_3062 = None
        add_1084 = torch.ops.aten.add.Tensor(mul_1264, unsqueeze_3063);  mul_1264 = unsqueeze_3063 = None
        add_1085 = torch.ops.aten.add.Tensor(add_1084, relu_318);  add_1084 = relu_318 = None
        relu_336 = torch.ops.aten.relu.default(add_1085);  add_1085 = None
        convolution_379 = torch.ops.aten.convolution.default(relu_336, arg271_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg271_1 = None
        add_1086 = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_379 = torch.ops.aten.sqrt.default(add_1086);  add_1086 = None
        reciprocal_379 = torch.ops.aten.reciprocal.default(sqrt_379);  sqrt_379 = None
        mul_1265 = torch.ops.aten.mul.Tensor(reciprocal_379, 1);  reciprocal_379 = None
        unsqueeze_3064 = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_3065 = torch.ops.aten.unsqueeze.default(unsqueeze_3064, -1);  unsqueeze_3064 = None
        unsqueeze_3066 = torch.ops.aten.unsqueeze.default(mul_1265, -1);  mul_1265 = None
        unsqueeze_3067 = torch.ops.aten.unsqueeze.default(unsqueeze_3066, -1);  unsqueeze_3066 = None
        sub_379 = torch.ops.aten.sub.Tensor(convolution_379, unsqueeze_3065);  convolution_379 = unsqueeze_3065 = None
        mul_1266 = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_3067);  sub_379 = unsqueeze_3067 = None
        unsqueeze_3068 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_3069 = torch.ops.aten.unsqueeze.default(unsqueeze_3068, -1);  unsqueeze_3068 = None
        mul_1267 = torch.ops.aten.mul.Tensor(mul_1266, unsqueeze_3069);  mul_1266 = unsqueeze_3069 = None
        unsqueeze_3070 = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_3071 = torch.ops.aten.unsqueeze.default(unsqueeze_3070, -1);  unsqueeze_3070 = None
        add_1087 = torch.ops.aten.add.Tensor(mul_1267, unsqueeze_3071);  mul_1267 = unsqueeze_3071 = None
        relu_337 = torch.ops.aten.relu.default(add_1087);  add_1087 = None
        convolution_380 = torch.ops.aten.convolution.default(relu_337, arg276_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_337 = arg276_1 = None
        add_1088 = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
        sqrt_380 = torch.ops.aten.sqrt.default(add_1088);  add_1088 = None
        reciprocal_380 = torch.ops.aten.reciprocal.default(sqrt_380);  sqrt_380 = None
        mul_1268 = torch.ops.aten.mul.Tensor(reciprocal_380, 1);  reciprocal_380 = None
        unsqueeze_3072 = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_3073 = torch.ops.aten.unsqueeze.default(unsqueeze_3072, -1);  unsqueeze_3072 = None
        unsqueeze_3074 = torch.ops.aten.unsqueeze.default(mul_1268, -1);  mul_1268 = None
        unsqueeze_3075 = torch.ops.aten.unsqueeze.default(unsqueeze_3074, -1);  unsqueeze_3074 = None
        sub_380 = torch.ops.aten.sub.Tensor(convolution_380, unsqueeze_3073);  convolution_380 = unsqueeze_3073 = None
        mul_1269 = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_3075);  sub_380 = unsqueeze_3075 = None
        unsqueeze_3076 = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_3077 = torch.ops.aten.unsqueeze.default(unsqueeze_3076, -1);  unsqueeze_3076 = None
        mul_1270 = torch.ops.aten.mul.Tensor(mul_1269, unsqueeze_3077);  mul_1269 = unsqueeze_3077 = None
        unsqueeze_3078 = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_3079 = torch.ops.aten.unsqueeze.default(unsqueeze_3078, -1);  unsqueeze_3078 = None
        add_1089 = torch.ops.aten.add.Tensor(mul_1270, unsqueeze_3079);  mul_1270 = unsqueeze_3079 = None
        add_1090 = torch.ops.aten.add.Tensor(add_1089, relu_336);  add_1089 = relu_336 = None
        relu_338 = torch.ops.aten.relu.default(add_1090);  add_1090 = None
        convolution_381 = torch.ops.aten.convolution.default(relu_338, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg281_1 = None
        add_1091 = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_381 = torch.ops.aten.sqrt.default(add_1091);  add_1091 = None
        reciprocal_381 = torch.ops.aten.reciprocal.default(sqrt_381);  sqrt_381 = None
        mul_1271 = torch.ops.aten.mul.Tensor(reciprocal_381, 1);  reciprocal_381 = None
        unsqueeze_3080 = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_3081 = torch.ops.aten.unsqueeze.default(unsqueeze_3080, -1);  unsqueeze_3080 = None
        unsqueeze_3082 = torch.ops.aten.unsqueeze.default(mul_1271, -1);  mul_1271 = None
        unsqueeze_3083 = torch.ops.aten.unsqueeze.default(unsqueeze_3082, -1);  unsqueeze_3082 = None
        sub_381 = torch.ops.aten.sub.Tensor(convolution_381, unsqueeze_3081);  convolution_381 = unsqueeze_3081 = None
        mul_1272 = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_3083);  sub_381 = unsqueeze_3083 = None
        unsqueeze_3084 = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_3085 = torch.ops.aten.unsqueeze.default(unsqueeze_3084, -1);  unsqueeze_3084 = None
        mul_1273 = torch.ops.aten.mul.Tensor(mul_1272, unsqueeze_3085);  mul_1272 = unsqueeze_3085 = None
        unsqueeze_3086 = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_3087 = torch.ops.aten.unsqueeze.default(unsqueeze_3086, -1);  unsqueeze_3086 = None
        add_1092 = torch.ops.aten.add.Tensor(mul_1273, unsqueeze_3087);  mul_1273 = unsqueeze_3087 = None
        relu_339 = torch.ops.aten.relu.default(add_1092);  add_1092 = None
        convolution_382 = torch.ops.aten.convolution.default(relu_339, arg286_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_339 = arg286_1 = None
        add_1093 = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_382 = torch.ops.aten.sqrt.default(add_1093);  add_1093 = None
        reciprocal_382 = torch.ops.aten.reciprocal.default(sqrt_382);  sqrt_382 = None
        mul_1274 = torch.ops.aten.mul.Tensor(reciprocal_382, 1);  reciprocal_382 = None
        unsqueeze_3088 = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_3089 = torch.ops.aten.unsqueeze.default(unsqueeze_3088, -1);  unsqueeze_3088 = None
        unsqueeze_3090 = torch.ops.aten.unsqueeze.default(mul_1274, -1);  mul_1274 = None
        unsqueeze_3091 = torch.ops.aten.unsqueeze.default(unsqueeze_3090, -1);  unsqueeze_3090 = None
        sub_382 = torch.ops.aten.sub.Tensor(convolution_382, unsqueeze_3089);  convolution_382 = unsqueeze_3089 = None
        mul_1275 = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_3091);  sub_382 = unsqueeze_3091 = None
        unsqueeze_3092 = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_3093 = torch.ops.aten.unsqueeze.default(unsqueeze_3092, -1);  unsqueeze_3092 = None
        mul_1276 = torch.ops.aten.mul.Tensor(mul_1275, unsqueeze_3093);  mul_1275 = unsqueeze_3093 = None
        unsqueeze_3094 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_3095 = torch.ops.aten.unsqueeze.default(unsqueeze_3094, -1);  unsqueeze_3094 = None
        add_1094 = torch.ops.aten.add.Tensor(mul_1276, unsqueeze_3095);  mul_1276 = unsqueeze_3095 = None
        add_1095 = torch.ops.aten.add.Tensor(add_1094, relu_338);  add_1094 = relu_338 = None
        relu_340 = torch.ops.aten.relu.default(add_1095);  add_1095 = None
        convolution_383 = torch.ops.aten.convolution.default(relu_340, arg291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg291_1 = None
        add_1096 = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_383 = torch.ops.aten.sqrt.default(add_1096);  add_1096 = None
        reciprocal_383 = torch.ops.aten.reciprocal.default(sqrt_383);  sqrt_383 = None
        mul_1277 = torch.ops.aten.mul.Tensor(reciprocal_383, 1);  reciprocal_383 = None
        unsqueeze_3096 = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_3097 = torch.ops.aten.unsqueeze.default(unsqueeze_3096, -1);  unsqueeze_3096 = None
        unsqueeze_3098 = torch.ops.aten.unsqueeze.default(mul_1277, -1);  mul_1277 = None
        unsqueeze_3099 = torch.ops.aten.unsqueeze.default(unsqueeze_3098, -1);  unsqueeze_3098 = None
        sub_383 = torch.ops.aten.sub.Tensor(convolution_383, unsqueeze_3097);  convolution_383 = unsqueeze_3097 = None
        mul_1278 = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_3099);  sub_383 = unsqueeze_3099 = None
        unsqueeze_3100 = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_3101 = torch.ops.aten.unsqueeze.default(unsqueeze_3100, -1);  unsqueeze_3100 = None
        mul_1279 = torch.ops.aten.mul.Tensor(mul_1278, unsqueeze_3101);  mul_1278 = unsqueeze_3101 = None
        unsqueeze_3102 = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_3103 = torch.ops.aten.unsqueeze.default(unsqueeze_3102, -1);  unsqueeze_3102 = None
        add_1097 = torch.ops.aten.add.Tensor(mul_1279, unsqueeze_3103);  mul_1279 = unsqueeze_3103 = None
        relu_341 = torch.ops.aten.relu.default(add_1097);  add_1097 = None
        convolution_384 = torch.ops.aten.convolution.default(relu_341, arg296_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_341 = arg296_1 = None
        add_1098 = torch.ops.aten.add.Tensor(arg298_1, 1e-05);  arg298_1 = None
        sqrt_384 = torch.ops.aten.sqrt.default(add_1098);  add_1098 = None
        reciprocal_384 = torch.ops.aten.reciprocal.default(sqrt_384);  sqrt_384 = None
        mul_1280 = torch.ops.aten.mul.Tensor(reciprocal_384, 1);  reciprocal_384 = None
        unsqueeze_3104 = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_3105 = torch.ops.aten.unsqueeze.default(unsqueeze_3104, -1);  unsqueeze_3104 = None
        unsqueeze_3106 = torch.ops.aten.unsqueeze.default(mul_1280, -1);  mul_1280 = None
        unsqueeze_3107 = torch.ops.aten.unsqueeze.default(unsqueeze_3106, -1);  unsqueeze_3106 = None
        sub_384 = torch.ops.aten.sub.Tensor(convolution_384, unsqueeze_3105);  convolution_384 = unsqueeze_3105 = None
        mul_1281 = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_3107);  sub_384 = unsqueeze_3107 = None
        unsqueeze_3108 = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_3109 = torch.ops.aten.unsqueeze.default(unsqueeze_3108, -1);  unsqueeze_3108 = None
        mul_1282 = torch.ops.aten.mul.Tensor(mul_1281, unsqueeze_3109);  mul_1281 = unsqueeze_3109 = None
        unsqueeze_3110 = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_3111 = torch.ops.aten.unsqueeze.default(unsqueeze_3110, -1);  unsqueeze_3110 = None
        add_1099 = torch.ops.aten.add.Tensor(mul_1282, unsqueeze_3111);  mul_1282 = unsqueeze_3111 = None
        add_1100 = torch.ops.aten.add.Tensor(add_1099, relu_340);  add_1099 = relu_340 = None
        relu_342 = torch.ops.aten.relu.default(add_1100);  add_1100 = None
        convolution_385 = torch.ops.aten.convolution.default(relu_334, arg301_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg301_1 = None
        add_1101 = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_385 = torch.ops.aten.sqrt.default(add_1101);  add_1101 = None
        reciprocal_385 = torch.ops.aten.reciprocal.default(sqrt_385);  sqrt_385 = None
        mul_1283 = torch.ops.aten.mul.Tensor(reciprocal_385, 1);  reciprocal_385 = None
        unsqueeze_3112 = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_3113 = torch.ops.aten.unsqueeze.default(unsqueeze_3112, -1);  unsqueeze_3112 = None
        unsqueeze_3114 = torch.ops.aten.unsqueeze.default(mul_1283, -1);  mul_1283 = None
        unsqueeze_3115 = torch.ops.aten.unsqueeze.default(unsqueeze_3114, -1);  unsqueeze_3114 = None
        sub_385 = torch.ops.aten.sub.Tensor(convolution_385, unsqueeze_3113);  convolution_385 = unsqueeze_3113 = None
        mul_1284 = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_3115);  sub_385 = unsqueeze_3115 = None
        unsqueeze_3116 = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_3117 = torch.ops.aten.unsqueeze.default(unsqueeze_3116, -1);  unsqueeze_3116 = None
        mul_1285 = torch.ops.aten.mul.Tensor(mul_1284, unsqueeze_3117);  mul_1284 = unsqueeze_3117 = None
        unsqueeze_3118 = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_3119 = torch.ops.aten.unsqueeze.default(unsqueeze_3118, -1);  unsqueeze_3118 = None
        add_1102 = torch.ops.aten.add.Tensor(mul_1285, unsqueeze_3119);  mul_1285 = unsqueeze_3119 = None
        iota_64 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1286 = torch.ops.aten.mul.Tensor(iota_64, 1);  iota_64 = None
        add_1103 = torch.ops.aten.add.Tensor(mul_1286, 0);  mul_1286 = None
        convert_element_type_900 = torch.ops.prims.convert_element_type.default(add_1103, torch.float32);  add_1103 = None
        add_1104 = torch.ops.aten.add.Tensor(convert_element_type_900, 0.0);  convert_element_type_900 = None
        mul_1287 = torch.ops.aten.mul.Tensor(add_1104, 0.5);  add_1104 = None
        convert_element_type_901 = torch.ops.prims.convert_element_type.default(mul_1287, torch.int64);  mul_1287 = None
        unsqueeze_3120 = torch.ops.aten.unsqueeze.default(convert_element_type_901, -1);  convert_element_type_901 = None
        iota_65 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1288 = torch.ops.aten.mul.Tensor(iota_65, 1);  iota_65 = None
        add_1105 = torch.ops.aten.add.Tensor(mul_1288, 0);  mul_1288 = None
        convert_element_type_902 = torch.ops.prims.convert_element_type.default(add_1105, torch.float32);  add_1105 = None
        add_1106 = torch.ops.aten.add.Tensor(convert_element_type_902, 0.0);  convert_element_type_902 = None
        mul_1289 = torch.ops.aten.mul.Tensor(add_1106, 0.5);  add_1106 = None
        convert_element_type_903 = torch.ops.prims.convert_element_type.default(mul_1289, torch.int64);  mul_1289 = None
        _unsafe_index_32 = torch.ops.aten._unsafe_index.Tensor(add_1102, [None, None, unsqueeze_3120, convert_element_type_903]);  add_1102 = unsqueeze_3120 = convert_element_type_903 = None
        add_1107 = torch.ops.aten.add.Tensor(relu_326, _unsafe_index_32);  _unsafe_index_32 = None
        convolution_386 = torch.ops.aten.convolution.default(relu_342, arg306_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg306_1 = None
        add_1108 = torch.ops.aten.add.Tensor(arg308_1, 1e-05);  arg308_1 = None
        sqrt_386 = torch.ops.aten.sqrt.default(add_1108);  add_1108 = None
        reciprocal_386 = torch.ops.aten.reciprocal.default(sqrt_386);  sqrt_386 = None
        mul_1290 = torch.ops.aten.mul.Tensor(reciprocal_386, 1);  reciprocal_386 = None
        unsqueeze_3121 = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
        unsqueeze_3122 = torch.ops.aten.unsqueeze.default(unsqueeze_3121, -1);  unsqueeze_3121 = None
        unsqueeze_3123 = torch.ops.aten.unsqueeze.default(mul_1290, -1);  mul_1290 = None
        unsqueeze_3124 = torch.ops.aten.unsqueeze.default(unsqueeze_3123, -1);  unsqueeze_3123 = None
        sub_386 = torch.ops.aten.sub.Tensor(convolution_386, unsqueeze_3122);  convolution_386 = unsqueeze_3122 = None
        mul_1291 = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_3124);  sub_386 = unsqueeze_3124 = None
        unsqueeze_3125 = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_3126 = torch.ops.aten.unsqueeze.default(unsqueeze_3125, -1);  unsqueeze_3125 = None
        mul_1292 = torch.ops.aten.mul.Tensor(mul_1291, unsqueeze_3126);  mul_1291 = unsqueeze_3126 = None
        unsqueeze_3127 = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_3128 = torch.ops.aten.unsqueeze.default(unsqueeze_3127, -1);  unsqueeze_3127 = None
        add_1109 = torch.ops.aten.add.Tensor(mul_1292, unsqueeze_3128);  mul_1292 = unsqueeze_3128 = None
        iota_66 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1293 = torch.ops.aten.mul.Tensor(iota_66, 1);  iota_66 = None
        add_1110 = torch.ops.aten.add.Tensor(mul_1293, 0);  mul_1293 = None
        convert_element_type_906 = torch.ops.prims.convert_element_type.default(add_1110, torch.float32);  add_1110 = None
        add_1111 = torch.ops.aten.add.Tensor(convert_element_type_906, 0.0);  convert_element_type_906 = None
        mul_1294 = torch.ops.aten.mul.Tensor(add_1111, 0.25);  add_1111 = None
        convert_element_type_907 = torch.ops.prims.convert_element_type.default(mul_1294, torch.int64);  mul_1294 = None
        unsqueeze_3129 = torch.ops.aten.unsqueeze.default(convert_element_type_907, -1);  convert_element_type_907 = None
        iota_67 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1295 = torch.ops.aten.mul.Tensor(iota_67, 1);  iota_67 = None
        add_1112 = torch.ops.aten.add.Tensor(mul_1295, 0);  mul_1295 = None
        convert_element_type_908 = torch.ops.prims.convert_element_type.default(add_1112, torch.float32);  add_1112 = None
        add_1113 = torch.ops.aten.add.Tensor(convert_element_type_908, 0.0);  convert_element_type_908 = None
        mul_1296 = torch.ops.aten.mul.Tensor(add_1113, 0.25);  add_1113 = None
        convert_element_type_909 = torch.ops.prims.convert_element_type.default(mul_1296, torch.int64);  mul_1296 = None
        _unsafe_index_33 = torch.ops.aten._unsafe_index.Tensor(add_1109, [None, None, unsqueeze_3129, convert_element_type_909]);  add_1109 = unsqueeze_3129 = convert_element_type_909 = None
        add_1114 = torch.ops.aten.add.Tensor(add_1107, _unsafe_index_33);  add_1107 = _unsafe_index_33 = None
        relu_343 = torch.ops.aten.relu.default(add_1114);  add_1114 = None
        convolution_387 = torch.ops.aten.convolution.default(relu_326, arg311_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg311_1 = None
        add_1115 = torch.ops.aten.add.Tensor(arg313_1, 1e-05);  arg313_1 = None
        sqrt_387 = torch.ops.aten.sqrt.default(add_1115);  add_1115 = None
        reciprocal_387 = torch.ops.aten.reciprocal.default(sqrt_387);  sqrt_387 = None
        mul_1297 = torch.ops.aten.mul.Tensor(reciprocal_387, 1);  reciprocal_387 = None
        unsqueeze_3130 = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_3131 = torch.ops.aten.unsqueeze.default(unsqueeze_3130, -1);  unsqueeze_3130 = None
        unsqueeze_3132 = torch.ops.aten.unsqueeze.default(mul_1297, -1);  mul_1297 = None
        unsqueeze_3133 = torch.ops.aten.unsqueeze.default(unsqueeze_3132, -1);  unsqueeze_3132 = None
        sub_387 = torch.ops.aten.sub.Tensor(convolution_387, unsqueeze_3131);  convolution_387 = unsqueeze_3131 = None
        mul_1298 = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_3133);  sub_387 = unsqueeze_3133 = None
        unsqueeze_3134 = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_3135 = torch.ops.aten.unsqueeze.default(unsqueeze_3134, -1);  unsqueeze_3134 = None
        mul_1299 = torch.ops.aten.mul.Tensor(mul_1298, unsqueeze_3135);  mul_1298 = unsqueeze_3135 = None
        unsqueeze_3136 = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_3137 = torch.ops.aten.unsqueeze.default(unsqueeze_3136, -1);  unsqueeze_3136 = None
        add_1116 = torch.ops.aten.add.Tensor(mul_1299, unsqueeze_3137);  mul_1299 = unsqueeze_3137 = None
        add_1117 = torch.ops.aten.add.Tensor(add_1116, relu_334);  add_1116 = None
        convolution_388 = torch.ops.aten.convolution.default(relu_342, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg316_1 = None
        add_1118 = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_388 = torch.ops.aten.sqrt.default(add_1118);  add_1118 = None
        reciprocal_388 = torch.ops.aten.reciprocal.default(sqrt_388);  sqrt_388 = None
        mul_1300 = torch.ops.aten.mul.Tensor(reciprocal_388, 1);  reciprocal_388 = None
        unsqueeze_3138 = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_3139 = torch.ops.aten.unsqueeze.default(unsqueeze_3138, -1);  unsqueeze_3138 = None
        unsqueeze_3140 = torch.ops.aten.unsqueeze.default(mul_1300, -1);  mul_1300 = None
        unsqueeze_3141 = torch.ops.aten.unsqueeze.default(unsqueeze_3140, -1);  unsqueeze_3140 = None
        sub_388 = torch.ops.aten.sub.Tensor(convolution_388, unsqueeze_3139);  convolution_388 = unsqueeze_3139 = None
        mul_1301 = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_3141);  sub_388 = unsqueeze_3141 = None
        unsqueeze_3142 = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_3143 = torch.ops.aten.unsqueeze.default(unsqueeze_3142, -1);  unsqueeze_3142 = None
        mul_1302 = torch.ops.aten.mul.Tensor(mul_1301, unsqueeze_3143);  mul_1301 = unsqueeze_3143 = None
        unsqueeze_3144 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_3145 = torch.ops.aten.unsqueeze.default(unsqueeze_3144, -1);  unsqueeze_3144 = None
        add_1119 = torch.ops.aten.add.Tensor(mul_1302, unsqueeze_3145);  mul_1302 = unsqueeze_3145 = None
        iota_68 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1303 = torch.ops.aten.mul.Tensor(iota_68, 1);  iota_68 = None
        add_1120 = torch.ops.aten.add.Tensor(mul_1303, 0);  mul_1303 = None
        convert_element_type_914 = torch.ops.prims.convert_element_type.default(add_1120, torch.float32);  add_1120 = None
        add_1121 = torch.ops.aten.add.Tensor(convert_element_type_914, 0.0);  convert_element_type_914 = None
        mul_1304 = torch.ops.aten.mul.Tensor(add_1121, 0.5);  add_1121 = None
        convert_element_type_915 = torch.ops.prims.convert_element_type.default(mul_1304, torch.int64);  mul_1304 = None
        unsqueeze_3146 = torch.ops.aten.unsqueeze.default(convert_element_type_915, -1);  convert_element_type_915 = None
        iota_69 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1305 = torch.ops.aten.mul.Tensor(iota_69, 1);  iota_69 = None
        add_1122 = torch.ops.aten.add.Tensor(mul_1305, 0);  mul_1305 = None
        convert_element_type_916 = torch.ops.prims.convert_element_type.default(add_1122, torch.float32);  add_1122 = None
        add_1123 = torch.ops.aten.add.Tensor(convert_element_type_916, 0.0);  convert_element_type_916 = None
        mul_1306 = torch.ops.aten.mul.Tensor(add_1123, 0.5);  add_1123 = None
        convert_element_type_917 = torch.ops.prims.convert_element_type.default(mul_1306, torch.int64);  mul_1306 = None
        _unsafe_index_34 = torch.ops.aten._unsafe_index.Tensor(add_1119, [None, None, unsqueeze_3146, convert_element_type_917]);  add_1119 = unsqueeze_3146 = convert_element_type_917 = None
        add_1124 = torch.ops.aten.add.Tensor(add_1117, _unsafe_index_34);  add_1117 = _unsafe_index_34 = None
        relu_344 = torch.ops.aten.relu.default(add_1124);  add_1124 = None
        convolution_389 = torch.ops.aten.convolution.default(relu_326, arg321_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_326 = arg321_1 = None
        add_1125 = torch.ops.aten.add.Tensor(arg323_1, 1e-05);  arg323_1 = None
        sqrt_389 = torch.ops.aten.sqrt.default(add_1125);  add_1125 = None
        reciprocal_389 = torch.ops.aten.reciprocal.default(sqrt_389);  sqrt_389 = None
        mul_1307 = torch.ops.aten.mul.Tensor(reciprocal_389, 1);  reciprocal_389 = None
        unsqueeze_3147 = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_3148 = torch.ops.aten.unsqueeze.default(unsqueeze_3147, -1);  unsqueeze_3147 = None
        unsqueeze_3149 = torch.ops.aten.unsqueeze.default(mul_1307, -1);  mul_1307 = None
        unsqueeze_3150 = torch.ops.aten.unsqueeze.default(unsqueeze_3149, -1);  unsqueeze_3149 = None
        sub_389 = torch.ops.aten.sub.Tensor(convolution_389, unsqueeze_3148);  convolution_389 = unsqueeze_3148 = None
        mul_1308 = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_3150);  sub_389 = unsqueeze_3150 = None
        unsqueeze_3151 = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_3152 = torch.ops.aten.unsqueeze.default(unsqueeze_3151, -1);  unsqueeze_3151 = None
        mul_1309 = torch.ops.aten.mul.Tensor(mul_1308, unsqueeze_3152);  mul_1308 = unsqueeze_3152 = None
        unsqueeze_3153 = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_3154 = torch.ops.aten.unsqueeze.default(unsqueeze_3153, -1);  unsqueeze_3153 = None
        add_1126 = torch.ops.aten.add.Tensor(mul_1309, unsqueeze_3154);  mul_1309 = unsqueeze_3154 = None
        relu_345 = torch.ops.aten.relu.default(add_1126);  add_1126 = None
        convolution_390 = torch.ops.aten.convolution.default(relu_345, arg326_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_345 = arg326_1 = None
        add_1127 = torch.ops.aten.add.Tensor(arg328_1, 1e-05);  arg328_1 = None
        sqrt_390 = torch.ops.aten.sqrt.default(add_1127);  add_1127 = None
        reciprocal_390 = torch.ops.aten.reciprocal.default(sqrt_390);  sqrt_390 = None
        mul_1310 = torch.ops.aten.mul.Tensor(reciprocal_390, 1);  reciprocal_390 = None
        unsqueeze_3155 = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_3156 = torch.ops.aten.unsqueeze.default(unsqueeze_3155, -1);  unsqueeze_3155 = None
        unsqueeze_3157 = torch.ops.aten.unsqueeze.default(mul_1310, -1);  mul_1310 = None
        unsqueeze_3158 = torch.ops.aten.unsqueeze.default(unsqueeze_3157, -1);  unsqueeze_3157 = None
        sub_390 = torch.ops.aten.sub.Tensor(convolution_390, unsqueeze_3156);  convolution_390 = unsqueeze_3156 = None
        mul_1311 = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_3158);  sub_390 = unsqueeze_3158 = None
        unsqueeze_3159 = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_3160 = torch.ops.aten.unsqueeze.default(unsqueeze_3159, -1);  unsqueeze_3159 = None
        mul_1312 = torch.ops.aten.mul.Tensor(mul_1311, unsqueeze_3160);  mul_1311 = unsqueeze_3160 = None
        unsqueeze_3161 = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_3162 = torch.ops.aten.unsqueeze.default(unsqueeze_3161, -1);  unsqueeze_3161 = None
        add_1128 = torch.ops.aten.add.Tensor(mul_1312, unsqueeze_3162);  mul_1312 = unsqueeze_3162 = None
        convolution_391 = torch.ops.aten.convolution.default(relu_334, arg331_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_334 = arg331_1 = None
        add_1129 = torch.ops.aten.add.Tensor(arg333_1, 1e-05);  arg333_1 = None
        sqrt_391 = torch.ops.aten.sqrt.default(add_1129);  add_1129 = None
        reciprocal_391 = torch.ops.aten.reciprocal.default(sqrt_391);  sqrt_391 = None
        mul_1313 = torch.ops.aten.mul.Tensor(reciprocal_391, 1);  reciprocal_391 = None
        unsqueeze_3163 = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_3164 = torch.ops.aten.unsqueeze.default(unsqueeze_3163, -1);  unsqueeze_3163 = None
        unsqueeze_3165 = torch.ops.aten.unsqueeze.default(mul_1313, -1);  mul_1313 = None
        unsqueeze_3166 = torch.ops.aten.unsqueeze.default(unsqueeze_3165, -1);  unsqueeze_3165 = None
        sub_391 = torch.ops.aten.sub.Tensor(convolution_391, unsqueeze_3164);  convolution_391 = unsqueeze_3164 = None
        mul_1314 = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_3166);  sub_391 = unsqueeze_3166 = None
        unsqueeze_3167 = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_3168 = torch.ops.aten.unsqueeze.default(unsqueeze_3167, -1);  unsqueeze_3167 = None
        mul_1315 = torch.ops.aten.mul.Tensor(mul_1314, unsqueeze_3168);  mul_1314 = unsqueeze_3168 = None
        unsqueeze_3169 = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_3170 = torch.ops.aten.unsqueeze.default(unsqueeze_3169, -1);  unsqueeze_3169 = None
        add_1130 = torch.ops.aten.add.Tensor(mul_1315, unsqueeze_3170);  mul_1315 = unsqueeze_3170 = None
        add_1131 = torch.ops.aten.add.Tensor(add_1128, add_1130);  add_1128 = add_1130 = None
        add_1132 = torch.ops.aten.add.Tensor(add_1131, relu_342);  add_1131 = relu_342 = None
        relu_346 = torch.ops.aten.relu.default(add_1132);  add_1132 = None
        convolution_392 = torch.ops.aten.convolution.default(relu_343, arg336_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg336_1 = None
        add_1133 = torch.ops.aten.add.Tensor(arg338_1, 1e-05);  arg338_1 = None
        sqrt_392 = torch.ops.aten.sqrt.default(add_1133);  add_1133 = None
        reciprocal_392 = torch.ops.aten.reciprocal.default(sqrt_392);  sqrt_392 = None
        mul_1316 = torch.ops.aten.mul.Tensor(reciprocal_392, 1);  reciprocal_392 = None
        unsqueeze_3171 = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_3172 = torch.ops.aten.unsqueeze.default(unsqueeze_3171, -1);  unsqueeze_3171 = None
        unsqueeze_3173 = torch.ops.aten.unsqueeze.default(mul_1316, -1);  mul_1316 = None
        unsqueeze_3174 = torch.ops.aten.unsqueeze.default(unsqueeze_3173, -1);  unsqueeze_3173 = None
        sub_392 = torch.ops.aten.sub.Tensor(convolution_392, unsqueeze_3172);  convolution_392 = unsqueeze_3172 = None
        mul_1317 = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_3174);  sub_392 = unsqueeze_3174 = None
        unsqueeze_3175 = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_3176 = torch.ops.aten.unsqueeze.default(unsqueeze_3175, -1);  unsqueeze_3175 = None
        mul_1318 = torch.ops.aten.mul.Tensor(mul_1317, unsqueeze_3176);  mul_1317 = unsqueeze_3176 = None
        unsqueeze_3177 = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_3178 = torch.ops.aten.unsqueeze.default(unsqueeze_3177, -1);  unsqueeze_3177 = None
        add_1134 = torch.ops.aten.add.Tensor(mul_1318, unsqueeze_3178);  mul_1318 = unsqueeze_3178 = None
        relu_347 = torch.ops.aten.relu.default(add_1134);  add_1134 = None
        convolution_393 = torch.ops.aten.convolution.default(relu_347, arg341_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_347 = arg341_1 = None
        add_1135 = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_393 = torch.ops.aten.sqrt.default(add_1135);  add_1135 = None
        reciprocal_393 = torch.ops.aten.reciprocal.default(sqrt_393);  sqrt_393 = None
        mul_1319 = torch.ops.aten.mul.Tensor(reciprocal_393, 1);  reciprocal_393 = None
        unsqueeze_3179 = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_3180 = torch.ops.aten.unsqueeze.default(unsqueeze_3179, -1);  unsqueeze_3179 = None
        unsqueeze_3181 = torch.ops.aten.unsqueeze.default(mul_1319, -1);  mul_1319 = None
        unsqueeze_3182 = torch.ops.aten.unsqueeze.default(unsqueeze_3181, -1);  unsqueeze_3181 = None
        sub_393 = torch.ops.aten.sub.Tensor(convolution_393, unsqueeze_3180);  convolution_393 = unsqueeze_3180 = None
        mul_1320 = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_3182);  sub_393 = unsqueeze_3182 = None
        unsqueeze_3183 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_3184 = torch.ops.aten.unsqueeze.default(unsqueeze_3183, -1);  unsqueeze_3183 = None
        mul_1321 = torch.ops.aten.mul.Tensor(mul_1320, unsqueeze_3184);  mul_1320 = unsqueeze_3184 = None
        unsqueeze_3185 = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_3186 = torch.ops.aten.unsqueeze.default(unsqueeze_3185, -1);  unsqueeze_3185 = None
        add_1136 = torch.ops.aten.add.Tensor(mul_1321, unsqueeze_3186);  mul_1321 = unsqueeze_3186 = None
        add_1137 = torch.ops.aten.add.Tensor(add_1136, relu_343);  add_1136 = relu_343 = None
        relu_348 = torch.ops.aten.relu.default(add_1137);  add_1137 = None
        convolution_394 = torch.ops.aten.convolution.default(relu_348, arg346_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg346_1 = None
        add_1138 = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_394 = torch.ops.aten.sqrt.default(add_1138);  add_1138 = None
        reciprocal_394 = torch.ops.aten.reciprocal.default(sqrt_394);  sqrt_394 = None
        mul_1322 = torch.ops.aten.mul.Tensor(reciprocal_394, 1);  reciprocal_394 = None
        unsqueeze_3187 = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_3188 = torch.ops.aten.unsqueeze.default(unsqueeze_3187, -1);  unsqueeze_3187 = None
        unsqueeze_3189 = torch.ops.aten.unsqueeze.default(mul_1322, -1);  mul_1322 = None
        unsqueeze_3190 = torch.ops.aten.unsqueeze.default(unsqueeze_3189, -1);  unsqueeze_3189 = None
        sub_394 = torch.ops.aten.sub.Tensor(convolution_394, unsqueeze_3188);  convolution_394 = unsqueeze_3188 = None
        mul_1323 = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_3190);  sub_394 = unsqueeze_3190 = None
        unsqueeze_3191 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_3192 = torch.ops.aten.unsqueeze.default(unsqueeze_3191, -1);  unsqueeze_3191 = None
        mul_1324 = torch.ops.aten.mul.Tensor(mul_1323, unsqueeze_3192);  mul_1323 = unsqueeze_3192 = None
        unsqueeze_3193 = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_3194 = torch.ops.aten.unsqueeze.default(unsqueeze_3193, -1);  unsqueeze_3193 = None
        add_1139 = torch.ops.aten.add.Tensor(mul_1324, unsqueeze_3194);  mul_1324 = unsqueeze_3194 = None
        relu_349 = torch.ops.aten.relu.default(add_1139);  add_1139 = None
        convolution_395 = torch.ops.aten.convolution.default(relu_349, arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_349 = arg351_1 = None
        add_1140 = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
        sqrt_395 = torch.ops.aten.sqrt.default(add_1140);  add_1140 = None
        reciprocal_395 = torch.ops.aten.reciprocal.default(sqrt_395);  sqrt_395 = None
        mul_1325 = torch.ops.aten.mul.Tensor(reciprocal_395, 1);  reciprocal_395 = None
        unsqueeze_3195 = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_3196 = torch.ops.aten.unsqueeze.default(unsqueeze_3195, -1);  unsqueeze_3195 = None
        unsqueeze_3197 = torch.ops.aten.unsqueeze.default(mul_1325, -1);  mul_1325 = None
        unsqueeze_3198 = torch.ops.aten.unsqueeze.default(unsqueeze_3197, -1);  unsqueeze_3197 = None
        sub_395 = torch.ops.aten.sub.Tensor(convolution_395, unsqueeze_3196);  convolution_395 = unsqueeze_3196 = None
        mul_1326 = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_3198);  sub_395 = unsqueeze_3198 = None
        unsqueeze_3199 = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_3200 = torch.ops.aten.unsqueeze.default(unsqueeze_3199, -1);  unsqueeze_3199 = None
        mul_1327 = torch.ops.aten.mul.Tensor(mul_1326, unsqueeze_3200);  mul_1326 = unsqueeze_3200 = None
        unsqueeze_3201 = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_3202 = torch.ops.aten.unsqueeze.default(unsqueeze_3201, -1);  unsqueeze_3201 = None
        add_1141 = torch.ops.aten.add.Tensor(mul_1327, unsqueeze_3202);  mul_1327 = unsqueeze_3202 = None
        add_1142 = torch.ops.aten.add.Tensor(add_1141, relu_348);  add_1141 = relu_348 = None
        relu_350 = torch.ops.aten.relu.default(add_1142);  add_1142 = None
        convolution_396 = torch.ops.aten.convolution.default(relu_350, arg356_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg356_1 = None
        add_1143 = torch.ops.aten.add.Tensor(arg358_1, 1e-05);  arg358_1 = None
        sqrt_396 = torch.ops.aten.sqrt.default(add_1143);  add_1143 = None
        reciprocal_396 = torch.ops.aten.reciprocal.default(sqrt_396);  sqrt_396 = None
        mul_1328 = torch.ops.aten.mul.Tensor(reciprocal_396, 1);  reciprocal_396 = None
        unsqueeze_3203 = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_3204 = torch.ops.aten.unsqueeze.default(unsqueeze_3203, -1);  unsqueeze_3203 = None
        unsqueeze_3205 = torch.ops.aten.unsqueeze.default(mul_1328, -1);  mul_1328 = None
        unsqueeze_3206 = torch.ops.aten.unsqueeze.default(unsqueeze_3205, -1);  unsqueeze_3205 = None
        sub_396 = torch.ops.aten.sub.Tensor(convolution_396, unsqueeze_3204);  convolution_396 = unsqueeze_3204 = None
        mul_1329 = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_3206);  sub_396 = unsqueeze_3206 = None
        unsqueeze_3207 = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_3208 = torch.ops.aten.unsqueeze.default(unsqueeze_3207, -1);  unsqueeze_3207 = None
        mul_1330 = torch.ops.aten.mul.Tensor(mul_1329, unsqueeze_3208);  mul_1329 = unsqueeze_3208 = None
        unsqueeze_3209 = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_3210 = torch.ops.aten.unsqueeze.default(unsqueeze_3209, -1);  unsqueeze_3209 = None
        add_1144 = torch.ops.aten.add.Tensor(mul_1330, unsqueeze_3210);  mul_1330 = unsqueeze_3210 = None
        relu_351 = torch.ops.aten.relu.default(add_1144);  add_1144 = None
        convolution_397 = torch.ops.aten.convolution.default(relu_351, arg361_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_351 = arg361_1 = None
        add_1145 = torch.ops.aten.add.Tensor(arg363_1, 1e-05);  arg363_1 = None
        sqrt_397 = torch.ops.aten.sqrt.default(add_1145);  add_1145 = None
        reciprocal_397 = torch.ops.aten.reciprocal.default(sqrt_397);  sqrt_397 = None
        mul_1331 = torch.ops.aten.mul.Tensor(reciprocal_397, 1);  reciprocal_397 = None
        unsqueeze_3211 = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
        unsqueeze_3212 = torch.ops.aten.unsqueeze.default(unsqueeze_3211, -1);  unsqueeze_3211 = None
        unsqueeze_3213 = torch.ops.aten.unsqueeze.default(mul_1331, -1);  mul_1331 = None
        unsqueeze_3214 = torch.ops.aten.unsqueeze.default(unsqueeze_3213, -1);  unsqueeze_3213 = None
        sub_397 = torch.ops.aten.sub.Tensor(convolution_397, unsqueeze_3212);  convolution_397 = unsqueeze_3212 = None
        mul_1332 = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_3214);  sub_397 = unsqueeze_3214 = None
        unsqueeze_3215 = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_3216 = torch.ops.aten.unsqueeze.default(unsqueeze_3215, -1);  unsqueeze_3215 = None
        mul_1333 = torch.ops.aten.mul.Tensor(mul_1332, unsqueeze_3216);  mul_1332 = unsqueeze_3216 = None
        unsqueeze_3217 = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_3218 = torch.ops.aten.unsqueeze.default(unsqueeze_3217, -1);  unsqueeze_3217 = None
        add_1146 = torch.ops.aten.add.Tensor(mul_1333, unsqueeze_3218);  mul_1333 = unsqueeze_3218 = None
        add_1147 = torch.ops.aten.add.Tensor(add_1146, relu_350);  add_1146 = relu_350 = None
        relu_352 = torch.ops.aten.relu.default(add_1147);  add_1147 = None
        convolution_398 = torch.ops.aten.convolution.default(relu_352, arg366_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg366_1 = None
        add_1148 = torch.ops.aten.add.Tensor(arg368_1, 1e-05);  arg368_1 = None
        sqrt_398 = torch.ops.aten.sqrt.default(add_1148);  add_1148 = None
        reciprocal_398 = torch.ops.aten.reciprocal.default(sqrt_398);  sqrt_398 = None
        mul_1334 = torch.ops.aten.mul.Tensor(reciprocal_398, 1);  reciprocal_398 = None
        unsqueeze_3219 = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_3220 = torch.ops.aten.unsqueeze.default(unsqueeze_3219, -1);  unsqueeze_3219 = None
        unsqueeze_3221 = torch.ops.aten.unsqueeze.default(mul_1334, -1);  mul_1334 = None
        unsqueeze_3222 = torch.ops.aten.unsqueeze.default(unsqueeze_3221, -1);  unsqueeze_3221 = None
        sub_398 = torch.ops.aten.sub.Tensor(convolution_398, unsqueeze_3220);  convolution_398 = unsqueeze_3220 = None
        mul_1335 = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_3222);  sub_398 = unsqueeze_3222 = None
        unsqueeze_3223 = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_3224 = torch.ops.aten.unsqueeze.default(unsqueeze_3223, -1);  unsqueeze_3223 = None
        mul_1336 = torch.ops.aten.mul.Tensor(mul_1335, unsqueeze_3224);  mul_1335 = unsqueeze_3224 = None
        unsqueeze_3225 = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
        unsqueeze_3226 = torch.ops.aten.unsqueeze.default(unsqueeze_3225, -1);  unsqueeze_3225 = None
        add_1149 = torch.ops.aten.add.Tensor(mul_1336, unsqueeze_3226);  mul_1336 = unsqueeze_3226 = None
        relu_353 = torch.ops.aten.relu.default(add_1149);  add_1149 = None
        convolution_399 = torch.ops.aten.convolution.default(relu_353, arg371_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_353 = arg371_1 = None
        add_1150 = torch.ops.aten.add.Tensor(arg373_1, 1e-05);  arg373_1 = None
        sqrt_399 = torch.ops.aten.sqrt.default(add_1150);  add_1150 = None
        reciprocal_399 = torch.ops.aten.reciprocal.default(sqrt_399);  sqrt_399 = None
        mul_1337 = torch.ops.aten.mul.Tensor(reciprocal_399, 1);  reciprocal_399 = None
        unsqueeze_3227 = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_3228 = torch.ops.aten.unsqueeze.default(unsqueeze_3227, -1);  unsqueeze_3227 = None
        unsqueeze_3229 = torch.ops.aten.unsqueeze.default(mul_1337, -1);  mul_1337 = None
        unsqueeze_3230 = torch.ops.aten.unsqueeze.default(unsqueeze_3229, -1);  unsqueeze_3229 = None
        sub_399 = torch.ops.aten.sub.Tensor(convolution_399, unsqueeze_3228);  convolution_399 = unsqueeze_3228 = None
        mul_1338 = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_3230);  sub_399 = unsqueeze_3230 = None
        unsqueeze_3231 = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_3232 = torch.ops.aten.unsqueeze.default(unsqueeze_3231, -1);  unsqueeze_3231 = None
        mul_1339 = torch.ops.aten.mul.Tensor(mul_1338, unsqueeze_3232);  mul_1338 = unsqueeze_3232 = None
        unsqueeze_3233 = torch.ops.aten.unsqueeze.default(arg375_1, -1);  arg375_1 = None
        unsqueeze_3234 = torch.ops.aten.unsqueeze.default(unsqueeze_3233, -1);  unsqueeze_3233 = None
        add_1151 = torch.ops.aten.add.Tensor(mul_1339, unsqueeze_3234);  mul_1339 = unsqueeze_3234 = None
        add_1152 = torch.ops.aten.add.Tensor(add_1151, relu_352);  add_1151 = relu_352 = None
        relu_354 = torch.ops.aten.relu.default(add_1152);  add_1152 = None
        convolution_400 = torch.ops.aten.convolution.default(relu_344, arg376_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg376_1 = None
        add_1153 = torch.ops.aten.add.Tensor(arg378_1, 1e-05);  arg378_1 = None
        sqrt_400 = torch.ops.aten.sqrt.default(add_1153);  add_1153 = None
        reciprocal_400 = torch.ops.aten.reciprocal.default(sqrt_400);  sqrt_400 = None
        mul_1340 = torch.ops.aten.mul.Tensor(reciprocal_400, 1);  reciprocal_400 = None
        unsqueeze_3235 = torch.ops.aten.unsqueeze.default(arg377_1, -1);  arg377_1 = None
        unsqueeze_3236 = torch.ops.aten.unsqueeze.default(unsqueeze_3235, -1);  unsqueeze_3235 = None
        unsqueeze_3237 = torch.ops.aten.unsqueeze.default(mul_1340, -1);  mul_1340 = None
        unsqueeze_3238 = torch.ops.aten.unsqueeze.default(unsqueeze_3237, -1);  unsqueeze_3237 = None
        sub_400 = torch.ops.aten.sub.Tensor(convolution_400, unsqueeze_3236);  convolution_400 = unsqueeze_3236 = None
        mul_1341 = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_3238);  sub_400 = unsqueeze_3238 = None
        unsqueeze_3239 = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_3240 = torch.ops.aten.unsqueeze.default(unsqueeze_3239, -1);  unsqueeze_3239 = None
        mul_1342 = torch.ops.aten.mul.Tensor(mul_1341, unsqueeze_3240);  mul_1341 = unsqueeze_3240 = None
        unsqueeze_3241 = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_3242 = torch.ops.aten.unsqueeze.default(unsqueeze_3241, -1);  unsqueeze_3241 = None
        add_1154 = torch.ops.aten.add.Tensor(mul_1342, unsqueeze_3242);  mul_1342 = unsqueeze_3242 = None
        relu_355 = torch.ops.aten.relu.default(add_1154);  add_1154 = None
        convolution_401 = torch.ops.aten.convolution.default(relu_355, arg381_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_355 = arg381_1 = None
        add_1155 = torch.ops.aten.add.Tensor(arg383_1, 1e-05);  arg383_1 = None
        sqrt_401 = torch.ops.aten.sqrt.default(add_1155);  add_1155 = None
        reciprocal_401 = torch.ops.aten.reciprocal.default(sqrt_401);  sqrt_401 = None
        mul_1343 = torch.ops.aten.mul.Tensor(reciprocal_401, 1);  reciprocal_401 = None
        unsqueeze_3243 = torch.ops.aten.unsqueeze.default(arg382_1, -1);  arg382_1 = None
        unsqueeze_3244 = torch.ops.aten.unsqueeze.default(unsqueeze_3243, -1);  unsqueeze_3243 = None
        unsqueeze_3245 = torch.ops.aten.unsqueeze.default(mul_1343, -1);  mul_1343 = None
        unsqueeze_3246 = torch.ops.aten.unsqueeze.default(unsqueeze_3245, -1);  unsqueeze_3245 = None
        sub_401 = torch.ops.aten.sub.Tensor(convolution_401, unsqueeze_3244);  convolution_401 = unsqueeze_3244 = None
        mul_1344 = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_3246);  sub_401 = unsqueeze_3246 = None
        unsqueeze_3247 = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_3248 = torch.ops.aten.unsqueeze.default(unsqueeze_3247, -1);  unsqueeze_3247 = None
        mul_1345 = torch.ops.aten.mul.Tensor(mul_1344, unsqueeze_3248);  mul_1344 = unsqueeze_3248 = None
        unsqueeze_3249 = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_3250 = torch.ops.aten.unsqueeze.default(unsqueeze_3249, -1);  unsqueeze_3249 = None
        add_1156 = torch.ops.aten.add.Tensor(mul_1345, unsqueeze_3250);  mul_1345 = unsqueeze_3250 = None
        add_1157 = torch.ops.aten.add.Tensor(add_1156, relu_344);  add_1156 = relu_344 = None
        relu_356 = torch.ops.aten.relu.default(add_1157);  add_1157 = None
        convolution_402 = torch.ops.aten.convolution.default(relu_356, arg386_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg386_1 = None
        add_1158 = torch.ops.aten.add.Tensor(arg388_1, 1e-05);  arg388_1 = None
        sqrt_402 = torch.ops.aten.sqrt.default(add_1158);  add_1158 = None
        reciprocal_402 = torch.ops.aten.reciprocal.default(sqrt_402);  sqrt_402 = None
        mul_1346 = torch.ops.aten.mul.Tensor(reciprocal_402, 1);  reciprocal_402 = None
        unsqueeze_3251 = torch.ops.aten.unsqueeze.default(arg387_1, -1);  arg387_1 = None
        unsqueeze_3252 = torch.ops.aten.unsqueeze.default(unsqueeze_3251, -1);  unsqueeze_3251 = None
        unsqueeze_3253 = torch.ops.aten.unsqueeze.default(mul_1346, -1);  mul_1346 = None
        unsqueeze_3254 = torch.ops.aten.unsqueeze.default(unsqueeze_3253, -1);  unsqueeze_3253 = None
        sub_402 = torch.ops.aten.sub.Tensor(convolution_402, unsqueeze_3252);  convolution_402 = unsqueeze_3252 = None
        mul_1347 = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_3254);  sub_402 = unsqueeze_3254 = None
        unsqueeze_3255 = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_3256 = torch.ops.aten.unsqueeze.default(unsqueeze_3255, -1);  unsqueeze_3255 = None
        mul_1348 = torch.ops.aten.mul.Tensor(mul_1347, unsqueeze_3256);  mul_1347 = unsqueeze_3256 = None
        unsqueeze_3257 = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_3258 = torch.ops.aten.unsqueeze.default(unsqueeze_3257, -1);  unsqueeze_3257 = None
        add_1159 = torch.ops.aten.add.Tensor(mul_1348, unsqueeze_3258);  mul_1348 = unsqueeze_3258 = None
        relu_357 = torch.ops.aten.relu.default(add_1159);  add_1159 = None
        convolution_403 = torch.ops.aten.convolution.default(relu_357, arg391_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_357 = arg391_1 = None
        add_1160 = torch.ops.aten.add.Tensor(arg393_1, 1e-05);  arg393_1 = None
        sqrt_403 = torch.ops.aten.sqrt.default(add_1160);  add_1160 = None
        reciprocal_403 = torch.ops.aten.reciprocal.default(sqrt_403);  sqrt_403 = None
        mul_1349 = torch.ops.aten.mul.Tensor(reciprocal_403, 1);  reciprocal_403 = None
        unsqueeze_3259 = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
        unsqueeze_3260 = torch.ops.aten.unsqueeze.default(unsqueeze_3259, -1);  unsqueeze_3259 = None
        unsqueeze_3261 = torch.ops.aten.unsqueeze.default(mul_1349, -1);  mul_1349 = None
        unsqueeze_3262 = torch.ops.aten.unsqueeze.default(unsqueeze_3261, -1);  unsqueeze_3261 = None
        sub_403 = torch.ops.aten.sub.Tensor(convolution_403, unsqueeze_3260);  convolution_403 = unsqueeze_3260 = None
        mul_1350 = torch.ops.aten.mul.Tensor(sub_403, unsqueeze_3262);  sub_403 = unsqueeze_3262 = None
        unsqueeze_3263 = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_3264 = torch.ops.aten.unsqueeze.default(unsqueeze_3263, -1);  unsqueeze_3263 = None
        mul_1351 = torch.ops.aten.mul.Tensor(mul_1350, unsqueeze_3264);  mul_1350 = unsqueeze_3264 = None
        unsqueeze_3265 = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_3266 = torch.ops.aten.unsqueeze.default(unsqueeze_3265, -1);  unsqueeze_3265 = None
        add_1161 = torch.ops.aten.add.Tensor(mul_1351, unsqueeze_3266);  mul_1351 = unsqueeze_3266 = None
        add_1162 = torch.ops.aten.add.Tensor(add_1161, relu_356);  add_1161 = relu_356 = None
        relu_358 = torch.ops.aten.relu.default(add_1162);  add_1162 = None
        convolution_404 = torch.ops.aten.convolution.default(relu_358, arg396_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg396_1 = None
        add_1163 = torch.ops.aten.add.Tensor(arg398_1, 1e-05);  arg398_1 = None
        sqrt_404 = torch.ops.aten.sqrt.default(add_1163);  add_1163 = None
        reciprocal_404 = torch.ops.aten.reciprocal.default(sqrt_404);  sqrt_404 = None
        mul_1352 = torch.ops.aten.mul.Tensor(reciprocal_404, 1);  reciprocal_404 = None
        unsqueeze_3267 = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_3268 = torch.ops.aten.unsqueeze.default(unsqueeze_3267, -1);  unsqueeze_3267 = None
        unsqueeze_3269 = torch.ops.aten.unsqueeze.default(mul_1352, -1);  mul_1352 = None
        unsqueeze_3270 = torch.ops.aten.unsqueeze.default(unsqueeze_3269, -1);  unsqueeze_3269 = None
        sub_404 = torch.ops.aten.sub.Tensor(convolution_404, unsqueeze_3268);  convolution_404 = unsqueeze_3268 = None
        mul_1353 = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_3270);  sub_404 = unsqueeze_3270 = None
        unsqueeze_3271 = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_3272 = torch.ops.aten.unsqueeze.default(unsqueeze_3271, -1);  unsqueeze_3271 = None
        mul_1354 = torch.ops.aten.mul.Tensor(mul_1353, unsqueeze_3272);  mul_1353 = unsqueeze_3272 = None
        unsqueeze_3273 = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
        unsqueeze_3274 = torch.ops.aten.unsqueeze.default(unsqueeze_3273, -1);  unsqueeze_3273 = None
        add_1164 = torch.ops.aten.add.Tensor(mul_1354, unsqueeze_3274);  mul_1354 = unsqueeze_3274 = None
        relu_359 = torch.ops.aten.relu.default(add_1164);  add_1164 = None
        convolution_405 = torch.ops.aten.convolution.default(relu_359, arg401_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_359 = arg401_1 = None
        add_1165 = torch.ops.aten.add.Tensor(arg403_1, 1e-05);  arg403_1 = None
        sqrt_405 = torch.ops.aten.sqrt.default(add_1165);  add_1165 = None
        reciprocal_405 = torch.ops.aten.reciprocal.default(sqrt_405);  sqrt_405 = None
        mul_1355 = torch.ops.aten.mul.Tensor(reciprocal_405, 1);  reciprocal_405 = None
        unsqueeze_3275 = torch.ops.aten.unsqueeze.default(arg402_1, -1);  arg402_1 = None
        unsqueeze_3276 = torch.ops.aten.unsqueeze.default(unsqueeze_3275, -1);  unsqueeze_3275 = None
        unsqueeze_3277 = torch.ops.aten.unsqueeze.default(mul_1355, -1);  mul_1355 = None
        unsqueeze_3278 = torch.ops.aten.unsqueeze.default(unsqueeze_3277, -1);  unsqueeze_3277 = None
        sub_405 = torch.ops.aten.sub.Tensor(convolution_405, unsqueeze_3276);  convolution_405 = unsqueeze_3276 = None
        mul_1356 = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_3278);  sub_405 = unsqueeze_3278 = None
        unsqueeze_3279 = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_3280 = torch.ops.aten.unsqueeze.default(unsqueeze_3279, -1);  unsqueeze_3279 = None
        mul_1357 = torch.ops.aten.mul.Tensor(mul_1356, unsqueeze_3280);  mul_1356 = unsqueeze_3280 = None
        unsqueeze_3281 = torch.ops.aten.unsqueeze.default(arg405_1, -1);  arg405_1 = None
        unsqueeze_3282 = torch.ops.aten.unsqueeze.default(unsqueeze_3281, -1);  unsqueeze_3281 = None
        add_1166 = torch.ops.aten.add.Tensor(mul_1357, unsqueeze_3282);  mul_1357 = unsqueeze_3282 = None
        add_1167 = torch.ops.aten.add.Tensor(add_1166, relu_358);  add_1166 = relu_358 = None
        relu_360 = torch.ops.aten.relu.default(add_1167);  add_1167 = None
        convolution_406 = torch.ops.aten.convolution.default(relu_360, arg406_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg406_1 = None
        add_1168 = torch.ops.aten.add.Tensor(arg408_1, 1e-05);  arg408_1 = None
        sqrt_406 = torch.ops.aten.sqrt.default(add_1168);  add_1168 = None
        reciprocal_406 = torch.ops.aten.reciprocal.default(sqrt_406);  sqrt_406 = None
        mul_1358 = torch.ops.aten.mul.Tensor(reciprocal_406, 1);  reciprocal_406 = None
        unsqueeze_3283 = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
        unsqueeze_3284 = torch.ops.aten.unsqueeze.default(unsqueeze_3283, -1);  unsqueeze_3283 = None
        unsqueeze_3285 = torch.ops.aten.unsqueeze.default(mul_1358, -1);  mul_1358 = None
        unsqueeze_3286 = torch.ops.aten.unsqueeze.default(unsqueeze_3285, -1);  unsqueeze_3285 = None
        sub_406 = torch.ops.aten.sub.Tensor(convolution_406, unsqueeze_3284);  convolution_406 = unsqueeze_3284 = None
        mul_1359 = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_3286);  sub_406 = unsqueeze_3286 = None
        unsqueeze_3287 = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_3288 = torch.ops.aten.unsqueeze.default(unsqueeze_3287, -1);  unsqueeze_3287 = None
        mul_1360 = torch.ops.aten.mul.Tensor(mul_1359, unsqueeze_3288);  mul_1359 = unsqueeze_3288 = None
        unsqueeze_3289 = torch.ops.aten.unsqueeze.default(arg410_1, -1);  arg410_1 = None
        unsqueeze_3290 = torch.ops.aten.unsqueeze.default(unsqueeze_3289, -1);  unsqueeze_3289 = None
        add_1169 = torch.ops.aten.add.Tensor(mul_1360, unsqueeze_3290);  mul_1360 = unsqueeze_3290 = None
        relu_361 = torch.ops.aten.relu.default(add_1169);  add_1169 = None
        convolution_407 = torch.ops.aten.convolution.default(relu_361, arg411_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_361 = arg411_1 = None
        add_1170 = torch.ops.aten.add.Tensor(arg413_1, 1e-05);  arg413_1 = None
        sqrt_407 = torch.ops.aten.sqrt.default(add_1170);  add_1170 = None
        reciprocal_407 = torch.ops.aten.reciprocal.default(sqrt_407);  sqrt_407 = None
        mul_1361 = torch.ops.aten.mul.Tensor(reciprocal_407, 1);  reciprocal_407 = None
        unsqueeze_3291 = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
        unsqueeze_3292 = torch.ops.aten.unsqueeze.default(unsqueeze_3291, -1);  unsqueeze_3291 = None
        unsqueeze_3293 = torch.ops.aten.unsqueeze.default(mul_1361, -1);  mul_1361 = None
        unsqueeze_3294 = torch.ops.aten.unsqueeze.default(unsqueeze_3293, -1);  unsqueeze_3293 = None
        sub_407 = torch.ops.aten.sub.Tensor(convolution_407, unsqueeze_3292);  convolution_407 = unsqueeze_3292 = None
        mul_1362 = torch.ops.aten.mul.Tensor(sub_407, unsqueeze_3294);  sub_407 = unsqueeze_3294 = None
        unsqueeze_3295 = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_3296 = torch.ops.aten.unsqueeze.default(unsqueeze_3295, -1);  unsqueeze_3295 = None
        mul_1363 = torch.ops.aten.mul.Tensor(mul_1362, unsqueeze_3296);  mul_1362 = unsqueeze_3296 = None
        unsqueeze_3297 = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
        unsqueeze_3298 = torch.ops.aten.unsqueeze.default(unsqueeze_3297, -1);  unsqueeze_3297 = None
        add_1171 = torch.ops.aten.add.Tensor(mul_1363, unsqueeze_3298);  mul_1363 = unsqueeze_3298 = None
        add_1172 = torch.ops.aten.add.Tensor(add_1171, relu_360);  add_1171 = relu_360 = None
        relu_362 = torch.ops.aten.relu.default(add_1172);  add_1172 = None
        convolution_408 = torch.ops.aten.convolution.default(relu_346, arg416_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg416_1 = None
        add_1173 = torch.ops.aten.add.Tensor(arg418_1, 1e-05);  arg418_1 = None
        sqrt_408 = torch.ops.aten.sqrt.default(add_1173);  add_1173 = None
        reciprocal_408 = torch.ops.aten.reciprocal.default(sqrt_408);  sqrt_408 = None
        mul_1364 = torch.ops.aten.mul.Tensor(reciprocal_408, 1);  reciprocal_408 = None
        unsqueeze_3299 = torch.ops.aten.unsqueeze.default(arg417_1, -1);  arg417_1 = None
        unsqueeze_3300 = torch.ops.aten.unsqueeze.default(unsqueeze_3299, -1);  unsqueeze_3299 = None
        unsqueeze_3301 = torch.ops.aten.unsqueeze.default(mul_1364, -1);  mul_1364 = None
        unsqueeze_3302 = torch.ops.aten.unsqueeze.default(unsqueeze_3301, -1);  unsqueeze_3301 = None
        sub_408 = torch.ops.aten.sub.Tensor(convolution_408, unsqueeze_3300);  convolution_408 = unsqueeze_3300 = None
        mul_1365 = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_3302);  sub_408 = unsqueeze_3302 = None
        unsqueeze_3303 = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_3304 = torch.ops.aten.unsqueeze.default(unsqueeze_3303, -1);  unsqueeze_3303 = None
        mul_1366 = torch.ops.aten.mul.Tensor(mul_1365, unsqueeze_3304);  mul_1365 = unsqueeze_3304 = None
        unsqueeze_3305 = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_3306 = torch.ops.aten.unsqueeze.default(unsqueeze_3305, -1);  unsqueeze_3305 = None
        add_1174 = torch.ops.aten.add.Tensor(mul_1366, unsqueeze_3306);  mul_1366 = unsqueeze_3306 = None
        relu_363 = torch.ops.aten.relu.default(add_1174);  add_1174 = None
        convolution_409 = torch.ops.aten.convolution.default(relu_363, arg421_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_363 = arg421_1 = None
        add_1175 = torch.ops.aten.add.Tensor(arg423_1, 1e-05);  arg423_1 = None
        sqrt_409 = torch.ops.aten.sqrt.default(add_1175);  add_1175 = None
        reciprocal_409 = torch.ops.aten.reciprocal.default(sqrt_409);  sqrt_409 = None
        mul_1367 = torch.ops.aten.mul.Tensor(reciprocal_409, 1);  reciprocal_409 = None
        unsqueeze_3307 = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
        unsqueeze_3308 = torch.ops.aten.unsqueeze.default(unsqueeze_3307, -1);  unsqueeze_3307 = None
        unsqueeze_3309 = torch.ops.aten.unsqueeze.default(mul_1367, -1);  mul_1367 = None
        unsqueeze_3310 = torch.ops.aten.unsqueeze.default(unsqueeze_3309, -1);  unsqueeze_3309 = None
        sub_409 = torch.ops.aten.sub.Tensor(convolution_409, unsqueeze_3308);  convolution_409 = unsqueeze_3308 = None
        mul_1368 = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_3310);  sub_409 = unsqueeze_3310 = None
        unsqueeze_3311 = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_3312 = torch.ops.aten.unsqueeze.default(unsqueeze_3311, -1);  unsqueeze_3311 = None
        mul_1369 = torch.ops.aten.mul.Tensor(mul_1368, unsqueeze_3312);  mul_1368 = unsqueeze_3312 = None
        unsqueeze_3313 = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_3314 = torch.ops.aten.unsqueeze.default(unsqueeze_3313, -1);  unsqueeze_3313 = None
        add_1176 = torch.ops.aten.add.Tensor(mul_1369, unsqueeze_3314);  mul_1369 = unsqueeze_3314 = None
        add_1177 = torch.ops.aten.add.Tensor(add_1176, relu_346);  add_1176 = relu_346 = None
        relu_364 = torch.ops.aten.relu.default(add_1177);  add_1177 = None
        convolution_410 = torch.ops.aten.convolution.default(relu_364, arg426_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg426_1 = None
        add_1178 = torch.ops.aten.add.Tensor(arg428_1, 1e-05);  arg428_1 = None
        sqrt_410 = torch.ops.aten.sqrt.default(add_1178);  add_1178 = None
        reciprocal_410 = torch.ops.aten.reciprocal.default(sqrt_410);  sqrt_410 = None
        mul_1370 = torch.ops.aten.mul.Tensor(reciprocal_410, 1);  reciprocal_410 = None
        unsqueeze_3315 = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
        unsqueeze_3316 = torch.ops.aten.unsqueeze.default(unsqueeze_3315, -1);  unsqueeze_3315 = None
        unsqueeze_3317 = torch.ops.aten.unsqueeze.default(mul_1370, -1);  mul_1370 = None
        unsqueeze_3318 = torch.ops.aten.unsqueeze.default(unsqueeze_3317, -1);  unsqueeze_3317 = None
        sub_410 = torch.ops.aten.sub.Tensor(convolution_410, unsqueeze_3316);  convolution_410 = unsqueeze_3316 = None
        mul_1371 = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_3318);  sub_410 = unsqueeze_3318 = None
        unsqueeze_3319 = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_3320 = torch.ops.aten.unsqueeze.default(unsqueeze_3319, -1);  unsqueeze_3319 = None
        mul_1372 = torch.ops.aten.mul.Tensor(mul_1371, unsqueeze_3320);  mul_1371 = unsqueeze_3320 = None
        unsqueeze_3321 = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_3322 = torch.ops.aten.unsqueeze.default(unsqueeze_3321, -1);  unsqueeze_3321 = None
        add_1179 = torch.ops.aten.add.Tensor(mul_1372, unsqueeze_3322);  mul_1372 = unsqueeze_3322 = None
        relu_365 = torch.ops.aten.relu.default(add_1179);  add_1179 = None
        convolution_411 = torch.ops.aten.convolution.default(relu_365, arg431_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_365 = arg431_1 = None
        add_1180 = torch.ops.aten.add.Tensor(arg433_1, 1e-05);  arg433_1 = None
        sqrt_411 = torch.ops.aten.sqrt.default(add_1180);  add_1180 = None
        reciprocal_411 = torch.ops.aten.reciprocal.default(sqrt_411);  sqrt_411 = None
        mul_1373 = torch.ops.aten.mul.Tensor(reciprocal_411, 1);  reciprocal_411 = None
        unsqueeze_3323 = torch.ops.aten.unsqueeze.default(arg432_1, -1);  arg432_1 = None
        unsqueeze_3324 = torch.ops.aten.unsqueeze.default(unsqueeze_3323, -1);  unsqueeze_3323 = None
        unsqueeze_3325 = torch.ops.aten.unsqueeze.default(mul_1373, -1);  mul_1373 = None
        unsqueeze_3326 = torch.ops.aten.unsqueeze.default(unsqueeze_3325, -1);  unsqueeze_3325 = None
        sub_411 = torch.ops.aten.sub.Tensor(convolution_411, unsqueeze_3324);  convolution_411 = unsqueeze_3324 = None
        mul_1374 = torch.ops.aten.mul.Tensor(sub_411, unsqueeze_3326);  sub_411 = unsqueeze_3326 = None
        unsqueeze_3327 = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_3328 = torch.ops.aten.unsqueeze.default(unsqueeze_3327, -1);  unsqueeze_3327 = None
        mul_1375 = torch.ops.aten.mul.Tensor(mul_1374, unsqueeze_3328);  mul_1374 = unsqueeze_3328 = None
        unsqueeze_3329 = torch.ops.aten.unsqueeze.default(arg435_1, -1);  arg435_1 = None
        unsqueeze_3330 = torch.ops.aten.unsqueeze.default(unsqueeze_3329, -1);  unsqueeze_3329 = None
        add_1181 = torch.ops.aten.add.Tensor(mul_1375, unsqueeze_3330);  mul_1375 = unsqueeze_3330 = None
        add_1182 = torch.ops.aten.add.Tensor(add_1181, relu_364);  add_1181 = relu_364 = None
        relu_366 = torch.ops.aten.relu.default(add_1182);  add_1182 = None
        convolution_412 = torch.ops.aten.convolution.default(relu_366, arg436_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg436_1 = None
        add_1183 = torch.ops.aten.add.Tensor(arg438_1, 1e-05);  arg438_1 = None
        sqrt_412 = torch.ops.aten.sqrt.default(add_1183);  add_1183 = None
        reciprocal_412 = torch.ops.aten.reciprocal.default(sqrt_412);  sqrt_412 = None
        mul_1376 = torch.ops.aten.mul.Tensor(reciprocal_412, 1);  reciprocal_412 = None
        unsqueeze_3331 = torch.ops.aten.unsqueeze.default(arg437_1, -1);  arg437_1 = None
        unsqueeze_3332 = torch.ops.aten.unsqueeze.default(unsqueeze_3331, -1);  unsqueeze_3331 = None
        unsqueeze_3333 = torch.ops.aten.unsqueeze.default(mul_1376, -1);  mul_1376 = None
        unsqueeze_3334 = torch.ops.aten.unsqueeze.default(unsqueeze_3333, -1);  unsqueeze_3333 = None
        sub_412 = torch.ops.aten.sub.Tensor(convolution_412, unsqueeze_3332);  convolution_412 = unsqueeze_3332 = None
        mul_1377 = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_3334);  sub_412 = unsqueeze_3334 = None
        unsqueeze_3335 = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
        unsqueeze_3336 = torch.ops.aten.unsqueeze.default(unsqueeze_3335, -1);  unsqueeze_3335 = None
        mul_1378 = torch.ops.aten.mul.Tensor(mul_1377, unsqueeze_3336);  mul_1377 = unsqueeze_3336 = None
        unsqueeze_3337 = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
        unsqueeze_3338 = torch.ops.aten.unsqueeze.default(unsqueeze_3337, -1);  unsqueeze_3337 = None
        add_1184 = torch.ops.aten.add.Tensor(mul_1378, unsqueeze_3338);  mul_1378 = unsqueeze_3338 = None
        relu_367 = torch.ops.aten.relu.default(add_1184);  add_1184 = None
        convolution_413 = torch.ops.aten.convolution.default(relu_367, arg441_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_367 = arg441_1 = None
        add_1185 = torch.ops.aten.add.Tensor(arg443_1, 1e-05);  arg443_1 = None
        sqrt_413 = torch.ops.aten.sqrt.default(add_1185);  add_1185 = None
        reciprocal_413 = torch.ops.aten.reciprocal.default(sqrt_413);  sqrt_413 = None
        mul_1379 = torch.ops.aten.mul.Tensor(reciprocal_413, 1);  reciprocal_413 = None
        unsqueeze_3339 = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
        unsqueeze_3340 = torch.ops.aten.unsqueeze.default(unsqueeze_3339, -1);  unsqueeze_3339 = None
        unsqueeze_3341 = torch.ops.aten.unsqueeze.default(mul_1379, -1);  mul_1379 = None
        unsqueeze_3342 = torch.ops.aten.unsqueeze.default(unsqueeze_3341, -1);  unsqueeze_3341 = None
        sub_413 = torch.ops.aten.sub.Tensor(convolution_413, unsqueeze_3340);  convolution_413 = unsqueeze_3340 = None
        mul_1380 = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_3342);  sub_413 = unsqueeze_3342 = None
        unsqueeze_3343 = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_3344 = torch.ops.aten.unsqueeze.default(unsqueeze_3343, -1);  unsqueeze_3343 = None
        mul_1381 = torch.ops.aten.mul.Tensor(mul_1380, unsqueeze_3344);  mul_1380 = unsqueeze_3344 = None
        unsqueeze_3345 = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
        unsqueeze_3346 = torch.ops.aten.unsqueeze.default(unsqueeze_3345, -1);  unsqueeze_3345 = None
        add_1186 = torch.ops.aten.add.Tensor(mul_1381, unsqueeze_3346);  mul_1381 = unsqueeze_3346 = None
        add_1187 = torch.ops.aten.add.Tensor(add_1186, relu_366);  add_1186 = relu_366 = None
        relu_368 = torch.ops.aten.relu.default(add_1187);  add_1187 = None
        convolution_414 = torch.ops.aten.convolution.default(relu_368, arg446_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg446_1 = None
        add_1188 = torch.ops.aten.add.Tensor(arg448_1, 1e-05);  arg448_1 = None
        sqrt_414 = torch.ops.aten.sqrt.default(add_1188);  add_1188 = None
        reciprocal_414 = torch.ops.aten.reciprocal.default(sqrt_414);  sqrt_414 = None
        mul_1382 = torch.ops.aten.mul.Tensor(reciprocal_414, 1);  reciprocal_414 = None
        unsqueeze_3347 = torch.ops.aten.unsqueeze.default(arg447_1, -1);  arg447_1 = None
        unsqueeze_3348 = torch.ops.aten.unsqueeze.default(unsqueeze_3347, -1);  unsqueeze_3347 = None
        unsqueeze_3349 = torch.ops.aten.unsqueeze.default(mul_1382, -1);  mul_1382 = None
        unsqueeze_3350 = torch.ops.aten.unsqueeze.default(unsqueeze_3349, -1);  unsqueeze_3349 = None
        sub_414 = torch.ops.aten.sub.Tensor(convolution_414, unsqueeze_3348);  convolution_414 = unsqueeze_3348 = None
        mul_1383 = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_3350);  sub_414 = unsqueeze_3350 = None
        unsqueeze_3351 = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_3352 = torch.ops.aten.unsqueeze.default(unsqueeze_3351, -1);  unsqueeze_3351 = None
        mul_1384 = torch.ops.aten.mul.Tensor(mul_1383, unsqueeze_3352);  mul_1383 = unsqueeze_3352 = None
        unsqueeze_3353 = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
        unsqueeze_3354 = torch.ops.aten.unsqueeze.default(unsqueeze_3353, -1);  unsqueeze_3353 = None
        add_1189 = torch.ops.aten.add.Tensor(mul_1384, unsqueeze_3354);  mul_1384 = unsqueeze_3354 = None
        relu_369 = torch.ops.aten.relu.default(add_1189);  add_1189 = None
        convolution_415 = torch.ops.aten.convolution.default(relu_369, arg451_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_369 = arg451_1 = None
        add_1190 = torch.ops.aten.add.Tensor(arg453_1, 1e-05);  arg453_1 = None
        sqrt_415 = torch.ops.aten.sqrt.default(add_1190);  add_1190 = None
        reciprocal_415 = torch.ops.aten.reciprocal.default(sqrt_415);  sqrt_415 = None
        mul_1385 = torch.ops.aten.mul.Tensor(reciprocal_415, 1);  reciprocal_415 = None
        unsqueeze_3355 = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
        unsqueeze_3356 = torch.ops.aten.unsqueeze.default(unsqueeze_3355, -1);  unsqueeze_3355 = None
        unsqueeze_3357 = torch.ops.aten.unsqueeze.default(mul_1385, -1);  mul_1385 = None
        unsqueeze_3358 = torch.ops.aten.unsqueeze.default(unsqueeze_3357, -1);  unsqueeze_3357 = None
        sub_415 = torch.ops.aten.sub.Tensor(convolution_415, unsqueeze_3356);  convolution_415 = unsqueeze_3356 = None
        mul_1386 = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_3358);  sub_415 = unsqueeze_3358 = None
        unsqueeze_3359 = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_3360 = torch.ops.aten.unsqueeze.default(unsqueeze_3359, -1);  unsqueeze_3359 = None
        mul_1387 = torch.ops.aten.mul.Tensor(mul_1386, unsqueeze_3360);  mul_1386 = unsqueeze_3360 = None
        unsqueeze_3361 = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_3362 = torch.ops.aten.unsqueeze.default(unsqueeze_3361, -1);  unsqueeze_3361 = None
        add_1191 = torch.ops.aten.add.Tensor(mul_1387, unsqueeze_3362);  mul_1387 = unsqueeze_3362 = None
        add_1192 = torch.ops.aten.add.Tensor(add_1191, relu_368);  add_1191 = relu_368 = None
        relu_370 = torch.ops.aten.relu.default(add_1192);  add_1192 = None
        convolution_416 = torch.ops.aten.convolution.default(relu_362, arg456_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg456_1 = None
        add_1193 = torch.ops.aten.add.Tensor(arg458_1, 1e-05);  arg458_1 = None
        sqrt_416 = torch.ops.aten.sqrt.default(add_1193);  add_1193 = None
        reciprocal_416 = torch.ops.aten.reciprocal.default(sqrt_416);  sqrt_416 = None
        mul_1388 = torch.ops.aten.mul.Tensor(reciprocal_416, 1);  reciprocal_416 = None
        unsqueeze_3363 = torch.ops.aten.unsqueeze.default(arg457_1, -1);  arg457_1 = None
        unsqueeze_3364 = torch.ops.aten.unsqueeze.default(unsqueeze_3363, -1);  unsqueeze_3363 = None
        unsqueeze_3365 = torch.ops.aten.unsqueeze.default(mul_1388, -1);  mul_1388 = None
        unsqueeze_3366 = torch.ops.aten.unsqueeze.default(unsqueeze_3365, -1);  unsqueeze_3365 = None
        sub_416 = torch.ops.aten.sub.Tensor(convolution_416, unsqueeze_3364);  convolution_416 = unsqueeze_3364 = None
        mul_1389 = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_3366);  sub_416 = unsqueeze_3366 = None
        unsqueeze_3367 = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_3368 = torch.ops.aten.unsqueeze.default(unsqueeze_3367, -1);  unsqueeze_3367 = None
        mul_1390 = torch.ops.aten.mul.Tensor(mul_1389, unsqueeze_3368);  mul_1389 = unsqueeze_3368 = None
        unsqueeze_3369 = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_3370 = torch.ops.aten.unsqueeze.default(unsqueeze_3369, -1);  unsqueeze_3369 = None
        add_1194 = torch.ops.aten.add.Tensor(mul_1390, unsqueeze_3370);  mul_1390 = unsqueeze_3370 = None
        iota_70 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1391 = torch.ops.aten.mul.Tensor(iota_70, 1);  iota_70 = None
        add_1195 = torch.ops.aten.add.Tensor(mul_1391, 0);  mul_1391 = None
        convert_element_type_974 = torch.ops.prims.convert_element_type.default(add_1195, torch.float32);  add_1195 = None
        add_1196 = torch.ops.aten.add.Tensor(convert_element_type_974, 0.0);  convert_element_type_974 = None
        mul_1392 = torch.ops.aten.mul.Tensor(add_1196, 0.5);  add_1196 = None
        convert_element_type_975 = torch.ops.prims.convert_element_type.default(mul_1392, torch.int64);  mul_1392 = None
        unsqueeze_3371 = torch.ops.aten.unsqueeze.default(convert_element_type_975, -1);  convert_element_type_975 = None
        iota_71 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1393 = torch.ops.aten.mul.Tensor(iota_71, 1);  iota_71 = None
        add_1197 = torch.ops.aten.add.Tensor(mul_1393, 0);  mul_1393 = None
        convert_element_type_976 = torch.ops.prims.convert_element_type.default(add_1197, torch.float32);  add_1197 = None
        add_1198 = torch.ops.aten.add.Tensor(convert_element_type_976, 0.0);  convert_element_type_976 = None
        mul_1394 = torch.ops.aten.mul.Tensor(add_1198, 0.5);  add_1198 = None
        convert_element_type_977 = torch.ops.prims.convert_element_type.default(mul_1394, torch.int64);  mul_1394 = None
        _unsafe_index_35 = torch.ops.aten._unsafe_index.Tensor(add_1194, [None, None, unsqueeze_3371, convert_element_type_977]);  add_1194 = unsqueeze_3371 = convert_element_type_977 = None
        add_1199 = torch.ops.aten.add.Tensor(relu_354, _unsafe_index_35);  _unsafe_index_35 = None
        convolution_417 = torch.ops.aten.convolution.default(relu_370, arg461_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg461_1 = None
        add_1200 = torch.ops.aten.add.Tensor(arg463_1, 1e-05);  arg463_1 = None
        sqrt_417 = torch.ops.aten.sqrt.default(add_1200);  add_1200 = None
        reciprocal_417 = torch.ops.aten.reciprocal.default(sqrt_417);  sqrt_417 = None
        mul_1395 = torch.ops.aten.mul.Tensor(reciprocal_417, 1);  reciprocal_417 = None
        unsqueeze_3372 = torch.ops.aten.unsqueeze.default(arg462_1, -1);  arg462_1 = None
        unsqueeze_3373 = torch.ops.aten.unsqueeze.default(unsqueeze_3372, -1);  unsqueeze_3372 = None
        unsqueeze_3374 = torch.ops.aten.unsqueeze.default(mul_1395, -1);  mul_1395 = None
        unsqueeze_3375 = torch.ops.aten.unsqueeze.default(unsqueeze_3374, -1);  unsqueeze_3374 = None
        sub_417 = torch.ops.aten.sub.Tensor(convolution_417, unsqueeze_3373);  convolution_417 = unsqueeze_3373 = None
        mul_1396 = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_3375);  sub_417 = unsqueeze_3375 = None
        unsqueeze_3376 = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_3377 = torch.ops.aten.unsqueeze.default(unsqueeze_3376, -1);  unsqueeze_3376 = None
        mul_1397 = torch.ops.aten.mul.Tensor(mul_1396, unsqueeze_3377);  mul_1396 = unsqueeze_3377 = None
        unsqueeze_3378 = torch.ops.aten.unsqueeze.default(arg465_1, -1);  arg465_1 = None
        unsqueeze_3379 = torch.ops.aten.unsqueeze.default(unsqueeze_3378, -1);  unsqueeze_3378 = None
        add_1201 = torch.ops.aten.add.Tensor(mul_1397, unsqueeze_3379);  mul_1397 = unsqueeze_3379 = None
        iota_72 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1398 = torch.ops.aten.mul.Tensor(iota_72, 1);  iota_72 = None
        add_1202 = torch.ops.aten.add.Tensor(mul_1398, 0);  mul_1398 = None
        convert_element_type_980 = torch.ops.prims.convert_element_type.default(add_1202, torch.float32);  add_1202 = None
        add_1203 = torch.ops.aten.add.Tensor(convert_element_type_980, 0.0);  convert_element_type_980 = None
        mul_1399 = torch.ops.aten.mul.Tensor(add_1203, 0.25);  add_1203 = None
        convert_element_type_981 = torch.ops.prims.convert_element_type.default(mul_1399, torch.int64);  mul_1399 = None
        unsqueeze_3380 = torch.ops.aten.unsqueeze.default(convert_element_type_981, -1);  convert_element_type_981 = None
        iota_73 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1400 = torch.ops.aten.mul.Tensor(iota_73, 1);  iota_73 = None
        add_1204 = torch.ops.aten.add.Tensor(mul_1400, 0);  mul_1400 = None
        convert_element_type_982 = torch.ops.prims.convert_element_type.default(add_1204, torch.float32);  add_1204 = None
        add_1205 = torch.ops.aten.add.Tensor(convert_element_type_982, 0.0);  convert_element_type_982 = None
        mul_1401 = torch.ops.aten.mul.Tensor(add_1205, 0.25);  add_1205 = None
        convert_element_type_983 = torch.ops.prims.convert_element_type.default(mul_1401, torch.int64);  mul_1401 = None
        _unsafe_index_36 = torch.ops.aten._unsafe_index.Tensor(add_1201, [None, None, unsqueeze_3380, convert_element_type_983]);  add_1201 = unsqueeze_3380 = convert_element_type_983 = None
        add_1206 = torch.ops.aten.add.Tensor(add_1199, _unsafe_index_36);  add_1199 = _unsafe_index_36 = None
        relu_371 = torch.ops.aten.relu.default(add_1206);  add_1206 = None
        convolution_418 = torch.ops.aten.convolution.default(relu_354, arg466_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg466_1 = None
        add_1207 = torch.ops.aten.add.Tensor(arg468_1, 1e-05);  arg468_1 = None
        sqrt_418 = torch.ops.aten.sqrt.default(add_1207);  add_1207 = None
        reciprocal_418 = torch.ops.aten.reciprocal.default(sqrt_418);  sqrt_418 = None
        mul_1402 = torch.ops.aten.mul.Tensor(reciprocal_418, 1);  reciprocal_418 = None
        unsqueeze_3381 = torch.ops.aten.unsqueeze.default(arg467_1, -1);  arg467_1 = None
        unsqueeze_3382 = torch.ops.aten.unsqueeze.default(unsqueeze_3381, -1);  unsqueeze_3381 = None
        unsqueeze_3383 = torch.ops.aten.unsqueeze.default(mul_1402, -1);  mul_1402 = None
        unsqueeze_3384 = torch.ops.aten.unsqueeze.default(unsqueeze_3383, -1);  unsqueeze_3383 = None
        sub_418 = torch.ops.aten.sub.Tensor(convolution_418, unsqueeze_3382);  convolution_418 = unsqueeze_3382 = None
        mul_1403 = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_3384);  sub_418 = unsqueeze_3384 = None
        unsqueeze_3385 = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_3386 = torch.ops.aten.unsqueeze.default(unsqueeze_3385, -1);  unsqueeze_3385 = None
        mul_1404 = torch.ops.aten.mul.Tensor(mul_1403, unsqueeze_3386);  mul_1403 = unsqueeze_3386 = None
        unsqueeze_3387 = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
        unsqueeze_3388 = torch.ops.aten.unsqueeze.default(unsqueeze_3387, -1);  unsqueeze_3387 = None
        add_1208 = torch.ops.aten.add.Tensor(mul_1404, unsqueeze_3388);  mul_1404 = unsqueeze_3388 = None
        add_1209 = torch.ops.aten.add.Tensor(add_1208, relu_362);  add_1208 = None
        convolution_419 = torch.ops.aten.convolution.default(relu_370, arg471_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg471_1 = None
        add_1210 = torch.ops.aten.add.Tensor(arg473_1, 1e-05);  arg473_1 = None
        sqrt_419 = torch.ops.aten.sqrt.default(add_1210);  add_1210 = None
        reciprocal_419 = torch.ops.aten.reciprocal.default(sqrt_419);  sqrt_419 = None
        mul_1405 = torch.ops.aten.mul.Tensor(reciprocal_419, 1);  reciprocal_419 = None
        unsqueeze_3389 = torch.ops.aten.unsqueeze.default(arg472_1, -1);  arg472_1 = None
        unsqueeze_3390 = torch.ops.aten.unsqueeze.default(unsqueeze_3389, -1);  unsqueeze_3389 = None
        unsqueeze_3391 = torch.ops.aten.unsqueeze.default(mul_1405, -1);  mul_1405 = None
        unsqueeze_3392 = torch.ops.aten.unsqueeze.default(unsqueeze_3391, -1);  unsqueeze_3391 = None
        sub_419 = torch.ops.aten.sub.Tensor(convolution_419, unsqueeze_3390);  convolution_419 = unsqueeze_3390 = None
        mul_1406 = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_3392);  sub_419 = unsqueeze_3392 = None
        unsqueeze_3393 = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_3394 = torch.ops.aten.unsqueeze.default(unsqueeze_3393, -1);  unsqueeze_3393 = None
        mul_1407 = torch.ops.aten.mul.Tensor(mul_1406, unsqueeze_3394);  mul_1406 = unsqueeze_3394 = None
        unsqueeze_3395 = torch.ops.aten.unsqueeze.default(arg475_1, -1);  arg475_1 = None
        unsqueeze_3396 = torch.ops.aten.unsqueeze.default(unsqueeze_3395, -1);  unsqueeze_3395 = None
        add_1211 = torch.ops.aten.add.Tensor(mul_1407, unsqueeze_3396);  mul_1407 = unsqueeze_3396 = None
        iota_74 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1408 = torch.ops.aten.mul.Tensor(iota_74, 1);  iota_74 = None
        add_1212 = torch.ops.aten.add.Tensor(mul_1408, 0);  mul_1408 = None
        convert_element_type_988 = torch.ops.prims.convert_element_type.default(add_1212, torch.float32);  add_1212 = None
        add_1213 = torch.ops.aten.add.Tensor(convert_element_type_988, 0.0);  convert_element_type_988 = None
        mul_1409 = torch.ops.aten.mul.Tensor(add_1213, 0.5);  add_1213 = None
        convert_element_type_989 = torch.ops.prims.convert_element_type.default(mul_1409, torch.int64);  mul_1409 = None
        unsqueeze_3397 = torch.ops.aten.unsqueeze.default(convert_element_type_989, -1);  convert_element_type_989 = None
        iota_75 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1410 = torch.ops.aten.mul.Tensor(iota_75, 1);  iota_75 = None
        add_1214 = torch.ops.aten.add.Tensor(mul_1410, 0);  mul_1410 = None
        convert_element_type_990 = torch.ops.prims.convert_element_type.default(add_1214, torch.float32);  add_1214 = None
        add_1215 = torch.ops.aten.add.Tensor(convert_element_type_990, 0.0);  convert_element_type_990 = None
        mul_1411 = torch.ops.aten.mul.Tensor(add_1215, 0.5);  add_1215 = None
        convert_element_type_991 = torch.ops.prims.convert_element_type.default(mul_1411, torch.int64);  mul_1411 = None
        _unsafe_index_37 = torch.ops.aten._unsafe_index.Tensor(add_1211, [None, None, unsqueeze_3397, convert_element_type_991]);  add_1211 = unsqueeze_3397 = convert_element_type_991 = None
        add_1216 = torch.ops.aten.add.Tensor(add_1209, _unsafe_index_37);  add_1209 = _unsafe_index_37 = None
        relu_372 = torch.ops.aten.relu.default(add_1216);  add_1216 = None
        convolution_420 = torch.ops.aten.convolution.default(relu_354, arg476_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_354 = arg476_1 = None
        add_1217 = torch.ops.aten.add.Tensor(arg478_1, 1e-05);  arg478_1 = None
        sqrt_420 = torch.ops.aten.sqrt.default(add_1217);  add_1217 = None
        reciprocal_420 = torch.ops.aten.reciprocal.default(sqrt_420);  sqrt_420 = None
        mul_1412 = torch.ops.aten.mul.Tensor(reciprocal_420, 1);  reciprocal_420 = None
        unsqueeze_3398 = torch.ops.aten.unsqueeze.default(arg477_1, -1);  arg477_1 = None
        unsqueeze_3399 = torch.ops.aten.unsqueeze.default(unsqueeze_3398, -1);  unsqueeze_3398 = None
        unsqueeze_3400 = torch.ops.aten.unsqueeze.default(mul_1412, -1);  mul_1412 = None
        unsqueeze_3401 = torch.ops.aten.unsqueeze.default(unsqueeze_3400, -1);  unsqueeze_3400 = None
        sub_420 = torch.ops.aten.sub.Tensor(convolution_420, unsqueeze_3399);  convolution_420 = unsqueeze_3399 = None
        mul_1413 = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_3401);  sub_420 = unsqueeze_3401 = None
        unsqueeze_3402 = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
        unsqueeze_3403 = torch.ops.aten.unsqueeze.default(unsqueeze_3402, -1);  unsqueeze_3402 = None
        mul_1414 = torch.ops.aten.mul.Tensor(mul_1413, unsqueeze_3403);  mul_1413 = unsqueeze_3403 = None
        unsqueeze_3404 = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
        unsqueeze_3405 = torch.ops.aten.unsqueeze.default(unsqueeze_3404, -1);  unsqueeze_3404 = None
        add_1218 = torch.ops.aten.add.Tensor(mul_1414, unsqueeze_3405);  mul_1414 = unsqueeze_3405 = None
        relu_373 = torch.ops.aten.relu.default(add_1218);  add_1218 = None
        convolution_421 = torch.ops.aten.convolution.default(relu_373, arg481_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_373 = arg481_1 = None
        add_1219 = torch.ops.aten.add.Tensor(arg483_1, 1e-05);  arg483_1 = None
        sqrt_421 = torch.ops.aten.sqrt.default(add_1219);  add_1219 = None
        reciprocal_421 = torch.ops.aten.reciprocal.default(sqrt_421);  sqrt_421 = None
        mul_1415 = torch.ops.aten.mul.Tensor(reciprocal_421, 1);  reciprocal_421 = None
        unsqueeze_3406 = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
        unsqueeze_3407 = torch.ops.aten.unsqueeze.default(unsqueeze_3406, -1);  unsqueeze_3406 = None
        unsqueeze_3408 = torch.ops.aten.unsqueeze.default(mul_1415, -1);  mul_1415 = None
        unsqueeze_3409 = torch.ops.aten.unsqueeze.default(unsqueeze_3408, -1);  unsqueeze_3408 = None
        sub_421 = torch.ops.aten.sub.Tensor(convolution_421, unsqueeze_3407);  convolution_421 = unsqueeze_3407 = None
        mul_1416 = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_3409);  sub_421 = unsqueeze_3409 = None
        unsqueeze_3410 = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
        unsqueeze_3411 = torch.ops.aten.unsqueeze.default(unsqueeze_3410, -1);  unsqueeze_3410 = None
        mul_1417 = torch.ops.aten.mul.Tensor(mul_1416, unsqueeze_3411);  mul_1416 = unsqueeze_3411 = None
        unsqueeze_3412 = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
        unsqueeze_3413 = torch.ops.aten.unsqueeze.default(unsqueeze_3412, -1);  unsqueeze_3412 = None
        add_1220 = torch.ops.aten.add.Tensor(mul_1417, unsqueeze_3413);  mul_1417 = unsqueeze_3413 = None
        convolution_422 = torch.ops.aten.convolution.default(relu_362, arg486_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_362 = arg486_1 = None
        add_1221 = torch.ops.aten.add.Tensor(arg488_1, 1e-05);  arg488_1 = None
        sqrt_422 = torch.ops.aten.sqrt.default(add_1221);  add_1221 = None
        reciprocal_422 = torch.ops.aten.reciprocal.default(sqrt_422);  sqrt_422 = None
        mul_1418 = torch.ops.aten.mul.Tensor(reciprocal_422, 1);  reciprocal_422 = None
        unsqueeze_3414 = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
        unsqueeze_3415 = torch.ops.aten.unsqueeze.default(unsqueeze_3414, -1);  unsqueeze_3414 = None
        unsqueeze_3416 = torch.ops.aten.unsqueeze.default(mul_1418, -1);  mul_1418 = None
        unsqueeze_3417 = torch.ops.aten.unsqueeze.default(unsqueeze_3416, -1);  unsqueeze_3416 = None
        sub_422 = torch.ops.aten.sub.Tensor(convolution_422, unsqueeze_3415);  convolution_422 = unsqueeze_3415 = None
        mul_1419 = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_3417);  sub_422 = unsqueeze_3417 = None
        unsqueeze_3418 = torch.ops.aten.unsqueeze.default(arg489_1, -1);  arg489_1 = None
        unsqueeze_3419 = torch.ops.aten.unsqueeze.default(unsqueeze_3418, -1);  unsqueeze_3418 = None
        mul_1420 = torch.ops.aten.mul.Tensor(mul_1419, unsqueeze_3419);  mul_1419 = unsqueeze_3419 = None
        unsqueeze_3420 = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_3421 = torch.ops.aten.unsqueeze.default(unsqueeze_3420, -1);  unsqueeze_3420 = None
        add_1222 = torch.ops.aten.add.Tensor(mul_1420, unsqueeze_3421);  mul_1420 = unsqueeze_3421 = None
        add_1223 = torch.ops.aten.add.Tensor(add_1220, add_1222);  add_1220 = add_1222 = None
        add_1224 = torch.ops.aten.add.Tensor(add_1223, relu_370);  add_1223 = relu_370 = None
        relu_374 = torch.ops.aten.relu.default(add_1224);  add_1224 = None
        convolution_423 = torch.ops.aten.convolution.default(relu_371, arg491_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg491_1 = None
        add_1225 = torch.ops.aten.add.Tensor(arg493_1, 1e-05);  arg493_1 = None
        sqrt_423 = torch.ops.aten.sqrt.default(add_1225);  add_1225 = None
        reciprocal_423 = torch.ops.aten.reciprocal.default(sqrt_423);  sqrt_423 = None
        mul_1421 = torch.ops.aten.mul.Tensor(reciprocal_423, 1);  reciprocal_423 = None
        unsqueeze_3422 = torch.ops.aten.unsqueeze.default(arg492_1, -1);  arg492_1 = None
        unsqueeze_3423 = torch.ops.aten.unsqueeze.default(unsqueeze_3422, -1);  unsqueeze_3422 = None
        unsqueeze_3424 = torch.ops.aten.unsqueeze.default(mul_1421, -1);  mul_1421 = None
        unsqueeze_3425 = torch.ops.aten.unsqueeze.default(unsqueeze_3424, -1);  unsqueeze_3424 = None
        sub_423 = torch.ops.aten.sub.Tensor(convolution_423, unsqueeze_3423);  convolution_423 = unsqueeze_3423 = None
        mul_1422 = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_3425);  sub_423 = unsqueeze_3425 = None
        unsqueeze_3426 = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_3427 = torch.ops.aten.unsqueeze.default(unsqueeze_3426, -1);  unsqueeze_3426 = None
        mul_1423 = torch.ops.aten.mul.Tensor(mul_1422, unsqueeze_3427);  mul_1422 = unsqueeze_3427 = None
        unsqueeze_3428 = torch.ops.aten.unsqueeze.default(arg495_1, -1);  arg495_1 = None
        unsqueeze_3429 = torch.ops.aten.unsqueeze.default(unsqueeze_3428, -1);  unsqueeze_3428 = None
        add_1226 = torch.ops.aten.add.Tensor(mul_1423, unsqueeze_3429);  mul_1423 = unsqueeze_3429 = None
        relu_375 = torch.ops.aten.relu.default(add_1226);  add_1226 = None
        convolution_424 = torch.ops.aten.convolution.default(relu_375, arg496_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_375 = arg496_1 = None
        add_1227 = torch.ops.aten.add.Tensor(arg498_1, 1e-05);  arg498_1 = None
        sqrt_424 = torch.ops.aten.sqrt.default(add_1227);  add_1227 = None
        reciprocal_424 = torch.ops.aten.reciprocal.default(sqrt_424);  sqrt_424 = None
        mul_1424 = torch.ops.aten.mul.Tensor(reciprocal_424, 1);  reciprocal_424 = None
        unsqueeze_3430 = torch.ops.aten.unsqueeze.default(arg497_1, -1);  arg497_1 = None
        unsqueeze_3431 = torch.ops.aten.unsqueeze.default(unsqueeze_3430, -1);  unsqueeze_3430 = None
        unsqueeze_3432 = torch.ops.aten.unsqueeze.default(mul_1424, -1);  mul_1424 = None
        unsqueeze_3433 = torch.ops.aten.unsqueeze.default(unsqueeze_3432, -1);  unsqueeze_3432 = None
        sub_424 = torch.ops.aten.sub.Tensor(convolution_424, unsqueeze_3431);  convolution_424 = unsqueeze_3431 = None
        mul_1425 = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_3433);  sub_424 = unsqueeze_3433 = None
        unsqueeze_3434 = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_3435 = torch.ops.aten.unsqueeze.default(unsqueeze_3434, -1);  unsqueeze_3434 = None
        mul_1426 = torch.ops.aten.mul.Tensor(mul_1425, unsqueeze_3435);  mul_1425 = unsqueeze_3435 = None
        unsqueeze_3436 = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
        unsqueeze_3437 = torch.ops.aten.unsqueeze.default(unsqueeze_3436, -1);  unsqueeze_3436 = None
        add_1228 = torch.ops.aten.add.Tensor(mul_1426, unsqueeze_3437);  mul_1426 = unsqueeze_3437 = None
        add_1229 = torch.ops.aten.add.Tensor(add_1228, relu_371);  add_1228 = relu_371 = None
        relu_376 = torch.ops.aten.relu.default(add_1229);  add_1229 = None
        convolution_425 = torch.ops.aten.convolution.default(relu_376, arg501_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg501_1 = None
        add_1230 = torch.ops.aten.add.Tensor(arg503_1, 1e-05);  arg503_1 = None
        sqrt_425 = torch.ops.aten.sqrt.default(add_1230);  add_1230 = None
        reciprocal_425 = torch.ops.aten.reciprocal.default(sqrt_425);  sqrt_425 = None
        mul_1427 = torch.ops.aten.mul.Tensor(reciprocal_425, 1);  reciprocal_425 = None
        unsqueeze_3438 = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
        unsqueeze_3439 = torch.ops.aten.unsqueeze.default(unsqueeze_3438, -1);  unsqueeze_3438 = None
        unsqueeze_3440 = torch.ops.aten.unsqueeze.default(mul_1427, -1);  mul_1427 = None
        unsqueeze_3441 = torch.ops.aten.unsqueeze.default(unsqueeze_3440, -1);  unsqueeze_3440 = None
        sub_425 = torch.ops.aten.sub.Tensor(convolution_425, unsqueeze_3439);  convolution_425 = unsqueeze_3439 = None
        mul_1428 = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_3441);  sub_425 = unsqueeze_3441 = None
        unsqueeze_3442 = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
        unsqueeze_3443 = torch.ops.aten.unsqueeze.default(unsqueeze_3442, -1);  unsqueeze_3442 = None
        mul_1429 = torch.ops.aten.mul.Tensor(mul_1428, unsqueeze_3443);  mul_1428 = unsqueeze_3443 = None
        unsqueeze_3444 = torch.ops.aten.unsqueeze.default(arg505_1, -1);  arg505_1 = None
        unsqueeze_3445 = torch.ops.aten.unsqueeze.default(unsqueeze_3444, -1);  unsqueeze_3444 = None
        add_1231 = torch.ops.aten.add.Tensor(mul_1429, unsqueeze_3445);  mul_1429 = unsqueeze_3445 = None
        relu_377 = torch.ops.aten.relu.default(add_1231);  add_1231 = None
        convolution_426 = torch.ops.aten.convolution.default(relu_377, arg506_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_377 = arg506_1 = None
        add_1232 = torch.ops.aten.add.Tensor(arg508_1, 1e-05);  arg508_1 = None
        sqrt_426 = torch.ops.aten.sqrt.default(add_1232);  add_1232 = None
        reciprocal_426 = torch.ops.aten.reciprocal.default(sqrt_426);  sqrt_426 = None
        mul_1430 = torch.ops.aten.mul.Tensor(reciprocal_426, 1);  reciprocal_426 = None
        unsqueeze_3446 = torch.ops.aten.unsqueeze.default(arg507_1, -1);  arg507_1 = None
        unsqueeze_3447 = torch.ops.aten.unsqueeze.default(unsqueeze_3446, -1);  unsqueeze_3446 = None
        unsqueeze_3448 = torch.ops.aten.unsqueeze.default(mul_1430, -1);  mul_1430 = None
        unsqueeze_3449 = torch.ops.aten.unsqueeze.default(unsqueeze_3448, -1);  unsqueeze_3448 = None
        sub_426 = torch.ops.aten.sub.Tensor(convolution_426, unsqueeze_3447);  convolution_426 = unsqueeze_3447 = None
        mul_1431 = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_3449);  sub_426 = unsqueeze_3449 = None
        unsqueeze_3450 = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
        unsqueeze_3451 = torch.ops.aten.unsqueeze.default(unsqueeze_3450, -1);  unsqueeze_3450 = None
        mul_1432 = torch.ops.aten.mul.Tensor(mul_1431, unsqueeze_3451);  mul_1431 = unsqueeze_3451 = None
        unsqueeze_3452 = torch.ops.aten.unsqueeze.default(arg510_1, -1);  arg510_1 = None
        unsqueeze_3453 = torch.ops.aten.unsqueeze.default(unsqueeze_3452, -1);  unsqueeze_3452 = None
        add_1233 = torch.ops.aten.add.Tensor(mul_1432, unsqueeze_3453);  mul_1432 = unsqueeze_3453 = None
        add_1234 = torch.ops.aten.add.Tensor(add_1233, relu_376);  add_1233 = relu_376 = None
        relu_378 = torch.ops.aten.relu.default(add_1234);  add_1234 = None
        convolution_427 = torch.ops.aten.convolution.default(relu_378, arg511_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg511_1 = None
        add_1235 = torch.ops.aten.add.Tensor(arg513_1, 1e-05);  arg513_1 = None
        sqrt_427 = torch.ops.aten.sqrt.default(add_1235);  add_1235 = None
        reciprocal_427 = torch.ops.aten.reciprocal.default(sqrt_427);  sqrt_427 = None
        mul_1433 = torch.ops.aten.mul.Tensor(reciprocal_427, 1);  reciprocal_427 = None
        unsqueeze_3454 = torch.ops.aten.unsqueeze.default(arg512_1, -1);  arg512_1 = None
        unsqueeze_3455 = torch.ops.aten.unsqueeze.default(unsqueeze_3454, -1);  unsqueeze_3454 = None
        unsqueeze_3456 = torch.ops.aten.unsqueeze.default(mul_1433, -1);  mul_1433 = None
        unsqueeze_3457 = torch.ops.aten.unsqueeze.default(unsqueeze_3456, -1);  unsqueeze_3456 = None
        sub_427 = torch.ops.aten.sub.Tensor(convolution_427, unsqueeze_3455);  convolution_427 = unsqueeze_3455 = None
        mul_1434 = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_3457);  sub_427 = unsqueeze_3457 = None
        unsqueeze_3458 = torch.ops.aten.unsqueeze.default(arg514_1, -1);  arg514_1 = None
        unsqueeze_3459 = torch.ops.aten.unsqueeze.default(unsqueeze_3458, -1);  unsqueeze_3458 = None
        mul_1435 = torch.ops.aten.mul.Tensor(mul_1434, unsqueeze_3459);  mul_1434 = unsqueeze_3459 = None
        unsqueeze_3460 = torch.ops.aten.unsqueeze.default(arg515_1, -1);  arg515_1 = None
        unsqueeze_3461 = torch.ops.aten.unsqueeze.default(unsqueeze_3460, -1);  unsqueeze_3460 = None
        add_1236 = torch.ops.aten.add.Tensor(mul_1435, unsqueeze_3461);  mul_1435 = unsqueeze_3461 = None
        relu_379 = torch.ops.aten.relu.default(add_1236);  add_1236 = None
        convolution_428 = torch.ops.aten.convolution.default(relu_379, arg516_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_379 = arg516_1 = None
        add_1237 = torch.ops.aten.add.Tensor(arg518_1, 1e-05);  arg518_1 = None
        sqrt_428 = torch.ops.aten.sqrt.default(add_1237);  add_1237 = None
        reciprocal_428 = torch.ops.aten.reciprocal.default(sqrt_428);  sqrt_428 = None
        mul_1436 = torch.ops.aten.mul.Tensor(reciprocal_428, 1);  reciprocal_428 = None
        unsqueeze_3462 = torch.ops.aten.unsqueeze.default(arg517_1, -1);  arg517_1 = None
        unsqueeze_3463 = torch.ops.aten.unsqueeze.default(unsqueeze_3462, -1);  unsqueeze_3462 = None
        unsqueeze_3464 = torch.ops.aten.unsqueeze.default(mul_1436, -1);  mul_1436 = None
        unsqueeze_3465 = torch.ops.aten.unsqueeze.default(unsqueeze_3464, -1);  unsqueeze_3464 = None
        sub_428 = torch.ops.aten.sub.Tensor(convolution_428, unsqueeze_3463);  convolution_428 = unsqueeze_3463 = None
        mul_1437 = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_3465);  sub_428 = unsqueeze_3465 = None
        unsqueeze_3466 = torch.ops.aten.unsqueeze.default(arg519_1, -1);  arg519_1 = None
        unsqueeze_3467 = torch.ops.aten.unsqueeze.default(unsqueeze_3466, -1);  unsqueeze_3466 = None
        mul_1438 = torch.ops.aten.mul.Tensor(mul_1437, unsqueeze_3467);  mul_1437 = unsqueeze_3467 = None
        unsqueeze_3468 = torch.ops.aten.unsqueeze.default(arg520_1, -1);  arg520_1 = None
        unsqueeze_3469 = torch.ops.aten.unsqueeze.default(unsqueeze_3468, -1);  unsqueeze_3468 = None
        add_1238 = torch.ops.aten.add.Tensor(mul_1438, unsqueeze_3469);  mul_1438 = unsqueeze_3469 = None
        add_1239 = torch.ops.aten.add.Tensor(add_1238, relu_378);  add_1238 = relu_378 = None
        relu_380 = torch.ops.aten.relu.default(add_1239);  add_1239 = None
        convolution_429 = torch.ops.aten.convolution.default(relu_380, arg521_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg521_1 = None
        add_1240 = torch.ops.aten.add.Tensor(arg523_1, 1e-05);  arg523_1 = None
        sqrt_429 = torch.ops.aten.sqrt.default(add_1240);  add_1240 = None
        reciprocal_429 = torch.ops.aten.reciprocal.default(sqrt_429);  sqrt_429 = None
        mul_1439 = torch.ops.aten.mul.Tensor(reciprocal_429, 1);  reciprocal_429 = None
        unsqueeze_3470 = torch.ops.aten.unsqueeze.default(arg522_1, -1);  arg522_1 = None
        unsqueeze_3471 = torch.ops.aten.unsqueeze.default(unsqueeze_3470, -1);  unsqueeze_3470 = None
        unsqueeze_3472 = torch.ops.aten.unsqueeze.default(mul_1439, -1);  mul_1439 = None
        unsqueeze_3473 = torch.ops.aten.unsqueeze.default(unsqueeze_3472, -1);  unsqueeze_3472 = None
        sub_429 = torch.ops.aten.sub.Tensor(convolution_429, unsqueeze_3471);  convolution_429 = unsqueeze_3471 = None
        mul_1440 = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_3473);  sub_429 = unsqueeze_3473 = None
        unsqueeze_3474 = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
        unsqueeze_3475 = torch.ops.aten.unsqueeze.default(unsqueeze_3474, -1);  unsqueeze_3474 = None
        mul_1441 = torch.ops.aten.mul.Tensor(mul_1440, unsqueeze_3475);  mul_1440 = unsqueeze_3475 = None
        unsqueeze_3476 = torch.ops.aten.unsqueeze.default(arg525_1, -1);  arg525_1 = None
        unsqueeze_3477 = torch.ops.aten.unsqueeze.default(unsqueeze_3476, -1);  unsqueeze_3476 = None
        add_1241 = torch.ops.aten.add.Tensor(mul_1441, unsqueeze_3477);  mul_1441 = unsqueeze_3477 = None
        relu_381 = torch.ops.aten.relu.default(add_1241);  add_1241 = None
        convolution_430 = torch.ops.aten.convolution.default(relu_381, arg526_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_381 = arg526_1 = None
        add_1242 = torch.ops.aten.add.Tensor(arg528_1, 1e-05);  arg528_1 = None
        sqrt_430 = torch.ops.aten.sqrt.default(add_1242);  add_1242 = None
        reciprocal_430 = torch.ops.aten.reciprocal.default(sqrt_430);  sqrt_430 = None
        mul_1442 = torch.ops.aten.mul.Tensor(reciprocal_430, 1);  reciprocal_430 = None
        unsqueeze_3478 = torch.ops.aten.unsqueeze.default(arg527_1, -1);  arg527_1 = None
        unsqueeze_3479 = torch.ops.aten.unsqueeze.default(unsqueeze_3478, -1);  unsqueeze_3478 = None
        unsqueeze_3480 = torch.ops.aten.unsqueeze.default(mul_1442, -1);  mul_1442 = None
        unsqueeze_3481 = torch.ops.aten.unsqueeze.default(unsqueeze_3480, -1);  unsqueeze_3480 = None
        sub_430 = torch.ops.aten.sub.Tensor(convolution_430, unsqueeze_3479);  convolution_430 = unsqueeze_3479 = None
        mul_1443 = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_3481);  sub_430 = unsqueeze_3481 = None
        unsqueeze_3482 = torch.ops.aten.unsqueeze.default(arg529_1, -1);  arg529_1 = None
        unsqueeze_3483 = torch.ops.aten.unsqueeze.default(unsqueeze_3482, -1);  unsqueeze_3482 = None
        mul_1444 = torch.ops.aten.mul.Tensor(mul_1443, unsqueeze_3483);  mul_1443 = unsqueeze_3483 = None
        unsqueeze_3484 = torch.ops.aten.unsqueeze.default(arg530_1, -1);  arg530_1 = None
        unsqueeze_3485 = torch.ops.aten.unsqueeze.default(unsqueeze_3484, -1);  unsqueeze_3484 = None
        add_1243 = torch.ops.aten.add.Tensor(mul_1444, unsqueeze_3485);  mul_1444 = unsqueeze_3485 = None
        add_1244 = torch.ops.aten.add.Tensor(add_1243, relu_380);  add_1243 = relu_380 = None
        relu_382 = torch.ops.aten.relu.default(add_1244);  add_1244 = None
        convolution_431 = torch.ops.aten.convolution.default(relu_372, arg531_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg531_1 = None
        add_1245 = torch.ops.aten.add.Tensor(arg533_1, 1e-05);  arg533_1 = None
        sqrt_431 = torch.ops.aten.sqrt.default(add_1245);  add_1245 = None
        reciprocal_431 = torch.ops.aten.reciprocal.default(sqrt_431);  sqrt_431 = None
        mul_1445 = torch.ops.aten.mul.Tensor(reciprocal_431, 1);  reciprocal_431 = None
        unsqueeze_3486 = torch.ops.aten.unsqueeze.default(arg532_1, -1);  arg532_1 = None
        unsqueeze_3487 = torch.ops.aten.unsqueeze.default(unsqueeze_3486, -1);  unsqueeze_3486 = None
        unsqueeze_3488 = torch.ops.aten.unsqueeze.default(mul_1445, -1);  mul_1445 = None
        unsqueeze_3489 = torch.ops.aten.unsqueeze.default(unsqueeze_3488, -1);  unsqueeze_3488 = None
        sub_431 = torch.ops.aten.sub.Tensor(convolution_431, unsqueeze_3487);  convolution_431 = unsqueeze_3487 = None
        mul_1446 = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_3489);  sub_431 = unsqueeze_3489 = None
        unsqueeze_3490 = torch.ops.aten.unsqueeze.default(arg534_1, -1);  arg534_1 = None
        unsqueeze_3491 = torch.ops.aten.unsqueeze.default(unsqueeze_3490, -1);  unsqueeze_3490 = None
        mul_1447 = torch.ops.aten.mul.Tensor(mul_1446, unsqueeze_3491);  mul_1446 = unsqueeze_3491 = None
        unsqueeze_3492 = torch.ops.aten.unsqueeze.default(arg535_1, -1);  arg535_1 = None
        unsqueeze_3493 = torch.ops.aten.unsqueeze.default(unsqueeze_3492, -1);  unsqueeze_3492 = None
        add_1246 = torch.ops.aten.add.Tensor(mul_1447, unsqueeze_3493);  mul_1447 = unsqueeze_3493 = None
        relu_383 = torch.ops.aten.relu.default(add_1246);  add_1246 = None
        convolution_432 = torch.ops.aten.convolution.default(relu_383, arg536_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_383 = arg536_1 = None
        add_1247 = torch.ops.aten.add.Tensor(arg538_1, 1e-05);  arg538_1 = None
        sqrt_432 = torch.ops.aten.sqrt.default(add_1247);  add_1247 = None
        reciprocal_432 = torch.ops.aten.reciprocal.default(sqrt_432);  sqrt_432 = None
        mul_1448 = torch.ops.aten.mul.Tensor(reciprocal_432, 1);  reciprocal_432 = None
        unsqueeze_3494 = torch.ops.aten.unsqueeze.default(arg537_1, -1);  arg537_1 = None
        unsqueeze_3495 = torch.ops.aten.unsqueeze.default(unsqueeze_3494, -1);  unsqueeze_3494 = None
        unsqueeze_3496 = torch.ops.aten.unsqueeze.default(mul_1448, -1);  mul_1448 = None
        unsqueeze_3497 = torch.ops.aten.unsqueeze.default(unsqueeze_3496, -1);  unsqueeze_3496 = None
        sub_432 = torch.ops.aten.sub.Tensor(convolution_432, unsqueeze_3495);  convolution_432 = unsqueeze_3495 = None
        mul_1449 = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_3497);  sub_432 = unsqueeze_3497 = None
        unsqueeze_3498 = torch.ops.aten.unsqueeze.default(arg539_1, -1);  arg539_1 = None
        unsqueeze_3499 = torch.ops.aten.unsqueeze.default(unsqueeze_3498, -1);  unsqueeze_3498 = None
        mul_1450 = torch.ops.aten.mul.Tensor(mul_1449, unsqueeze_3499);  mul_1449 = unsqueeze_3499 = None
        unsqueeze_3500 = torch.ops.aten.unsqueeze.default(arg540_1, -1);  arg540_1 = None
        unsqueeze_3501 = torch.ops.aten.unsqueeze.default(unsqueeze_3500, -1);  unsqueeze_3500 = None
        add_1248 = torch.ops.aten.add.Tensor(mul_1450, unsqueeze_3501);  mul_1450 = unsqueeze_3501 = None
        add_1249 = torch.ops.aten.add.Tensor(add_1248, relu_372);  add_1248 = relu_372 = None
        relu_384 = torch.ops.aten.relu.default(add_1249);  add_1249 = None
        convolution_433 = torch.ops.aten.convolution.default(relu_384, arg541_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg541_1 = None
        add_1250 = torch.ops.aten.add.Tensor(arg543_1, 1e-05);  arg543_1 = None
        sqrt_433 = torch.ops.aten.sqrt.default(add_1250);  add_1250 = None
        reciprocal_433 = torch.ops.aten.reciprocal.default(sqrt_433);  sqrt_433 = None
        mul_1451 = torch.ops.aten.mul.Tensor(reciprocal_433, 1);  reciprocal_433 = None
        unsqueeze_3502 = torch.ops.aten.unsqueeze.default(arg542_1, -1);  arg542_1 = None
        unsqueeze_3503 = torch.ops.aten.unsqueeze.default(unsqueeze_3502, -1);  unsqueeze_3502 = None
        unsqueeze_3504 = torch.ops.aten.unsqueeze.default(mul_1451, -1);  mul_1451 = None
        unsqueeze_3505 = torch.ops.aten.unsqueeze.default(unsqueeze_3504, -1);  unsqueeze_3504 = None
        sub_433 = torch.ops.aten.sub.Tensor(convolution_433, unsqueeze_3503);  convolution_433 = unsqueeze_3503 = None
        mul_1452 = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_3505);  sub_433 = unsqueeze_3505 = None
        unsqueeze_3506 = torch.ops.aten.unsqueeze.default(arg544_1, -1);  arg544_1 = None
        unsqueeze_3507 = torch.ops.aten.unsqueeze.default(unsqueeze_3506, -1);  unsqueeze_3506 = None
        mul_1453 = torch.ops.aten.mul.Tensor(mul_1452, unsqueeze_3507);  mul_1452 = unsqueeze_3507 = None
        unsqueeze_3508 = torch.ops.aten.unsqueeze.default(arg545_1, -1);  arg545_1 = None
        unsqueeze_3509 = torch.ops.aten.unsqueeze.default(unsqueeze_3508, -1);  unsqueeze_3508 = None
        add_1251 = torch.ops.aten.add.Tensor(mul_1453, unsqueeze_3509);  mul_1453 = unsqueeze_3509 = None
        relu_385 = torch.ops.aten.relu.default(add_1251);  add_1251 = None
        convolution_434 = torch.ops.aten.convolution.default(relu_385, arg546_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_385 = arg546_1 = None
        add_1252 = torch.ops.aten.add.Tensor(arg548_1, 1e-05);  arg548_1 = None
        sqrt_434 = torch.ops.aten.sqrt.default(add_1252);  add_1252 = None
        reciprocal_434 = torch.ops.aten.reciprocal.default(sqrt_434);  sqrt_434 = None
        mul_1454 = torch.ops.aten.mul.Tensor(reciprocal_434, 1);  reciprocal_434 = None
        unsqueeze_3510 = torch.ops.aten.unsqueeze.default(arg547_1, -1);  arg547_1 = None
        unsqueeze_3511 = torch.ops.aten.unsqueeze.default(unsqueeze_3510, -1);  unsqueeze_3510 = None
        unsqueeze_3512 = torch.ops.aten.unsqueeze.default(mul_1454, -1);  mul_1454 = None
        unsqueeze_3513 = torch.ops.aten.unsqueeze.default(unsqueeze_3512, -1);  unsqueeze_3512 = None
        sub_434 = torch.ops.aten.sub.Tensor(convolution_434, unsqueeze_3511);  convolution_434 = unsqueeze_3511 = None
        mul_1455 = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_3513);  sub_434 = unsqueeze_3513 = None
        unsqueeze_3514 = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_3515 = torch.ops.aten.unsqueeze.default(unsqueeze_3514, -1);  unsqueeze_3514 = None
        mul_1456 = torch.ops.aten.mul.Tensor(mul_1455, unsqueeze_3515);  mul_1455 = unsqueeze_3515 = None
        unsqueeze_3516 = torch.ops.aten.unsqueeze.default(arg550_1, -1);  arg550_1 = None
        unsqueeze_3517 = torch.ops.aten.unsqueeze.default(unsqueeze_3516, -1);  unsqueeze_3516 = None
        add_1253 = torch.ops.aten.add.Tensor(mul_1456, unsqueeze_3517);  mul_1456 = unsqueeze_3517 = None
        add_1254 = torch.ops.aten.add.Tensor(add_1253, relu_384);  add_1253 = relu_384 = None
        relu_386 = torch.ops.aten.relu.default(add_1254);  add_1254 = None
        convolution_435 = torch.ops.aten.convolution.default(relu_386, arg551_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg551_1 = None
        add_1255 = torch.ops.aten.add.Tensor(arg553_1, 1e-05);  arg553_1 = None
        sqrt_435 = torch.ops.aten.sqrt.default(add_1255);  add_1255 = None
        reciprocal_435 = torch.ops.aten.reciprocal.default(sqrt_435);  sqrt_435 = None
        mul_1457 = torch.ops.aten.mul.Tensor(reciprocal_435, 1);  reciprocal_435 = None
        unsqueeze_3518 = torch.ops.aten.unsqueeze.default(arg552_1, -1);  arg552_1 = None
        unsqueeze_3519 = torch.ops.aten.unsqueeze.default(unsqueeze_3518, -1);  unsqueeze_3518 = None
        unsqueeze_3520 = torch.ops.aten.unsqueeze.default(mul_1457, -1);  mul_1457 = None
        unsqueeze_3521 = torch.ops.aten.unsqueeze.default(unsqueeze_3520, -1);  unsqueeze_3520 = None
        sub_435 = torch.ops.aten.sub.Tensor(convolution_435, unsqueeze_3519);  convolution_435 = unsqueeze_3519 = None
        mul_1458 = torch.ops.aten.mul.Tensor(sub_435, unsqueeze_3521);  sub_435 = unsqueeze_3521 = None
        unsqueeze_3522 = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
        unsqueeze_3523 = torch.ops.aten.unsqueeze.default(unsqueeze_3522, -1);  unsqueeze_3522 = None
        mul_1459 = torch.ops.aten.mul.Tensor(mul_1458, unsqueeze_3523);  mul_1458 = unsqueeze_3523 = None
        unsqueeze_3524 = torch.ops.aten.unsqueeze.default(arg555_1, -1);  arg555_1 = None
        unsqueeze_3525 = torch.ops.aten.unsqueeze.default(unsqueeze_3524, -1);  unsqueeze_3524 = None
        add_1256 = torch.ops.aten.add.Tensor(mul_1459, unsqueeze_3525);  mul_1459 = unsqueeze_3525 = None
        relu_387 = torch.ops.aten.relu.default(add_1256);  add_1256 = None
        convolution_436 = torch.ops.aten.convolution.default(relu_387, arg556_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_387 = arg556_1 = None
        add_1257 = torch.ops.aten.add.Tensor(arg558_1, 1e-05);  arg558_1 = None
        sqrt_436 = torch.ops.aten.sqrt.default(add_1257);  add_1257 = None
        reciprocal_436 = torch.ops.aten.reciprocal.default(sqrt_436);  sqrt_436 = None
        mul_1460 = torch.ops.aten.mul.Tensor(reciprocal_436, 1);  reciprocal_436 = None
        unsqueeze_3526 = torch.ops.aten.unsqueeze.default(arg557_1, -1);  arg557_1 = None
        unsqueeze_3527 = torch.ops.aten.unsqueeze.default(unsqueeze_3526, -1);  unsqueeze_3526 = None
        unsqueeze_3528 = torch.ops.aten.unsqueeze.default(mul_1460, -1);  mul_1460 = None
        unsqueeze_3529 = torch.ops.aten.unsqueeze.default(unsqueeze_3528, -1);  unsqueeze_3528 = None
        sub_436 = torch.ops.aten.sub.Tensor(convolution_436, unsqueeze_3527);  convolution_436 = unsqueeze_3527 = None
        mul_1461 = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_3529);  sub_436 = unsqueeze_3529 = None
        unsqueeze_3530 = torch.ops.aten.unsqueeze.default(arg559_1, -1);  arg559_1 = None
        unsqueeze_3531 = torch.ops.aten.unsqueeze.default(unsqueeze_3530, -1);  unsqueeze_3530 = None
        mul_1462 = torch.ops.aten.mul.Tensor(mul_1461, unsqueeze_3531);  mul_1461 = unsqueeze_3531 = None
        unsqueeze_3532 = torch.ops.aten.unsqueeze.default(arg560_1, -1);  arg560_1 = None
        unsqueeze_3533 = torch.ops.aten.unsqueeze.default(unsqueeze_3532, -1);  unsqueeze_3532 = None
        add_1258 = torch.ops.aten.add.Tensor(mul_1462, unsqueeze_3533);  mul_1462 = unsqueeze_3533 = None
        add_1259 = torch.ops.aten.add.Tensor(add_1258, relu_386);  add_1258 = relu_386 = None
        relu_388 = torch.ops.aten.relu.default(add_1259);  add_1259 = None
        convolution_437 = torch.ops.aten.convolution.default(relu_388, arg561_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg561_1 = None
        add_1260 = torch.ops.aten.add.Tensor(arg563_1, 1e-05);  arg563_1 = None
        sqrt_437 = torch.ops.aten.sqrt.default(add_1260);  add_1260 = None
        reciprocal_437 = torch.ops.aten.reciprocal.default(sqrt_437);  sqrt_437 = None
        mul_1463 = torch.ops.aten.mul.Tensor(reciprocal_437, 1);  reciprocal_437 = None
        unsqueeze_3534 = torch.ops.aten.unsqueeze.default(arg562_1, -1);  arg562_1 = None
        unsqueeze_3535 = torch.ops.aten.unsqueeze.default(unsqueeze_3534, -1);  unsqueeze_3534 = None
        unsqueeze_3536 = torch.ops.aten.unsqueeze.default(mul_1463, -1);  mul_1463 = None
        unsqueeze_3537 = torch.ops.aten.unsqueeze.default(unsqueeze_3536, -1);  unsqueeze_3536 = None
        sub_437 = torch.ops.aten.sub.Tensor(convolution_437, unsqueeze_3535);  convolution_437 = unsqueeze_3535 = None
        mul_1464 = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_3537);  sub_437 = unsqueeze_3537 = None
        unsqueeze_3538 = torch.ops.aten.unsqueeze.default(arg564_1, -1);  arg564_1 = None
        unsqueeze_3539 = torch.ops.aten.unsqueeze.default(unsqueeze_3538, -1);  unsqueeze_3538 = None
        mul_1465 = torch.ops.aten.mul.Tensor(mul_1464, unsqueeze_3539);  mul_1464 = unsqueeze_3539 = None
        unsqueeze_3540 = torch.ops.aten.unsqueeze.default(arg565_1, -1);  arg565_1 = None
        unsqueeze_3541 = torch.ops.aten.unsqueeze.default(unsqueeze_3540, -1);  unsqueeze_3540 = None
        add_1261 = torch.ops.aten.add.Tensor(mul_1465, unsqueeze_3541);  mul_1465 = unsqueeze_3541 = None
        relu_389 = torch.ops.aten.relu.default(add_1261);  add_1261 = None
        convolution_438 = torch.ops.aten.convolution.default(relu_389, arg566_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_389 = arg566_1 = None
        add_1262 = torch.ops.aten.add.Tensor(arg568_1, 1e-05);  arg568_1 = None
        sqrt_438 = torch.ops.aten.sqrt.default(add_1262);  add_1262 = None
        reciprocal_438 = torch.ops.aten.reciprocal.default(sqrt_438);  sqrt_438 = None
        mul_1466 = torch.ops.aten.mul.Tensor(reciprocal_438, 1);  reciprocal_438 = None
        unsqueeze_3542 = torch.ops.aten.unsqueeze.default(arg567_1, -1);  arg567_1 = None
        unsqueeze_3543 = torch.ops.aten.unsqueeze.default(unsqueeze_3542, -1);  unsqueeze_3542 = None
        unsqueeze_3544 = torch.ops.aten.unsqueeze.default(mul_1466, -1);  mul_1466 = None
        unsqueeze_3545 = torch.ops.aten.unsqueeze.default(unsqueeze_3544, -1);  unsqueeze_3544 = None
        sub_438 = torch.ops.aten.sub.Tensor(convolution_438, unsqueeze_3543);  convolution_438 = unsqueeze_3543 = None
        mul_1467 = torch.ops.aten.mul.Tensor(sub_438, unsqueeze_3545);  sub_438 = unsqueeze_3545 = None
        unsqueeze_3546 = torch.ops.aten.unsqueeze.default(arg569_1, -1);  arg569_1 = None
        unsqueeze_3547 = torch.ops.aten.unsqueeze.default(unsqueeze_3546, -1);  unsqueeze_3546 = None
        mul_1468 = torch.ops.aten.mul.Tensor(mul_1467, unsqueeze_3547);  mul_1467 = unsqueeze_3547 = None
        unsqueeze_3548 = torch.ops.aten.unsqueeze.default(arg570_1, -1);  arg570_1 = None
        unsqueeze_3549 = torch.ops.aten.unsqueeze.default(unsqueeze_3548, -1);  unsqueeze_3548 = None
        add_1263 = torch.ops.aten.add.Tensor(mul_1468, unsqueeze_3549);  mul_1468 = unsqueeze_3549 = None
        add_1264 = torch.ops.aten.add.Tensor(add_1263, relu_388);  add_1263 = relu_388 = None
        relu_390 = torch.ops.aten.relu.default(add_1264);  add_1264 = None
        convolution_439 = torch.ops.aten.convolution.default(relu_374, arg571_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg571_1 = None
        add_1265 = torch.ops.aten.add.Tensor(arg573_1, 1e-05);  arg573_1 = None
        sqrt_439 = torch.ops.aten.sqrt.default(add_1265);  add_1265 = None
        reciprocal_439 = torch.ops.aten.reciprocal.default(sqrt_439);  sqrt_439 = None
        mul_1469 = torch.ops.aten.mul.Tensor(reciprocal_439, 1);  reciprocal_439 = None
        unsqueeze_3550 = torch.ops.aten.unsqueeze.default(arg572_1, -1);  arg572_1 = None
        unsqueeze_3551 = torch.ops.aten.unsqueeze.default(unsqueeze_3550, -1);  unsqueeze_3550 = None
        unsqueeze_3552 = torch.ops.aten.unsqueeze.default(mul_1469, -1);  mul_1469 = None
        unsqueeze_3553 = torch.ops.aten.unsqueeze.default(unsqueeze_3552, -1);  unsqueeze_3552 = None
        sub_439 = torch.ops.aten.sub.Tensor(convolution_439, unsqueeze_3551);  convolution_439 = unsqueeze_3551 = None
        mul_1470 = torch.ops.aten.mul.Tensor(sub_439, unsqueeze_3553);  sub_439 = unsqueeze_3553 = None
        unsqueeze_3554 = torch.ops.aten.unsqueeze.default(arg574_1, -1);  arg574_1 = None
        unsqueeze_3555 = torch.ops.aten.unsqueeze.default(unsqueeze_3554, -1);  unsqueeze_3554 = None
        mul_1471 = torch.ops.aten.mul.Tensor(mul_1470, unsqueeze_3555);  mul_1470 = unsqueeze_3555 = None
        unsqueeze_3556 = torch.ops.aten.unsqueeze.default(arg575_1, -1);  arg575_1 = None
        unsqueeze_3557 = torch.ops.aten.unsqueeze.default(unsqueeze_3556, -1);  unsqueeze_3556 = None
        add_1266 = torch.ops.aten.add.Tensor(mul_1471, unsqueeze_3557);  mul_1471 = unsqueeze_3557 = None
        relu_391 = torch.ops.aten.relu.default(add_1266);  add_1266 = None
        convolution_440 = torch.ops.aten.convolution.default(relu_391, arg576_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_391 = arg576_1 = None
        add_1267 = torch.ops.aten.add.Tensor(arg578_1, 1e-05);  arg578_1 = None
        sqrt_440 = torch.ops.aten.sqrt.default(add_1267);  add_1267 = None
        reciprocal_440 = torch.ops.aten.reciprocal.default(sqrt_440);  sqrt_440 = None
        mul_1472 = torch.ops.aten.mul.Tensor(reciprocal_440, 1);  reciprocal_440 = None
        unsqueeze_3558 = torch.ops.aten.unsqueeze.default(arg577_1, -1);  arg577_1 = None
        unsqueeze_3559 = torch.ops.aten.unsqueeze.default(unsqueeze_3558, -1);  unsqueeze_3558 = None
        unsqueeze_3560 = torch.ops.aten.unsqueeze.default(mul_1472, -1);  mul_1472 = None
        unsqueeze_3561 = torch.ops.aten.unsqueeze.default(unsqueeze_3560, -1);  unsqueeze_3560 = None
        sub_440 = torch.ops.aten.sub.Tensor(convolution_440, unsqueeze_3559);  convolution_440 = unsqueeze_3559 = None
        mul_1473 = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_3561);  sub_440 = unsqueeze_3561 = None
        unsqueeze_3562 = torch.ops.aten.unsqueeze.default(arg579_1, -1);  arg579_1 = None
        unsqueeze_3563 = torch.ops.aten.unsqueeze.default(unsqueeze_3562, -1);  unsqueeze_3562 = None
        mul_1474 = torch.ops.aten.mul.Tensor(mul_1473, unsqueeze_3563);  mul_1473 = unsqueeze_3563 = None
        unsqueeze_3564 = torch.ops.aten.unsqueeze.default(arg580_1, -1);  arg580_1 = None
        unsqueeze_3565 = torch.ops.aten.unsqueeze.default(unsqueeze_3564, -1);  unsqueeze_3564 = None
        add_1268 = torch.ops.aten.add.Tensor(mul_1474, unsqueeze_3565);  mul_1474 = unsqueeze_3565 = None
        add_1269 = torch.ops.aten.add.Tensor(add_1268, relu_374);  add_1268 = relu_374 = None
        relu_392 = torch.ops.aten.relu.default(add_1269);  add_1269 = None
        convolution_441 = torch.ops.aten.convolution.default(relu_392, arg581_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg581_1 = None
        add_1270 = torch.ops.aten.add.Tensor(arg583_1, 1e-05);  arg583_1 = None
        sqrt_441 = torch.ops.aten.sqrt.default(add_1270);  add_1270 = None
        reciprocal_441 = torch.ops.aten.reciprocal.default(sqrt_441);  sqrt_441 = None
        mul_1475 = torch.ops.aten.mul.Tensor(reciprocal_441, 1);  reciprocal_441 = None
        unsqueeze_3566 = torch.ops.aten.unsqueeze.default(arg582_1, -1);  arg582_1 = None
        unsqueeze_3567 = torch.ops.aten.unsqueeze.default(unsqueeze_3566, -1);  unsqueeze_3566 = None
        unsqueeze_3568 = torch.ops.aten.unsqueeze.default(mul_1475, -1);  mul_1475 = None
        unsqueeze_3569 = torch.ops.aten.unsqueeze.default(unsqueeze_3568, -1);  unsqueeze_3568 = None
        sub_441 = torch.ops.aten.sub.Tensor(convolution_441, unsqueeze_3567);  convolution_441 = unsqueeze_3567 = None
        mul_1476 = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_3569);  sub_441 = unsqueeze_3569 = None
        unsqueeze_3570 = torch.ops.aten.unsqueeze.default(arg584_1, -1);  arg584_1 = None
        unsqueeze_3571 = torch.ops.aten.unsqueeze.default(unsqueeze_3570, -1);  unsqueeze_3570 = None
        mul_1477 = torch.ops.aten.mul.Tensor(mul_1476, unsqueeze_3571);  mul_1476 = unsqueeze_3571 = None
        unsqueeze_3572 = torch.ops.aten.unsqueeze.default(arg585_1, -1);  arg585_1 = None
        unsqueeze_3573 = torch.ops.aten.unsqueeze.default(unsqueeze_3572, -1);  unsqueeze_3572 = None
        add_1271 = torch.ops.aten.add.Tensor(mul_1477, unsqueeze_3573);  mul_1477 = unsqueeze_3573 = None
        relu_393 = torch.ops.aten.relu.default(add_1271);  add_1271 = None
        convolution_442 = torch.ops.aten.convolution.default(relu_393, arg586_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_393 = arg586_1 = None
        add_1272 = torch.ops.aten.add.Tensor(arg588_1, 1e-05);  arg588_1 = None
        sqrt_442 = torch.ops.aten.sqrt.default(add_1272);  add_1272 = None
        reciprocal_442 = torch.ops.aten.reciprocal.default(sqrt_442);  sqrt_442 = None
        mul_1478 = torch.ops.aten.mul.Tensor(reciprocal_442, 1);  reciprocal_442 = None
        unsqueeze_3574 = torch.ops.aten.unsqueeze.default(arg587_1, -1);  arg587_1 = None
        unsqueeze_3575 = torch.ops.aten.unsqueeze.default(unsqueeze_3574, -1);  unsqueeze_3574 = None
        unsqueeze_3576 = torch.ops.aten.unsqueeze.default(mul_1478, -1);  mul_1478 = None
        unsqueeze_3577 = torch.ops.aten.unsqueeze.default(unsqueeze_3576, -1);  unsqueeze_3576 = None
        sub_442 = torch.ops.aten.sub.Tensor(convolution_442, unsqueeze_3575);  convolution_442 = unsqueeze_3575 = None
        mul_1479 = torch.ops.aten.mul.Tensor(sub_442, unsqueeze_3577);  sub_442 = unsqueeze_3577 = None
        unsqueeze_3578 = torch.ops.aten.unsqueeze.default(arg589_1, -1);  arg589_1 = None
        unsqueeze_3579 = torch.ops.aten.unsqueeze.default(unsqueeze_3578, -1);  unsqueeze_3578 = None
        mul_1480 = torch.ops.aten.mul.Tensor(mul_1479, unsqueeze_3579);  mul_1479 = unsqueeze_3579 = None
        unsqueeze_3580 = torch.ops.aten.unsqueeze.default(arg590_1, -1);  arg590_1 = None
        unsqueeze_3581 = torch.ops.aten.unsqueeze.default(unsqueeze_3580, -1);  unsqueeze_3580 = None
        add_1273 = torch.ops.aten.add.Tensor(mul_1480, unsqueeze_3581);  mul_1480 = unsqueeze_3581 = None
        add_1274 = torch.ops.aten.add.Tensor(add_1273, relu_392);  add_1273 = relu_392 = None
        relu_394 = torch.ops.aten.relu.default(add_1274);  add_1274 = None
        convolution_443 = torch.ops.aten.convolution.default(relu_394, arg591_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg591_1 = None
        add_1275 = torch.ops.aten.add.Tensor(arg593_1, 1e-05);  arg593_1 = None
        sqrt_443 = torch.ops.aten.sqrt.default(add_1275);  add_1275 = None
        reciprocal_443 = torch.ops.aten.reciprocal.default(sqrt_443);  sqrt_443 = None
        mul_1481 = torch.ops.aten.mul.Tensor(reciprocal_443, 1);  reciprocal_443 = None
        unsqueeze_3582 = torch.ops.aten.unsqueeze.default(arg592_1, -1);  arg592_1 = None
        unsqueeze_3583 = torch.ops.aten.unsqueeze.default(unsqueeze_3582, -1);  unsqueeze_3582 = None
        unsqueeze_3584 = torch.ops.aten.unsqueeze.default(mul_1481, -1);  mul_1481 = None
        unsqueeze_3585 = torch.ops.aten.unsqueeze.default(unsqueeze_3584, -1);  unsqueeze_3584 = None
        sub_443 = torch.ops.aten.sub.Tensor(convolution_443, unsqueeze_3583);  convolution_443 = unsqueeze_3583 = None
        mul_1482 = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_3585);  sub_443 = unsqueeze_3585 = None
        unsqueeze_3586 = torch.ops.aten.unsqueeze.default(arg594_1, -1);  arg594_1 = None
        unsqueeze_3587 = torch.ops.aten.unsqueeze.default(unsqueeze_3586, -1);  unsqueeze_3586 = None
        mul_1483 = torch.ops.aten.mul.Tensor(mul_1482, unsqueeze_3587);  mul_1482 = unsqueeze_3587 = None
        unsqueeze_3588 = torch.ops.aten.unsqueeze.default(arg595_1, -1);  arg595_1 = None
        unsqueeze_3589 = torch.ops.aten.unsqueeze.default(unsqueeze_3588, -1);  unsqueeze_3588 = None
        add_1276 = torch.ops.aten.add.Tensor(mul_1483, unsqueeze_3589);  mul_1483 = unsqueeze_3589 = None
        relu_395 = torch.ops.aten.relu.default(add_1276);  add_1276 = None
        convolution_444 = torch.ops.aten.convolution.default(relu_395, arg596_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_395 = arg596_1 = None
        add_1277 = torch.ops.aten.add.Tensor(arg598_1, 1e-05);  arg598_1 = None
        sqrt_444 = torch.ops.aten.sqrt.default(add_1277);  add_1277 = None
        reciprocal_444 = torch.ops.aten.reciprocal.default(sqrt_444);  sqrt_444 = None
        mul_1484 = torch.ops.aten.mul.Tensor(reciprocal_444, 1);  reciprocal_444 = None
        unsqueeze_3590 = torch.ops.aten.unsqueeze.default(arg597_1, -1);  arg597_1 = None
        unsqueeze_3591 = torch.ops.aten.unsqueeze.default(unsqueeze_3590, -1);  unsqueeze_3590 = None
        unsqueeze_3592 = torch.ops.aten.unsqueeze.default(mul_1484, -1);  mul_1484 = None
        unsqueeze_3593 = torch.ops.aten.unsqueeze.default(unsqueeze_3592, -1);  unsqueeze_3592 = None
        sub_444 = torch.ops.aten.sub.Tensor(convolution_444, unsqueeze_3591);  convolution_444 = unsqueeze_3591 = None
        mul_1485 = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_3593);  sub_444 = unsqueeze_3593 = None
        unsqueeze_3594 = torch.ops.aten.unsqueeze.default(arg599_1, -1);  arg599_1 = None
        unsqueeze_3595 = torch.ops.aten.unsqueeze.default(unsqueeze_3594, -1);  unsqueeze_3594 = None
        mul_1486 = torch.ops.aten.mul.Tensor(mul_1485, unsqueeze_3595);  mul_1485 = unsqueeze_3595 = None
        unsqueeze_3596 = torch.ops.aten.unsqueeze.default(arg600_1, -1);  arg600_1 = None
        unsqueeze_3597 = torch.ops.aten.unsqueeze.default(unsqueeze_3596, -1);  unsqueeze_3596 = None
        add_1278 = torch.ops.aten.add.Tensor(mul_1486, unsqueeze_3597);  mul_1486 = unsqueeze_3597 = None
        add_1279 = torch.ops.aten.add.Tensor(add_1278, relu_394);  add_1278 = relu_394 = None
        relu_396 = torch.ops.aten.relu.default(add_1279);  add_1279 = None
        convolution_445 = torch.ops.aten.convolution.default(relu_396, arg601_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg601_1 = None
        add_1280 = torch.ops.aten.add.Tensor(arg603_1, 1e-05);  arg603_1 = None
        sqrt_445 = torch.ops.aten.sqrt.default(add_1280);  add_1280 = None
        reciprocal_445 = torch.ops.aten.reciprocal.default(sqrt_445);  sqrt_445 = None
        mul_1487 = torch.ops.aten.mul.Tensor(reciprocal_445, 1);  reciprocal_445 = None
        unsqueeze_3598 = torch.ops.aten.unsqueeze.default(arg602_1, -1);  arg602_1 = None
        unsqueeze_3599 = torch.ops.aten.unsqueeze.default(unsqueeze_3598, -1);  unsqueeze_3598 = None
        unsqueeze_3600 = torch.ops.aten.unsqueeze.default(mul_1487, -1);  mul_1487 = None
        unsqueeze_3601 = torch.ops.aten.unsqueeze.default(unsqueeze_3600, -1);  unsqueeze_3600 = None
        sub_445 = torch.ops.aten.sub.Tensor(convolution_445, unsqueeze_3599);  convolution_445 = unsqueeze_3599 = None
        mul_1488 = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_3601);  sub_445 = unsqueeze_3601 = None
        unsqueeze_3602 = torch.ops.aten.unsqueeze.default(arg604_1, -1);  arg604_1 = None
        unsqueeze_3603 = torch.ops.aten.unsqueeze.default(unsqueeze_3602, -1);  unsqueeze_3602 = None
        mul_1489 = torch.ops.aten.mul.Tensor(mul_1488, unsqueeze_3603);  mul_1488 = unsqueeze_3603 = None
        unsqueeze_3604 = torch.ops.aten.unsqueeze.default(arg605_1, -1);  arg605_1 = None
        unsqueeze_3605 = torch.ops.aten.unsqueeze.default(unsqueeze_3604, -1);  unsqueeze_3604 = None
        add_1281 = torch.ops.aten.add.Tensor(mul_1489, unsqueeze_3605);  mul_1489 = unsqueeze_3605 = None
        relu_397 = torch.ops.aten.relu.default(add_1281);  add_1281 = None
        convolution_446 = torch.ops.aten.convolution.default(relu_397, arg606_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_397 = arg606_1 = None
        add_1282 = torch.ops.aten.add.Tensor(arg608_1, 1e-05);  arg608_1 = None
        sqrt_446 = torch.ops.aten.sqrt.default(add_1282);  add_1282 = None
        reciprocal_446 = torch.ops.aten.reciprocal.default(sqrt_446);  sqrt_446 = None
        mul_1490 = torch.ops.aten.mul.Tensor(reciprocal_446, 1);  reciprocal_446 = None
        unsqueeze_3606 = torch.ops.aten.unsqueeze.default(arg607_1, -1);  arg607_1 = None
        unsqueeze_3607 = torch.ops.aten.unsqueeze.default(unsqueeze_3606, -1);  unsqueeze_3606 = None
        unsqueeze_3608 = torch.ops.aten.unsqueeze.default(mul_1490, -1);  mul_1490 = None
        unsqueeze_3609 = torch.ops.aten.unsqueeze.default(unsqueeze_3608, -1);  unsqueeze_3608 = None
        sub_446 = torch.ops.aten.sub.Tensor(convolution_446, unsqueeze_3607);  convolution_446 = unsqueeze_3607 = None
        mul_1491 = torch.ops.aten.mul.Tensor(sub_446, unsqueeze_3609);  sub_446 = unsqueeze_3609 = None
        unsqueeze_3610 = torch.ops.aten.unsqueeze.default(arg609_1, -1);  arg609_1 = None
        unsqueeze_3611 = torch.ops.aten.unsqueeze.default(unsqueeze_3610, -1);  unsqueeze_3610 = None
        mul_1492 = torch.ops.aten.mul.Tensor(mul_1491, unsqueeze_3611);  mul_1491 = unsqueeze_3611 = None
        unsqueeze_3612 = torch.ops.aten.unsqueeze.default(arg610_1, -1);  arg610_1 = None
        unsqueeze_3613 = torch.ops.aten.unsqueeze.default(unsqueeze_3612, -1);  unsqueeze_3612 = None
        add_1283 = torch.ops.aten.add.Tensor(mul_1492, unsqueeze_3613);  mul_1492 = unsqueeze_3613 = None
        add_1284 = torch.ops.aten.add.Tensor(add_1283, relu_396);  add_1283 = relu_396 = None
        relu_398 = torch.ops.aten.relu.default(add_1284);  add_1284 = None
        convolution_447 = torch.ops.aten.convolution.default(relu_390, arg611_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg611_1 = None
        add_1285 = torch.ops.aten.add.Tensor(arg613_1, 1e-05);  arg613_1 = None
        sqrt_447 = torch.ops.aten.sqrt.default(add_1285);  add_1285 = None
        reciprocal_447 = torch.ops.aten.reciprocal.default(sqrt_447);  sqrt_447 = None
        mul_1493 = torch.ops.aten.mul.Tensor(reciprocal_447, 1);  reciprocal_447 = None
        unsqueeze_3614 = torch.ops.aten.unsqueeze.default(arg612_1, -1);  arg612_1 = None
        unsqueeze_3615 = torch.ops.aten.unsqueeze.default(unsqueeze_3614, -1);  unsqueeze_3614 = None
        unsqueeze_3616 = torch.ops.aten.unsqueeze.default(mul_1493, -1);  mul_1493 = None
        unsqueeze_3617 = torch.ops.aten.unsqueeze.default(unsqueeze_3616, -1);  unsqueeze_3616 = None
        sub_447 = torch.ops.aten.sub.Tensor(convolution_447, unsqueeze_3615);  convolution_447 = unsqueeze_3615 = None
        mul_1494 = torch.ops.aten.mul.Tensor(sub_447, unsqueeze_3617);  sub_447 = unsqueeze_3617 = None
        unsqueeze_3618 = torch.ops.aten.unsqueeze.default(arg614_1, -1);  arg614_1 = None
        unsqueeze_3619 = torch.ops.aten.unsqueeze.default(unsqueeze_3618, -1);  unsqueeze_3618 = None
        mul_1495 = torch.ops.aten.mul.Tensor(mul_1494, unsqueeze_3619);  mul_1494 = unsqueeze_3619 = None
        unsqueeze_3620 = torch.ops.aten.unsqueeze.default(arg615_1, -1);  arg615_1 = None
        unsqueeze_3621 = torch.ops.aten.unsqueeze.default(unsqueeze_3620, -1);  unsqueeze_3620 = None
        add_1286 = torch.ops.aten.add.Tensor(mul_1495, unsqueeze_3621);  mul_1495 = unsqueeze_3621 = None
        iota_76 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1496 = torch.ops.aten.mul.Tensor(iota_76, 1);  iota_76 = None
        add_1287 = torch.ops.aten.add.Tensor(mul_1496, 0);  mul_1496 = None
        convert_element_type_1048 = torch.ops.prims.convert_element_type.default(add_1287, torch.float32);  add_1287 = None
        add_1288 = torch.ops.aten.add.Tensor(convert_element_type_1048, 0.0);  convert_element_type_1048 = None
        mul_1497 = torch.ops.aten.mul.Tensor(add_1288, 0.5);  add_1288 = None
        convert_element_type_1049 = torch.ops.prims.convert_element_type.default(mul_1497, torch.int64);  mul_1497 = None
        unsqueeze_3622 = torch.ops.aten.unsqueeze.default(convert_element_type_1049, -1);  convert_element_type_1049 = None
        iota_77 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1498 = torch.ops.aten.mul.Tensor(iota_77, 1);  iota_77 = None
        add_1289 = torch.ops.aten.add.Tensor(mul_1498, 0);  mul_1498 = None
        convert_element_type_1050 = torch.ops.prims.convert_element_type.default(add_1289, torch.float32);  add_1289 = None
        add_1290 = torch.ops.aten.add.Tensor(convert_element_type_1050, 0.0);  convert_element_type_1050 = None
        mul_1499 = torch.ops.aten.mul.Tensor(add_1290, 0.5);  add_1290 = None
        convert_element_type_1051 = torch.ops.prims.convert_element_type.default(mul_1499, torch.int64);  mul_1499 = None
        _unsafe_index_38 = torch.ops.aten._unsafe_index.Tensor(add_1286, [None, None, unsqueeze_3622, convert_element_type_1051]);  add_1286 = unsqueeze_3622 = convert_element_type_1051 = None
        add_1291 = torch.ops.aten.add.Tensor(relu_382, _unsafe_index_38);  _unsafe_index_38 = None
        convolution_448 = torch.ops.aten.convolution.default(relu_398, arg616_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg616_1 = None
        add_1292 = torch.ops.aten.add.Tensor(arg618_1, 1e-05);  arg618_1 = None
        sqrt_448 = torch.ops.aten.sqrt.default(add_1292);  add_1292 = None
        reciprocal_448 = torch.ops.aten.reciprocal.default(sqrt_448);  sqrt_448 = None
        mul_1500 = torch.ops.aten.mul.Tensor(reciprocal_448, 1);  reciprocal_448 = None
        unsqueeze_3623 = torch.ops.aten.unsqueeze.default(arg617_1, -1);  arg617_1 = None
        unsqueeze_3624 = torch.ops.aten.unsqueeze.default(unsqueeze_3623, -1);  unsqueeze_3623 = None
        unsqueeze_3625 = torch.ops.aten.unsqueeze.default(mul_1500, -1);  mul_1500 = None
        unsqueeze_3626 = torch.ops.aten.unsqueeze.default(unsqueeze_3625, -1);  unsqueeze_3625 = None
        sub_448 = torch.ops.aten.sub.Tensor(convolution_448, unsqueeze_3624);  convolution_448 = unsqueeze_3624 = None
        mul_1501 = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_3626);  sub_448 = unsqueeze_3626 = None
        unsqueeze_3627 = torch.ops.aten.unsqueeze.default(arg619_1, -1);  arg619_1 = None
        unsqueeze_3628 = torch.ops.aten.unsqueeze.default(unsqueeze_3627, -1);  unsqueeze_3627 = None
        mul_1502 = torch.ops.aten.mul.Tensor(mul_1501, unsqueeze_3628);  mul_1501 = unsqueeze_3628 = None
        unsqueeze_3629 = torch.ops.aten.unsqueeze.default(arg620_1, -1);  arg620_1 = None
        unsqueeze_3630 = torch.ops.aten.unsqueeze.default(unsqueeze_3629, -1);  unsqueeze_3629 = None
        add_1293 = torch.ops.aten.add.Tensor(mul_1502, unsqueeze_3630);  mul_1502 = unsqueeze_3630 = None
        iota_78 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1503 = torch.ops.aten.mul.Tensor(iota_78, 1);  iota_78 = None
        add_1294 = torch.ops.aten.add.Tensor(mul_1503, 0);  mul_1503 = None
        convert_element_type_1054 = torch.ops.prims.convert_element_type.default(add_1294, torch.float32);  add_1294 = None
        add_1295 = torch.ops.aten.add.Tensor(convert_element_type_1054, 0.0);  convert_element_type_1054 = None
        mul_1504 = torch.ops.aten.mul.Tensor(add_1295, 0.25);  add_1295 = None
        convert_element_type_1055 = torch.ops.prims.convert_element_type.default(mul_1504, torch.int64);  mul_1504 = None
        unsqueeze_3631 = torch.ops.aten.unsqueeze.default(convert_element_type_1055, -1);  convert_element_type_1055 = None
        iota_79 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1505 = torch.ops.aten.mul.Tensor(iota_79, 1);  iota_79 = None
        add_1296 = torch.ops.aten.add.Tensor(mul_1505, 0);  mul_1505 = None
        convert_element_type_1056 = torch.ops.prims.convert_element_type.default(add_1296, torch.float32);  add_1296 = None
        add_1297 = torch.ops.aten.add.Tensor(convert_element_type_1056, 0.0);  convert_element_type_1056 = None
        mul_1506 = torch.ops.aten.mul.Tensor(add_1297, 0.25);  add_1297 = None
        convert_element_type_1057 = torch.ops.prims.convert_element_type.default(mul_1506, torch.int64);  mul_1506 = None
        _unsafe_index_39 = torch.ops.aten._unsafe_index.Tensor(add_1293, [None, None, unsqueeze_3631, convert_element_type_1057]);  add_1293 = unsqueeze_3631 = convert_element_type_1057 = None
        add_1298 = torch.ops.aten.add.Tensor(add_1291, _unsafe_index_39);  add_1291 = _unsafe_index_39 = None
        relu_399 = torch.ops.aten.relu.default(add_1298);  add_1298 = None
        convolution_449 = torch.ops.aten.convolution.default(relu_382, arg621_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg621_1 = None
        add_1299 = torch.ops.aten.add.Tensor(arg623_1, 1e-05);  arg623_1 = None
        sqrt_449 = torch.ops.aten.sqrt.default(add_1299);  add_1299 = None
        reciprocal_449 = torch.ops.aten.reciprocal.default(sqrt_449);  sqrt_449 = None
        mul_1507 = torch.ops.aten.mul.Tensor(reciprocal_449, 1);  reciprocal_449 = None
        unsqueeze_3632 = torch.ops.aten.unsqueeze.default(arg622_1, -1);  arg622_1 = None
        unsqueeze_3633 = torch.ops.aten.unsqueeze.default(unsqueeze_3632, -1);  unsqueeze_3632 = None
        unsqueeze_3634 = torch.ops.aten.unsqueeze.default(mul_1507, -1);  mul_1507 = None
        unsqueeze_3635 = torch.ops.aten.unsqueeze.default(unsqueeze_3634, -1);  unsqueeze_3634 = None
        sub_449 = torch.ops.aten.sub.Tensor(convolution_449, unsqueeze_3633);  convolution_449 = unsqueeze_3633 = None
        mul_1508 = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_3635);  sub_449 = unsqueeze_3635 = None
        unsqueeze_3636 = torch.ops.aten.unsqueeze.default(arg624_1, -1);  arg624_1 = None
        unsqueeze_3637 = torch.ops.aten.unsqueeze.default(unsqueeze_3636, -1);  unsqueeze_3636 = None
        mul_1509 = torch.ops.aten.mul.Tensor(mul_1508, unsqueeze_3637);  mul_1508 = unsqueeze_3637 = None
        unsqueeze_3638 = torch.ops.aten.unsqueeze.default(arg625_1, -1);  arg625_1 = None
        unsqueeze_3639 = torch.ops.aten.unsqueeze.default(unsqueeze_3638, -1);  unsqueeze_3638 = None
        add_1300 = torch.ops.aten.add.Tensor(mul_1509, unsqueeze_3639);  mul_1509 = unsqueeze_3639 = None
        add_1301 = torch.ops.aten.add.Tensor(add_1300, relu_390);  add_1300 = None
        convolution_450 = torch.ops.aten.convolution.default(relu_398, arg626_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg626_1 = None
        add_1302 = torch.ops.aten.add.Tensor(arg628_1, 1e-05);  arg628_1 = None
        sqrt_450 = torch.ops.aten.sqrt.default(add_1302);  add_1302 = None
        reciprocal_450 = torch.ops.aten.reciprocal.default(sqrt_450);  sqrt_450 = None
        mul_1510 = torch.ops.aten.mul.Tensor(reciprocal_450, 1);  reciprocal_450 = None
        unsqueeze_3640 = torch.ops.aten.unsqueeze.default(arg627_1, -1);  arg627_1 = None
        unsqueeze_3641 = torch.ops.aten.unsqueeze.default(unsqueeze_3640, -1);  unsqueeze_3640 = None
        unsqueeze_3642 = torch.ops.aten.unsqueeze.default(mul_1510, -1);  mul_1510 = None
        unsqueeze_3643 = torch.ops.aten.unsqueeze.default(unsqueeze_3642, -1);  unsqueeze_3642 = None
        sub_450 = torch.ops.aten.sub.Tensor(convolution_450, unsqueeze_3641);  convolution_450 = unsqueeze_3641 = None
        mul_1511 = torch.ops.aten.mul.Tensor(sub_450, unsqueeze_3643);  sub_450 = unsqueeze_3643 = None
        unsqueeze_3644 = torch.ops.aten.unsqueeze.default(arg629_1, -1);  arg629_1 = None
        unsqueeze_3645 = torch.ops.aten.unsqueeze.default(unsqueeze_3644, -1);  unsqueeze_3644 = None
        mul_1512 = torch.ops.aten.mul.Tensor(mul_1511, unsqueeze_3645);  mul_1511 = unsqueeze_3645 = None
        unsqueeze_3646 = torch.ops.aten.unsqueeze.default(arg630_1, -1);  arg630_1 = None
        unsqueeze_3647 = torch.ops.aten.unsqueeze.default(unsqueeze_3646, -1);  unsqueeze_3646 = None
        add_1303 = torch.ops.aten.add.Tensor(mul_1512, unsqueeze_3647);  mul_1512 = unsqueeze_3647 = None
        iota_80 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1513 = torch.ops.aten.mul.Tensor(iota_80, 1);  iota_80 = None
        add_1304 = torch.ops.aten.add.Tensor(mul_1513, 0);  mul_1513 = None
        convert_element_type_1062 = torch.ops.prims.convert_element_type.default(add_1304, torch.float32);  add_1304 = None
        add_1305 = torch.ops.aten.add.Tensor(convert_element_type_1062, 0.0);  convert_element_type_1062 = None
        mul_1514 = torch.ops.aten.mul.Tensor(add_1305, 0.5);  add_1305 = None
        convert_element_type_1063 = torch.ops.prims.convert_element_type.default(mul_1514, torch.int64);  mul_1514 = None
        unsqueeze_3648 = torch.ops.aten.unsqueeze.default(convert_element_type_1063, -1);  convert_element_type_1063 = None
        iota_81 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1515 = torch.ops.aten.mul.Tensor(iota_81, 1);  iota_81 = None
        add_1306 = torch.ops.aten.add.Tensor(mul_1515, 0);  mul_1515 = None
        convert_element_type_1064 = torch.ops.prims.convert_element_type.default(add_1306, torch.float32);  add_1306 = None
        add_1307 = torch.ops.aten.add.Tensor(convert_element_type_1064, 0.0);  convert_element_type_1064 = None
        mul_1516 = torch.ops.aten.mul.Tensor(add_1307, 0.5);  add_1307 = None
        convert_element_type_1065 = torch.ops.prims.convert_element_type.default(mul_1516, torch.int64);  mul_1516 = None
        _unsafe_index_40 = torch.ops.aten._unsafe_index.Tensor(add_1303, [None, None, unsqueeze_3648, convert_element_type_1065]);  add_1303 = unsqueeze_3648 = convert_element_type_1065 = None
        add_1308 = torch.ops.aten.add.Tensor(add_1301, _unsafe_index_40);  add_1301 = _unsafe_index_40 = None
        relu_400 = torch.ops.aten.relu.default(add_1308);  add_1308 = None
        convolution_451 = torch.ops.aten.convolution.default(relu_382, arg631_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_382 = arg631_1 = None
        add_1309 = torch.ops.aten.add.Tensor(arg633_1, 1e-05);  arg633_1 = None
        sqrt_451 = torch.ops.aten.sqrt.default(add_1309);  add_1309 = None
        reciprocal_451 = torch.ops.aten.reciprocal.default(sqrt_451);  sqrt_451 = None
        mul_1517 = torch.ops.aten.mul.Tensor(reciprocal_451, 1);  reciprocal_451 = None
        unsqueeze_3649 = torch.ops.aten.unsqueeze.default(arg632_1, -1);  arg632_1 = None
        unsqueeze_3650 = torch.ops.aten.unsqueeze.default(unsqueeze_3649, -1);  unsqueeze_3649 = None
        unsqueeze_3651 = torch.ops.aten.unsqueeze.default(mul_1517, -1);  mul_1517 = None
        unsqueeze_3652 = torch.ops.aten.unsqueeze.default(unsqueeze_3651, -1);  unsqueeze_3651 = None
        sub_451 = torch.ops.aten.sub.Tensor(convolution_451, unsqueeze_3650);  convolution_451 = unsqueeze_3650 = None
        mul_1518 = torch.ops.aten.mul.Tensor(sub_451, unsqueeze_3652);  sub_451 = unsqueeze_3652 = None
        unsqueeze_3653 = torch.ops.aten.unsqueeze.default(arg634_1, -1);  arg634_1 = None
        unsqueeze_3654 = torch.ops.aten.unsqueeze.default(unsqueeze_3653, -1);  unsqueeze_3653 = None
        mul_1519 = torch.ops.aten.mul.Tensor(mul_1518, unsqueeze_3654);  mul_1518 = unsqueeze_3654 = None
        unsqueeze_3655 = torch.ops.aten.unsqueeze.default(arg635_1, -1);  arg635_1 = None
        unsqueeze_3656 = torch.ops.aten.unsqueeze.default(unsqueeze_3655, -1);  unsqueeze_3655 = None
        add_1310 = torch.ops.aten.add.Tensor(mul_1519, unsqueeze_3656);  mul_1519 = unsqueeze_3656 = None
        relu_401 = torch.ops.aten.relu.default(add_1310);  add_1310 = None
        convolution_452 = torch.ops.aten.convolution.default(relu_401, arg636_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_401 = arg636_1 = None
        add_1311 = torch.ops.aten.add.Tensor(arg638_1, 1e-05);  arg638_1 = None
        sqrt_452 = torch.ops.aten.sqrt.default(add_1311);  add_1311 = None
        reciprocal_452 = torch.ops.aten.reciprocal.default(sqrt_452);  sqrt_452 = None
        mul_1520 = torch.ops.aten.mul.Tensor(reciprocal_452, 1);  reciprocal_452 = None
        unsqueeze_3657 = torch.ops.aten.unsqueeze.default(arg637_1, -1);  arg637_1 = None
        unsqueeze_3658 = torch.ops.aten.unsqueeze.default(unsqueeze_3657, -1);  unsqueeze_3657 = None
        unsqueeze_3659 = torch.ops.aten.unsqueeze.default(mul_1520, -1);  mul_1520 = None
        unsqueeze_3660 = torch.ops.aten.unsqueeze.default(unsqueeze_3659, -1);  unsqueeze_3659 = None
        sub_452 = torch.ops.aten.sub.Tensor(convolution_452, unsqueeze_3658);  convolution_452 = unsqueeze_3658 = None
        mul_1521 = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_3660);  sub_452 = unsqueeze_3660 = None
        unsqueeze_3661 = torch.ops.aten.unsqueeze.default(arg639_1, -1);  arg639_1 = None
        unsqueeze_3662 = torch.ops.aten.unsqueeze.default(unsqueeze_3661, -1);  unsqueeze_3661 = None
        mul_1522 = torch.ops.aten.mul.Tensor(mul_1521, unsqueeze_3662);  mul_1521 = unsqueeze_3662 = None
        unsqueeze_3663 = torch.ops.aten.unsqueeze.default(arg640_1, -1);  arg640_1 = None
        unsqueeze_3664 = torch.ops.aten.unsqueeze.default(unsqueeze_3663, -1);  unsqueeze_3663 = None
        add_1312 = torch.ops.aten.add.Tensor(mul_1522, unsqueeze_3664);  mul_1522 = unsqueeze_3664 = None
        convolution_453 = torch.ops.aten.convolution.default(relu_390, arg641_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_390 = arg641_1 = None
        add_1313 = torch.ops.aten.add.Tensor(arg643_1, 1e-05);  arg643_1 = None
        sqrt_453 = torch.ops.aten.sqrt.default(add_1313);  add_1313 = None
        reciprocal_453 = torch.ops.aten.reciprocal.default(sqrt_453);  sqrt_453 = None
        mul_1523 = torch.ops.aten.mul.Tensor(reciprocal_453, 1);  reciprocal_453 = None
        unsqueeze_3665 = torch.ops.aten.unsqueeze.default(arg642_1, -1);  arg642_1 = None
        unsqueeze_3666 = torch.ops.aten.unsqueeze.default(unsqueeze_3665, -1);  unsqueeze_3665 = None
        unsqueeze_3667 = torch.ops.aten.unsqueeze.default(mul_1523, -1);  mul_1523 = None
        unsqueeze_3668 = torch.ops.aten.unsqueeze.default(unsqueeze_3667, -1);  unsqueeze_3667 = None
        sub_453 = torch.ops.aten.sub.Tensor(convolution_453, unsqueeze_3666);  convolution_453 = unsqueeze_3666 = None
        mul_1524 = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_3668);  sub_453 = unsqueeze_3668 = None
        unsqueeze_3669 = torch.ops.aten.unsqueeze.default(arg644_1, -1);  arg644_1 = None
        unsqueeze_3670 = torch.ops.aten.unsqueeze.default(unsqueeze_3669, -1);  unsqueeze_3669 = None
        mul_1525 = torch.ops.aten.mul.Tensor(mul_1524, unsqueeze_3670);  mul_1524 = unsqueeze_3670 = None
        unsqueeze_3671 = torch.ops.aten.unsqueeze.default(arg645_1, -1);  arg645_1 = None
        unsqueeze_3672 = torch.ops.aten.unsqueeze.default(unsqueeze_3671, -1);  unsqueeze_3671 = None
        add_1314 = torch.ops.aten.add.Tensor(mul_1525, unsqueeze_3672);  mul_1525 = unsqueeze_3672 = None
        add_1315 = torch.ops.aten.add.Tensor(add_1312, add_1314);  add_1312 = add_1314 = None
        add_1316 = torch.ops.aten.add.Tensor(add_1315, relu_398);  add_1315 = relu_398 = None
        relu_402 = torch.ops.aten.relu.default(add_1316);  add_1316 = None
        convolution_454 = torch.ops.aten.convolution.default(relu_399, arg646_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg646_1 = None
        add_1317 = torch.ops.aten.add.Tensor(arg648_1, 1e-05);  arg648_1 = None
        sqrt_454 = torch.ops.aten.sqrt.default(add_1317);  add_1317 = None
        reciprocal_454 = torch.ops.aten.reciprocal.default(sqrt_454);  sqrt_454 = None
        mul_1526 = torch.ops.aten.mul.Tensor(reciprocal_454, 1);  reciprocal_454 = None
        unsqueeze_3673 = torch.ops.aten.unsqueeze.default(arg647_1, -1);  arg647_1 = None
        unsqueeze_3674 = torch.ops.aten.unsqueeze.default(unsqueeze_3673, -1);  unsqueeze_3673 = None
        unsqueeze_3675 = torch.ops.aten.unsqueeze.default(mul_1526, -1);  mul_1526 = None
        unsqueeze_3676 = torch.ops.aten.unsqueeze.default(unsqueeze_3675, -1);  unsqueeze_3675 = None
        sub_454 = torch.ops.aten.sub.Tensor(convolution_454, unsqueeze_3674);  convolution_454 = unsqueeze_3674 = None
        mul_1527 = torch.ops.aten.mul.Tensor(sub_454, unsqueeze_3676);  sub_454 = unsqueeze_3676 = None
        unsqueeze_3677 = torch.ops.aten.unsqueeze.default(arg649_1, -1);  arg649_1 = None
        unsqueeze_3678 = torch.ops.aten.unsqueeze.default(unsqueeze_3677, -1);  unsqueeze_3677 = None
        mul_1528 = torch.ops.aten.mul.Tensor(mul_1527, unsqueeze_3678);  mul_1527 = unsqueeze_3678 = None
        unsqueeze_3679 = torch.ops.aten.unsqueeze.default(arg650_1, -1);  arg650_1 = None
        unsqueeze_3680 = torch.ops.aten.unsqueeze.default(unsqueeze_3679, -1);  unsqueeze_3679 = None
        add_1318 = torch.ops.aten.add.Tensor(mul_1528, unsqueeze_3680);  mul_1528 = unsqueeze_3680 = None
        relu_403 = torch.ops.aten.relu.default(add_1318);  add_1318 = None
        convolution_455 = torch.ops.aten.convolution.default(relu_403, arg651_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_403 = arg651_1 = None
        add_1319 = torch.ops.aten.add.Tensor(arg653_1, 1e-05);  arg653_1 = None
        sqrt_455 = torch.ops.aten.sqrt.default(add_1319);  add_1319 = None
        reciprocal_455 = torch.ops.aten.reciprocal.default(sqrt_455);  sqrt_455 = None
        mul_1529 = torch.ops.aten.mul.Tensor(reciprocal_455, 1);  reciprocal_455 = None
        unsqueeze_3681 = torch.ops.aten.unsqueeze.default(arg652_1, -1);  arg652_1 = None
        unsqueeze_3682 = torch.ops.aten.unsqueeze.default(unsqueeze_3681, -1);  unsqueeze_3681 = None
        unsqueeze_3683 = torch.ops.aten.unsqueeze.default(mul_1529, -1);  mul_1529 = None
        unsqueeze_3684 = torch.ops.aten.unsqueeze.default(unsqueeze_3683, -1);  unsqueeze_3683 = None
        sub_455 = torch.ops.aten.sub.Tensor(convolution_455, unsqueeze_3682);  convolution_455 = unsqueeze_3682 = None
        mul_1530 = torch.ops.aten.mul.Tensor(sub_455, unsqueeze_3684);  sub_455 = unsqueeze_3684 = None
        unsqueeze_3685 = torch.ops.aten.unsqueeze.default(arg654_1, -1);  arg654_1 = None
        unsqueeze_3686 = torch.ops.aten.unsqueeze.default(unsqueeze_3685, -1);  unsqueeze_3685 = None
        mul_1531 = torch.ops.aten.mul.Tensor(mul_1530, unsqueeze_3686);  mul_1530 = unsqueeze_3686 = None
        unsqueeze_3687 = torch.ops.aten.unsqueeze.default(arg655_1, -1);  arg655_1 = None
        unsqueeze_3688 = torch.ops.aten.unsqueeze.default(unsqueeze_3687, -1);  unsqueeze_3687 = None
        add_1320 = torch.ops.aten.add.Tensor(mul_1531, unsqueeze_3688);  mul_1531 = unsqueeze_3688 = None
        add_1321 = torch.ops.aten.add.Tensor(add_1320, relu_399);  add_1320 = relu_399 = None
        relu_404 = torch.ops.aten.relu.default(add_1321);  add_1321 = None
        convolution_456 = torch.ops.aten.convolution.default(relu_404, arg656_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg656_1 = None
        add_1322 = torch.ops.aten.add.Tensor(arg658_1, 1e-05);  arg658_1 = None
        sqrt_456 = torch.ops.aten.sqrt.default(add_1322);  add_1322 = None
        reciprocal_456 = torch.ops.aten.reciprocal.default(sqrt_456);  sqrt_456 = None
        mul_1532 = torch.ops.aten.mul.Tensor(reciprocal_456, 1);  reciprocal_456 = None
        unsqueeze_3689 = torch.ops.aten.unsqueeze.default(arg657_1, -1);  arg657_1 = None
        unsqueeze_3690 = torch.ops.aten.unsqueeze.default(unsqueeze_3689, -1);  unsqueeze_3689 = None
        unsqueeze_3691 = torch.ops.aten.unsqueeze.default(mul_1532, -1);  mul_1532 = None
        unsqueeze_3692 = torch.ops.aten.unsqueeze.default(unsqueeze_3691, -1);  unsqueeze_3691 = None
        sub_456 = torch.ops.aten.sub.Tensor(convolution_456, unsqueeze_3690);  convolution_456 = unsqueeze_3690 = None
        mul_1533 = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_3692);  sub_456 = unsqueeze_3692 = None
        unsqueeze_3693 = torch.ops.aten.unsqueeze.default(arg659_1, -1);  arg659_1 = None
        unsqueeze_3694 = torch.ops.aten.unsqueeze.default(unsqueeze_3693, -1);  unsqueeze_3693 = None
        mul_1534 = torch.ops.aten.mul.Tensor(mul_1533, unsqueeze_3694);  mul_1533 = unsqueeze_3694 = None
        unsqueeze_3695 = torch.ops.aten.unsqueeze.default(arg660_1, -1);  arg660_1 = None
        unsqueeze_3696 = torch.ops.aten.unsqueeze.default(unsqueeze_3695, -1);  unsqueeze_3695 = None
        add_1323 = torch.ops.aten.add.Tensor(mul_1534, unsqueeze_3696);  mul_1534 = unsqueeze_3696 = None
        relu_405 = torch.ops.aten.relu.default(add_1323);  add_1323 = None
        convolution_457 = torch.ops.aten.convolution.default(relu_405, arg661_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_405 = arg661_1 = None
        add_1324 = torch.ops.aten.add.Tensor(arg663_1, 1e-05);  arg663_1 = None
        sqrt_457 = torch.ops.aten.sqrt.default(add_1324);  add_1324 = None
        reciprocal_457 = torch.ops.aten.reciprocal.default(sqrt_457);  sqrt_457 = None
        mul_1535 = torch.ops.aten.mul.Tensor(reciprocal_457, 1);  reciprocal_457 = None
        unsqueeze_3697 = torch.ops.aten.unsqueeze.default(arg662_1, -1);  arg662_1 = None
        unsqueeze_3698 = torch.ops.aten.unsqueeze.default(unsqueeze_3697, -1);  unsqueeze_3697 = None
        unsqueeze_3699 = torch.ops.aten.unsqueeze.default(mul_1535, -1);  mul_1535 = None
        unsqueeze_3700 = torch.ops.aten.unsqueeze.default(unsqueeze_3699, -1);  unsqueeze_3699 = None
        sub_457 = torch.ops.aten.sub.Tensor(convolution_457, unsqueeze_3698);  convolution_457 = unsqueeze_3698 = None
        mul_1536 = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_3700);  sub_457 = unsqueeze_3700 = None
        unsqueeze_3701 = torch.ops.aten.unsqueeze.default(arg664_1, -1);  arg664_1 = None
        unsqueeze_3702 = torch.ops.aten.unsqueeze.default(unsqueeze_3701, -1);  unsqueeze_3701 = None
        mul_1537 = torch.ops.aten.mul.Tensor(mul_1536, unsqueeze_3702);  mul_1536 = unsqueeze_3702 = None
        unsqueeze_3703 = torch.ops.aten.unsqueeze.default(arg665_1, -1);  arg665_1 = None
        unsqueeze_3704 = torch.ops.aten.unsqueeze.default(unsqueeze_3703, -1);  unsqueeze_3703 = None
        add_1325 = torch.ops.aten.add.Tensor(mul_1537, unsqueeze_3704);  mul_1537 = unsqueeze_3704 = None
        add_1326 = torch.ops.aten.add.Tensor(add_1325, relu_404);  add_1325 = relu_404 = None
        relu_406 = torch.ops.aten.relu.default(add_1326);  add_1326 = None
        convolution_458 = torch.ops.aten.convolution.default(relu_406, arg666_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg666_1 = None
        add_1327 = torch.ops.aten.add.Tensor(arg668_1, 1e-05);  arg668_1 = None
        sqrt_458 = torch.ops.aten.sqrt.default(add_1327);  add_1327 = None
        reciprocal_458 = torch.ops.aten.reciprocal.default(sqrt_458);  sqrt_458 = None
        mul_1538 = torch.ops.aten.mul.Tensor(reciprocal_458, 1);  reciprocal_458 = None
        unsqueeze_3705 = torch.ops.aten.unsqueeze.default(arg667_1, -1);  arg667_1 = None
        unsqueeze_3706 = torch.ops.aten.unsqueeze.default(unsqueeze_3705, -1);  unsqueeze_3705 = None
        unsqueeze_3707 = torch.ops.aten.unsqueeze.default(mul_1538, -1);  mul_1538 = None
        unsqueeze_3708 = torch.ops.aten.unsqueeze.default(unsqueeze_3707, -1);  unsqueeze_3707 = None
        sub_458 = torch.ops.aten.sub.Tensor(convolution_458, unsqueeze_3706);  convolution_458 = unsqueeze_3706 = None
        mul_1539 = torch.ops.aten.mul.Tensor(sub_458, unsqueeze_3708);  sub_458 = unsqueeze_3708 = None
        unsqueeze_3709 = torch.ops.aten.unsqueeze.default(arg669_1, -1);  arg669_1 = None
        unsqueeze_3710 = torch.ops.aten.unsqueeze.default(unsqueeze_3709, -1);  unsqueeze_3709 = None
        mul_1540 = torch.ops.aten.mul.Tensor(mul_1539, unsqueeze_3710);  mul_1539 = unsqueeze_3710 = None
        unsqueeze_3711 = torch.ops.aten.unsqueeze.default(arg670_1, -1);  arg670_1 = None
        unsqueeze_3712 = torch.ops.aten.unsqueeze.default(unsqueeze_3711, -1);  unsqueeze_3711 = None
        add_1328 = torch.ops.aten.add.Tensor(mul_1540, unsqueeze_3712);  mul_1540 = unsqueeze_3712 = None
        relu_407 = torch.ops.aten.relu.default(add_1328);  add_1328 = None
        convolution_459 = torch.ops.aten.convolution.default(relu_407, arg671_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_407 = arg671_1 = None
        add_1329 = torch.ops.aten.add.Tensor(arg673_1, 1e-05);  arg673_1 = None
        sqrt_459 = torch.ops.aten.sqrt.default(add_1329);  add_1329 = None
        reciprocal_459 = torch.ops.aten.reciprocal.default(sqrt_459);  sqrt_459 = None
        mul_1541 = torch.ops.aten.mul.Tensor(reciprocal_459, 1);  reciprocal_459 = None
        unsqueeze_3713 = torch.ops.aten.unsqueeze.default(arg672_1, -1);  arg672_1 = None
        unsqueeze_3714 = torch.ops.aten.unsqueeze.default(unsqueeze_3713, -1);  unsqueeze_3713 = None
        unsqueeze_3715 = torch.ops.aten.unsqueeze.default(mul_1541, -1);  mul_1541 = None
        unsqueeze_3716 = torch.ops.aten.unsqueeze.default(unsqueeze_3715, -1);  unsqueeze_3715 = None
        sub_459 = torch.ops.aten.sub.Tensor(convolution_459, unsqueeze_3714);  convolution_459 = unsqueeze_3714 = None
        mul_1542 = torch.ops.aten.mul.Tensor(sub_459, unsqueeze_3716);  sub_459 = unsqueeze_3716 = None
        unsqueeze_3717 = torch.ops.aten.unsqueeze.default(arg674_1, -1);  arg674_1 = None
        unsqueeze_3718 = torch.ops.aten.unsqueeze.default(unsqueeze_3717, -1);  unsqueeze_3717 = None
        mul_1543 = torch.ops.aten.mul.Tensor(mul_1542, unsqueeze_3718);  mul_1542 = unsqueeze_3718 = None
        unsqueeze_3719 = torch.ops.aten.unsqueeze.default(arg675_1, -1);  arg675_1 = None
        unsqueeze_3720 = torch.ops.aten.unsqueeze.default(unsqueeze_3719, -1);  unsqueeze_3719 = None
        add_1330 = torch.ops.aten.add.Tensor(mul_1543, unsqueeze_3720);  mul_1543 = unsqueeze_3720 = None
        add_1331 = torch.ops.aten.add.Tensor(add_1330, relu_406);  add_1330 = relu_406 = None
        relu_408 = torch.ops.aten.relu.default(add_1331);  add_1331 = None
        convolution_460 = torch.ops.aten.convolution.default(relu_408, arg676_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg676_1 = None
        add_1332 = torch.ops.aten.add.Tensor(arg678_1, 1e-05);  arg678_1 = None
        sqrt_460 = torch.ops.aten.sqrt.default(add_1332);  add_1332 = None
        reciprocal_460 = torch.ops.aten.reciprocal.default(sqrt_460);  sqrt_460 = None
        mul_1544 = torch.ops.aten.mul.Tensor(reciprocal_460, 1);  reciprocal_460 = None
        unsqueeze_3721 = torch.ops.aten.unsqueeze.default(arg677_1, -1);  arg677_1 = None
        unsqueeze_3722 = torch.ops.aten.unsqueeze.default(unsqueeze_3721, -1);  unsqueeze_3721 = None
        unsqueeze_3723 = torch.ops.aten.unsqueeze.default(mul_1544, -1);  mul_1544 = None
        unsqueeze_3724 = torch.ops.aten.unsqueeze.default(unsqueeze_3723, -1);  unsqueeze_3723 = None
        sub_460 = torch.ops.aten.sub.Tensor(convolution_460, unsqueeze_3722);  convolution_460 = unsqueeze_3722 = None
        mul_1545 = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_3724);  sub_460 = unsqueeze_3724 = None
        unsqueeze_3725 = torch.ops.aten.unsqueeze.default(arg679_1, -1);  arg679_1 = None
        unsqueeze_3726 = torch.ops.aten.unsqueeze.default(unsqueeze_3725, -1);  unsqueeze_3725 = None
        mul_1546 = torch.ops.aten.mul.Tensor(mul_1545, unsqueeze_3726);  mul_1545 = unsqueeze_3726 = None
        unsqueeze_3727 = torch.ops.aten.unsqueeze.default(arg680_1, -1);  arg680_1 = None
        unsqueeze_3728 = torch.ops.aten.unsqueeze.default(unsqueeze_3727, -1);  unsqueeze_3727 = None
        add_1333 = torch.ops.aten.add.Tensor(mul_1546, unsqueeze_3728);  mul_1546 = unsqueeze_3728 = None
        relu_409 = torch.ops.aten.relu.default(add_1333);  add_1333 = None
        convolution_461 = torch.ops.aten.convolution.default(relu_409, arg681_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_409 = arg681_1 = None
        add_1334 = torch.ops.aten.add.Tensor(arg683_1, 1e-05);  arg683_1 = None
        sqrt_461 = torch.ops.aten.sqrt.default(add_1334);  add_1334 = None
        reciprocal_461 = torch.ops.aten.reciprocal.default(sqrt_461);  sqrt_461 = None
        mul_1547 = torch.ops.aten.mul.Tensor(reciprocal_461, 1);  reciprocal_461 = None
        unsqueeze_3729 = torch.ops.aten.unsqueeze.default(arg682_1, -1);  arg682_1 = None
        unsqueeze_3730 = torch.ops.aten.unsqueeze.default(unsqueeze_3729, -1);  unsqueeze_3729 = None
        unsqueeze_3731 = torch.ops.aten.unsqueeze.default(mul_1547, -1);  mul_1547 = None
        unsqueeze_3732 = torch.ops.aten.unsqueeze.default(unsqueeze_3731, -1);  unsqueeze_3731 = None
        sub_461 = torch.ops.aten.sub.Tensor(convolution_461, unsqueeze_3730);  convolution_461 = unsqueeze_3730 = None
        mul_1548 = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_3732);  sub_461 = unsqueeze_3732 = None
        unsqueeze_3733 = torch.ops.aten.unsqueeze.default(arg684_1, -1);  arg684_1 = None
        unsqueeze_3734 = torch.ops.aten.unsqueeze.default(unsqueeze_3733, -1);  unsqueeze_3733 = None
        mul_1549 = torch.ops.aten.mul.Tensor(mul_1548, unsqueeze_3734);  mul_1548 = unsqueeze_3734 = None
        unsqueeze_3735 = torch.ops.aten.unsqueeze.default(arg685_1, -1);  arg685_1 = None
        unsqueeze_3736 = torch.ops.aten.unsqueeze.default(unsqueeze_3735, -1);  unsqueeze_3735 = None
        add_1335 = torch.ops.aten.add.Tensor(mul_1549, unsqueeze_3736);  mul_1549 = unsqueeze_3736 = None
        add_1336 = torch.ops.aten.add.Tensor(add_1335, relu_408);  add_1335 = relu_408 = None
        relu_410 = torch.ops.aten.relu.default(add_1336);  add_1336 = None
        convolution_462 = torch.ops.aten.convolution.default(relu_400, arg686_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg686_1 = None
        add_1337 = torch.ops.aten.add.Tensor(arg688_1, 1e-05);  arg688_1 = None
        sqrt_462 = torch.ops.aten.sqrt.default(add_1337);  add_1337 = None
        reciprocal_462 = torch.ops.aten.reciprocal.default(sqrt_462);  sqrt_462 = None
        mul_1550 = torch.ops.aten.mul.Tensor(reciprocal_462, 1);  reciprocal_462 = None
        unsqueeze_3737 = torch.ops.aten.unsqueeze.default(arg687_1, -1);  arg687_1 = None
        unsqueeze_3738 = torch.ops.aten.unsqueeze.default(unsqueeze_3737, -1);  unsqueeze_3737 = None
        unsqueeze_3739 = torch.ops.aten.unsqueeze.default(mul_1550, -1);  mul_1550 = None
        unsqueeze_3740 = torch.ops.aten.unsqueeze.default(unsqueeze_3739, -1);  unsqueeze_3739 = None
        sub_462 = torch.ops.aten.sub.Tensor(convolution_462, unsqueeze_3738);  convolution_462 = unsqueeze_3738 = None
        mul_1551 = torch.ops.aten.mul.Tensor(sub_462, unsqueeze_3740);  sub_462 = unsqueeze_3740 = None
        unsqueeze_3741 = torch.ops.aten.unsqueeze.default(arg689_1, -1);  arg689_1 = None
        unsqueeze_3742 = torch.ops.aten.unsqueeze.default(unsqueeze_3741, -1);  unsqueeze_3741 = None
        mul_1552 = torch.ops.aten.mul.Tensor(mul_1551, unsqueeze_3742);  mul_1551 = unsqueeze_3742 = None
        unsqueeze_3743 = torch.ops.aten.unsqueeze.default(arg690_1, -1);  arg690_1 = None
        unsqueeze_3744 = torch.ops.aten.unsqueeze.default(unsqueeze_3743, -1);  unsqueeze_3743 = None
        add_1338 = torch.ops.aten.add.Tensor(mul_1552, unsqueeze_3744);  mul_1552 = unsqueeze_3744 = None
        relu_411 = torch.ops.aten.relu.default(add_1338);  add_1338 = None
        convolution_463 = torch.ops.aten.convolution.default(relu_411, arg691_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_411 = arg691_1 = None
        add_1339 = torch.ops.aten.add.Tensor(arg693_1, 1e-05);  arg693_1 = None
        sqrt_463 = torch.ops.aten.sqrt.default(add_1339);  add_1339 = None
        reciprocal_463 = torch.ops.aten.reciprocal.default(sqrt_463);  sqrt_463 = None
        mul_1553 = torch.ops.aten.mul.Tensor(reciprocal_463, 1);  reciprocal_463 = None
        unsqueeze_3745 = torch.ops.aten.unsqueeze.default(arg692_1, -1);  arg692_1 = None
        unsqueeze_3746 = torch.ops.aten.unsqueeze.default(unsqueeze_3745, -1);  unsqueeze_3745 = None
        unsqueeze_3747 = torch.ops.aten.unsqueeze.default(mul_1553, -1);  mul_1553 = None
        unsqueeze_3748 = torch.ops.aten.unsqueeze.default(unsqueeze_3747, -1);  unsqueeze_3747 = None
        sub_463 = torch.ops.aten.sub.Tensor(convolution_463, unsqueeze_3746);  convolution_463 = unsqueeze_3746 = None
        mul_1554 = torch.ops.aten.mul.Tensor(sub_463, unsqueeze_3748);  sub_463 = unsqueeze_3748 = None
        unsqueeze_3749 = torch.ops.aten.unsqueeze.default(arg694_1, -1);  arg694_1 = None
        unsqueeze_3750 = torch.ops.aten.unsqueeze.default(unsqueeze_3749, -1);  unsqueeze_3749 = None
        mul_1555 = torch.ops.aten.mul.Tensor(mul_1554, unsqueeze_3750);  mul_1554 = unsqueeze_3750 = None
        unsqueeze_3751 = torch.ops.aten.unsqueeze.default(arg695_1, -1);  arg695_1 = None
        unsqueeze_3752 = torch.ops.aten.unsqueeze.default(unsqueeze_3751, -1);  unsqueeze_3751 = None
        add_1340 = torch.ops.aten.add.Tensor(mul_1555, unsqueeze_3752);  mul_1555 = unsqueeze_3752 = None
        add_1341 = torch.ops.aten.add.Tensor(add_1340, relu_400);  add_1340 = relu_400 = None
        relu_412 = torch.ops.aten.relu.default(add_1341);  add_1341 = None
        convolution_464 = torch.ops.aten.convolution.default(relu_412, arg696_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg696_1 = None
        add_1342 = torch.ops.aten.add.Tensor(arg698_1, 1e-05);  arg698_1 = None
        sqrt_464 = torch.ops.aten.sqrt.default(add_1342);  add_1342 = None
        reciprocal_464 = torch.ops.aten.reciprocal.default(sqrt_464);  sqrt_464 = None
        mul_1556 = torch.ops.aten.mul.Tensor(reciprocal_464, 1);  reciprocal_464 = None
        unsqueeze_3753 = torch.ops.aten.unsqueeze.default(arg697_1, -1);  arg697_1 = None
        unsqueeze_3754 = torch.ops.aten.unsqueeze.default(unsqueeze_3753, -1);  unsqueeze_3753 = None
        unsqueeze_3755 = torch.ops.aten.unsqueeze.default(mul_1556, -1);  mul_1556 = None
        unsqueeze_3756 = torch.ops.aten.unsqueeze.default(unsqueeze_3755, -1);  unsqueeze_3755 = None
        sub_464 = torch.ops.aten.sub.Tensor(convolution_464, unsqueeze_3754);  convolution_464 = unsqueeze_3754 = None
        mul_1557 = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_3756);  sub_464 = unsqueeze_3756 = None
        unsqueeze_3757 = torch.ops.aten.unsqueeze.default(arg699_1, -1);  arg699_1 = None
        unsqueeze_3758 = torch.ops.aten.unsqueeze.default(unsqueeze_3757, -1);  unsqueeze_3757 = None
        mul_1558 = torch.ops.aten.mul.Tensor(mul_1557, unsqueeze_3758);  mul_1557 = unsqueeze_3758 = None
        unsqueeze_3759 = torch.ops.aten.unsqueeze.default(arg700_1, -1);  arg700_1 = None
        unsqueeze_3760 = torch.ops.aten.unsqueeze.default(unsqueeze_3759, -1);  unsqueeze_3759 = None
        add_1343 = torch.ops.aten.add.Tensor(mul_1558, unsqueeze_3760);  mul_1558 = unsqueeze_3760 = None
        relu_413 = torch.ops.aten.relu.default(add_1343);  add_1343 = None
        convolution_465 = torch.ops.aten.convolution.default(relu_413, arg701_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_413 = arg701_1 = None
        add_1344 = torch.ops.aten.add.Tensor(arg703_1, 1e-05);  arg703_1 = None
        sqrt_465 = torch.ops.aten.sqrt.default(add_1344);  add_1344 = None
        reciprocal_465 = torch.ops.aten.reciprocal.default(sqrt_465);  sqrt_465 = None
        mul_1559 = torch.ops.aten.mul.Tensor(reciprocal_465, 1);  reciprocal_465 = None
        unsqueeze_3761 = torch.ops.aten.unsqueeze.default(arg702_1, -1);  arg702_1 = None
        unsqueeze_3762 = torch.ops.aten.unsqueeze.default(unsqueeze_3761, -1);  unsqueeze_3761 = None
        unsqueeze_3763 = torch.ops.aten.unsqueeze.default(mul_1559, -1);  mul_1559 = None
        unsqueeze_3764 = torch.ops.aten.unsqueeze.default(unsqueeze_3763, -1);  unsqueeze_3763 = None
        sub_465 = torch.ops.aten.sub.Tensor(convolution_465, unsqueeze_3762);  convolution_465 = unsqueeze_3762 = None
        mul_1560 = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_3764);  sub_465 = unsqueeze_3764 = None
        unsqueeze_3765 = torch.ops.aten.unsqueeze.default(arg704_1, -1);  arg704_1 = None
        unsqueeze_3766 = torch.ops.aten.unsqueeze.default(unsqueeze_3765, -1);  unsqueeze_3765 = None
        mul_1561 = torch.ops.aten.mul.Tensor(mul_1560, unsqueeze_3766);  mul_1560 = unsqueeze_3766 = None
        unsqueeze_3767 = torch.ops.aten.unsqueeze.default(arg705_1, -1);  arg705_1 = None
        unsqueeze_3768 = torch.ops.aten.unsqueeze.default(unsqueeze_3767, -1);  unsqueeze_3767 = None
        add_1345 = torch.ops.aten.add.Tensor(mul_1561, unsqueeze_3768);  mul_1561 = unsqueeze_3768 = None
        add_1346 = torch.ops.aten.add.Tensor(add_1345, relu_412);  add_1345 = relu_412 = None
        relu_414 = torch.ops.aten.relu.default(add_1346);  add_1346 = None
        convolution_466 = torch.ops.aten.convolution.default(relu_414, arg706_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg706_1 = None
        add_1347 = torch.ops.aten.add.Tensor(arg708_1, 1e-05);  arg708_1 = None
        sqrt_466 = torch.ops.aten.sqrt.default(add_1347);  add_1347 = None
        reciprocal_466 = torch.ops.aten.reciprocal.default(sqrt_466);  sqrt_466 = None
        mul_1562 = torch.ops.aten.mul.Tensor(reciprocal_466, 1);  reciprocal_466 = None
        unsqueeze_3769 = torch.ops.aten.unsqueeze.default(arg707_1, -1);  arg707_1 = None
        unsqueeze_3770 = torch.ops.aten.unsqueeze.default(unsqueeze_3769, -1);  unsqueeze_3769 = None
        unsqueeze_3771 = torch.ops.aten.unsqueeze.default(mul_1562, -1);  mul_1562 = None
        unsqueeze_3772 = torch.ops.aten.unsqueeze.default(unsqueeze_3771, -1);  unsqueeze_3771 = None
        sub_466 = torch.ops.aten.sub.Tensor(convolution_466, unsqueeze_3770);  convolution_466 = unsqueeze_3770 = None
        mul_1563 = torch.ops.aten.mul.Tensor(sub_466, unsqueeze_3772);  sub_466 = unsqueeze_3772 = None
        unsqueeze_3773 = torch.ops.aten.unsqueeze.default(arg709_1, -1);  arg709_1 = None
        unsqueeze_3774 = torch.ops.aten.unsqueeze.default(unsqueeze_3773, -1);  unsqueeze_3773 = None
        mul_1564 = torch.ops.aten.mul.Tensor(mul_1563, unsqueeze_3774);  mul_1563 = unsqueeze_3774 = None
        unsqueeze_3775 = torch.ops.aten.unsqueeze.default(arg710_1, -1);  arg710_1 = None
        unsqueeze_3776 = torch.ops.aten.unsqueeze.default(unsqueeze_3775, -1);  unsqueeze_3775 = None
        add_1348 = torch.ops.aten.add.Tensor(mul_1564, unsqueeze_3776);  mul_1564 = unsqueeze_3776 = None
        relu_415 = torch.ops.aten.relu.default(add_1348);  add_1348 = None
        convolution_467 = torch.ops.aten.convolution.default(relu_415, arg711_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_415 = arg711_1 = None
        add_1349 = torch.ops.aten.add.Tensor(arg713_1, 1e-05);  arg713_1 = None
        sqrt_467 = torch.ops.aten.sqrt.default(add_1349);  add_1349 = None
        reciprocal_467 = torch.ops.aten.reciprocal.default(sqrt_467);  sqrt_467 = None
        mul_1565 = torch.ops.aten.mul.Tensor(reciprocal_467, 1);  reciprocal_467 = None
        unsqueeze_3777 = torch.ops.aten.unsqueeze.default(arg712_1, -1);  arg712_1 = None
        unsqueeze_3778 = torch.ops.aten.unsqueeze.default(unsqueeze_3777, -1);  unsqueeze_3777 = None
        unsqueeze_3779 = torch.ops.aten.unsqueeze.default(mul_1565, -1);  mul_1565 = None
        unsqueeze_3780 = torch.ops.aten.unsqueeze.default(unsqueeze_3779, -1);  unsqueeze_3779 = None
        sub_467 = torch.ops.aten.sub.Tensor(convolution_467, unsqueeze_3778);  convolution_467 = unsqueeze_3778 = None
        mul_1566 = torch.ops.aten.mul.Tensor(sub_467, unsqueeze_3780);  sub_467 = unsqueeze_3780 = None
        unsqueeze_3781 = torch.ops.aten.unsqueeze.default(arg714_1, -1);  arg714_1 = None
        unsqueeze_3782 = torch.ops.aten.unsqueeze.default(unsqueeze_3781, -1);  unsqueeze_3781 = None
        mul_1567 = torch.ops.aten.mul.Tensor(mul_1566, unsqueeze_3782);  mul_1566 = unsqueeze_3782 = None
        unsqueeze_3783 = torch.ops.aten.unsqueeze.default(arg715_1, -1);  arg715_1 = None
        unsqueeze_3784 = torch.ops.aten.unsqueeze.default(unsqueeze_3783, -1);  unsqueeze_3783 = None
        add_1350 = torch.ops.aten.add.Tensor(mul_1567, unsqueeze_3784);  mul_1567 = unsqueeze_3784 = None
        add_1351 = torch.ops.aten.add.Tensor(add_1350, relu_414);  add_1350 = relu_414 = None
        relu_416 = torch.ops.aten.relu.default(add_1351);  add_1351 = None
        convolution_468 = torch.ops.aten.convolution.default(relu_416, arg716_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg716_1 = None
        add_1352 = torch.ops.aten.add.Tensor(arg718_1, 1e-05);  arg718_1 = None
        sqrt_468 = torch.ops.aten.sqrt.default(add_1352);  add_1352 = None
        reciprocal_468 = torch.ops.aten.reciprocal.default(sqrt_468);  sqrt_468 = None
        mul_1568 = torch.ops.aten.mul.Tensor(reciprocal_468, 1);  reciprocal_468 = None
        unsqueeze_3785 = torch.ops.aten.unsqueeze.default(arg717_1, -1);  arg717_1 = None
        unsqueeze_3786 = torch.ops.aten.unsqueeze.default(unsqueeze_3785, -1);  unsqueeze_3785 = None
        unsqueeze_3787 = torch.ops.aten.unsqueeze.default(mul_1568, -1);  mul_1568 = None
        unsqueeze_3788 = torch.ops.aten.unsqueeze.default(unsqueeze_3787, -1);  unsqueeze_3787 = None
        sub_468 = torch.ops.aten.sub.Tensor(convolution_468, unsqueeze_3786);  convolution_468 = unsqueeze_3786 = None
        mul_1569 = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_3788);  sub_468 = unsqueeze_3788 = None
        unsqueeze_3789 = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
        unsqueeze_3790 = torch.ops.aten.unsqueeze.default(unsqueeze_3789, -1);  unsqueeze_3789 = None
        mul_1570 = torch.ops.aten.mul.Tensor(mul_1569, unsqueeze_3790);  mul_1569 = unsqueeze_3790 = None
        unsqueeze_3791 = torch.ops.aten.unsqueeze.default(arg720_1, -1);  arg720_1 = None
        unsqueeze_3792 = torch.ops.aten.unsqueeze.default(unsqueeze_3791, -1);  unsqueeze_3791 = None
        add_1353 = torch.ops.aten.add.Tensor(mul_1570, unsqueeze_3792);  mul_1570 = unsqueeze_3792 = None
        relu_417 = torch.ops.aten.relu.default(add_1353);  add_1353 = None
        convolution_469 = torch.ops.aten.convolution.default(relu_417, arg721_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_417 = arg721_1 = None
        add_1354 = torch.ops.aten.add.Tensor(arg723_1, 1e-05);  arg723_1 = None
        sqrt_469 = torch.ops.aten.sqrt.default(add_1354);  add_1354 = None
        reciprocal_469 = torch.ops.aten.reciprocal.default(sqrt_469);  sqrt_469 = None
        mul_1571 = torch.ops.aten.mul.Tensor(reciprocal_469, 1);  reciprocal_469 = None
        unsqueeze_3793 = torch.ops.aten.unsqueeze.default(arg722_1, -1);  arg722_1 = None
        unsqueeze_3794 = torch.ops.aten.unsqueeze.default(unsqueeze_3793, -1);  unsqueeze_3793 = None
        unsqueeze_3795 = torch.ops.aten.unsqueeze.default(mul_1571, -1);  mul_1571 = None
        unsqueeze_3796 = torch.ops.aten.unsqueeze.default(unsqueeze_3795, -1);  unsqueeze_3795 = None
        sub_469 = torch.ops.aten.sub.Tensor(convolution_469, unsqueeze_3794);  convolution_469 = unsqueeze_3794 = None
        mul_1572 = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_3796);  sub_469 = unsqueeze_3796 = None
        unsqueeze_3797 = torch.ops.aten.unsqueeze.default(arg724_1, -1);  arg724_1 = None
        unsqueeze_3798 = torch.ops.aten.unsqueeze.default(unsqueeze_3797, -1);  unsqueeze_3797 = None
        mul_1573 = torch.ops.aten.mul.Tensor(mul_1572, unsqueeze_3798);  mul_1572 = unsqueeze_3798 = None
        unsqueeze_3799 = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
        unsqueeze_3800 = torch.ops.aten.unsqueeze.default(unsqueeze_3799, -1);  unsqueeze_3799 = None
        add_1355 = torch.ops.aten.add.Tensor(mul_1573, unsqueeze_3800);  mul_1573 = unsqueeze_3800 = None
        add_1356 = torch.ops.aten.add.Tensor(add_1355, relu_416);  add_1355 = relu_416 = None
        relu_418 = torch.ops.aten.relu.default(add_1356);  add_1356 = None
        convolution_470 = torch.ops.aten.convolution.default(relu_402, arg726_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg726_1 = None
        add_1357 = torch.ops.aten.add.Tensor(arg728_1, 1e-05);  arg728_1 = None
        sqrt_470 = torch.ops.aten.sqrt.default(add_1357);  add_1357 = None
        reciprocal_470 = torch.ops.aten.reciprocal.default(sqrt_470);  sqrt_470 = None
        mul_1574 = torch.ops.aten.mul.Tensor(reciprocal_470, 1);  reciprocal_470 = None
        unsqueeze_3801 = torch.ops.aten.unsqueeze.default(arg727_1, -1);  arg727_1 = None
        unsqueeze_3802 = torch.ops.aten.unsqueeze.default(unsqueeze_3801, -1);  unsqueeze_3801 = None
        unsqueeze_3803 = torch.ops.aten.unsqueeze.default(mul_1574, -1);  mul_1574 = None
        unsqueeze_3804 = torch.ops.aten.unsqueeze.default(unsqueeze_3803, -1);  unsqueeze_3803 = None
        sub_470 = torch.ops.aten.sub.Tensor(convolution_470, unsqueeze_3802);  convolution_470 = unsqueeze_3802 = None
        mul_1575 = torch.ops.aten.mul.Tensor(sub_470, unsqueeze_3804);  sub_470 = unsqueeze_3804 = None
        unsqueeze_3805 = torch.ops.aten.unsqueeze.default(arg729_1, -1);  arg729_1 = None
        unsqueeze_3806 = torch.ops.aten.unsqueeze.default(unsqueeze_3805, -1);  unsqueeze_3805 = None
        mul_1576 = torch.ops.aten.mul.Tensor(mul_1575, unsqueeze_3806);  mul_1575 = unsqueeze_3806 = None
        unsqueeze_3807 = torch.ops.aten.unsqueeze.default(arg730_1, -1);  arg730_1 = None
        unsqueeze_3808 = torch.ops.aten.unsqueeze.default(unsqueeze_3807, -1);  unsqueeze_3807 = None
        add_1358 = torch.ops.aten.add.Tensor(mul_1576, unsqueeze_3808);  mul_1576 = unsqueeze_3808 = None
        relu_419 = torch.ops.aten.relu.default(add_1358);  add_1358 = None
        convolution_471 = torch.ops.aten.convolution.default(relu_419, arg731_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_419 = arg731_1 = None
        add_1359 = torch.ops.aten.add.Tensor(arg733_1, 1e-05);  arg733_1 = None
        sqrt_471 = torch.ops.aten.sqrt.default(add_1359);  add_1359 = None
        reciprocal_471 = torch.ops.aten.reciprocal.default(sqrt_471);  sqrt_471 = None
        mul_1577 = torch.ops.aten.mul.Tensor(reciprocal_471, 1);  reciprocal_471 = None
        unsqueeze_3809 = torch.ops.aten.unsqueeze.default(arg732_1, -1);  arg732_1 = None
        unsqueeze_3810 = torch.ops.aten.unsqueeze.default(unsqueeze_3809, -1);  unsqueeze_3809 = None
        unsqueeze_3811 = torch.ops.aten.unsqueeze.default(mul_1577, -1);  mul_1577 = None
        unsqueeze_3812 = torch.ops.aten.unsqueeze.default(unsqueeze_3811, -1);  unsqueeze_3811 = None
        sub_471 = torch.ops.aten.sub.Tensor(convolution_471, unsqueeze_3810);  convolution_471 = unsqueeze_3810 = None
        mul_1578 = torch.ops.aten.mul.Tensor(sub_471, unsqueeze_3812);  sub_471 = unsqueeze_3812 = None
        unsqueeze_3813 = torch.ops.aten.unsqueeze.default(arg734_1, -1);  arg734_1 = None
        unsqueeze_3814 = torch.ops.aten.unsqueeze.default(unsqueeze_3813, -1);  unsqueeze_3813 = None
        mul_1579 = torch.ops.aten.mul.Tensor(mul_1578, unsqueeze_3814);  mul_1578 = unsqueeze_3814 = None
        unsqueeze_3815 = torch.ops.aten.unsqueeze.default(arg735_1, -1);  arg735_1 = None
        unsqueeze_3816 = torch.ops.aten.unsqueeze.default(unsqueeze_3815, -1);  unsqueeze_3815 = None
        add_1360 = torch.ops.aten.add.Tensor(mul_1579, unsqueeze_3816);  mul_1579 = unsqueeze_3816 = None
        add_1361 = torch.ops.aten.add.Tensor(add_1360, relu_402);  add_1360 = relu_402 = None
        relu_420 = torch.ops.aten.relu.default(add_1361);  add_1361 = None
        convolution_472 = torch.ops.aten.convolution.default(relu_420, arg736_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg736_1 = None
        add_1362 = torch.ops.aten.add.Tensor(arg738_1, 1e-05);  arg738_1 = None
        sqrt_472 = torch.ops.aten.sqrt.default(add_1362);  add_1362 = None
        reciprocal_472 = torch.ops.aten.reciprocal.default(sqrt_472);  sqrt_472 = None
        mul_1580 = torch.ops.aten.mul.Tensor(reciprocal_472, 1);  reciprocal_472 = None
        unsqueeze_3817 = torch.ops.aten.unsqueeze.default(arg737_1, -1);  arg737_1 = None
        unsqueeze_3818 = torch.ops.aten.unsqueeze.default(unsqueeze_3817, -1);  unsqueeze_3817 = None
        unsqueeze_3819 = torch.ops.aten.unsqueeze.default(mul_1580, -1);  mul_1580 = None
        unsqueeze_3820 = torch.ops.aten.unsqueeze.default(unsqueeze_3819, -1);  unsqueeze_3819 = None
        sub_472 = torch.ops.aten.sub.Tensor(convolution_472, unsqueeze_3818);  convolution_472 = unsqueeze_3818 = None
        mul_1581 = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_3820);  sub_472 = unsqueeze_3820 = None
        unsqueeze_3821 = torch.ops.aten.unsqueeze.default(arg739_1, -1);  arg739_1 = None
        unsqueeze_3822 = torch.ops.aten.unsqueeze.default(unsqueeze_3821, -1);  unsqueeze_3821 = None
        mul_1582 = torch.ops.aten.mul.Tensor(mul_1581, unsqueeze_3822);  mul_1581 = unsqueeze_3822 = None
        unsqueeze_3823 = torch.ops.aten.unsqueeze.default(arg740_1, -1);  arg740_1 = None
        unsqueeze_3824 = torch.ops.aten.unsqueeze.default(unsqueeze_3823, -1);  unsqueeze_3823 = None
        add_1363 = torch.ops.aten.add.Tensor(mul_1582, unsqueeze_3824);  mul_1582 = unsqueeze_3824 = None
        relu_421 = torch.ops.aten.relu.default(add_1363);  add_1363 = None
        convolution_473 = torch.ops.aten.convolution.default(relu_421, arg741_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_421 = arg741_1 = None
        add_1364 = torch.ops.aten.add.Tensor(arg743_1, 1e-05);  arg743_1 = None
        sqrt_473 = torch.ops.aten.sqrt.default(add_1364);  add_1364 = None
        reciprocal_473 = torch.ops.aten.reciprocal.default(sqrt_473);  sqrt_473 = None
        mul_1583 = torch.ops.aten.mul.Tensor(reciprocal_473, 1);  reciprocal_473 = None
        unsqueeze_3825 = torch.ops.aten.unsqueeze.default(arg742_1, -1);  arg742_1 = None
        unsqueeze_3826 = torch.ops.aten.unsqueeze.default(unsqueeze_3825, -1);  unsqueeze_3825 = None
        unsqueeze_3827 = torch.ops.aten.unsqueeze.default(mul_1583, -1);  mul_1583 = None
        unsqueeze_3828 = torch.ops.aten.unsqueeze.default(unsqueeze_3827, -1);  unsqueeze_3827 = None
        sub_473 = torch.ops.aten.sub.Tensor(convolution_473, unsqueeze_3826);  convolution_473 = unsqueeze_3826 = None
        mul_1584 = torch.ops.aten.mul.Tensor(sub_473, unsqueeze_3828);  sub_473 = unsqueeze_3828 = None
        unsqueeze_3829 = torch.ops.aten.unsqueeze.default(arg744_1, -1);  arg744_1 = None
        unsqueeze_3830 = torch.ops.aten.unsqueeze.default(unsqueeze_3829, -1);  unsqueeze_3829 = None
        mul_1585 = torch.ops.aten.mul.Tensor(mul_1584, unsqueeze_3830);  mul_1584 = unsqueeze_3830 = None
        unsqueeze_3831 = torch.ops.aten.unsqueeze.default(arg745_1, -1);  arg745_1 = None
        unsqueeze_3832 = torch.ops.aten.unsqueeze.default(unsqueeze_3831, -1);  unsqueeze_3831 = None
        add_1365 = torch.ops.aten.add.Tensor(mul_1585, unsqueeze_3832);  mul_1585 = unsqueeze_3832 = None
        add_1366 = torch.ops.aten.add.Tensor(add_1365, relu_420);  add_1365 = relu_420 = None
        relu_422 = torch.ops.aten.relu.default(add_1366);  add_1366 = None
        convolution_474 = torch.ops.aten.convolution.default(relu_422, arg746_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg746_1 = None
        add_1367 = torch.ops.aten.add.Tensor(arg748_1, 1e-05);  arg748_1 = None
        sqrt_474 = torch.ops.aten.sqrt.default(add_1367);  add_1367 = None
        reciprocal_474 = torch.ops.aten.reciprocal.default(sqrt_474);  sqrt_474 = None
        mul_1586 = torch.ops.aten.mul.Tensor(reciprocal_474, 1);  reciprocal_474 = None
        unsqueeze_3833 = torch.ops.aten.unsqueeze.default(arg747_1, -1);  arg747_1 = None
        unsqueeze_3834 = torch.ops.aten.unsqueeze.default(unsqueeze_3833, -1);  unsqueeze_3833 = None
        unsqueeze_3835 = torch.ops.aten.unsqueeze.default(mul_1586, -1);  mul_1586 = None
        unsqueeze_3836 = torch.ops.aten.unsqueeze.default(unsqueeze_3835, -1);  unsqueeze_3835 = None
        sub_474 = torch.ops.aten.sub.Tensor(convolution_474, unsqueeze_3834);  convolution_474 = unsqueeze_3834 = None
        mul_1587 = torch.ops.aten.mul.Tensor(sub_474, unsqueeze_3836);  sub_474 = unsqueeze_3836 = None
        unsqueeze_3837 = torch.ops.aten.unsqueeze.default(arg749_1, -1);  arg749_1 = None
        unsqueeze_3838 = torch.ops.aten.unsqueeze.default(unsqueeze_3837, -1);  unsqueeze_3837 = None
        mul_1588 = torch.ops.aten.mul.Tensor(mul_1587, unsqueeze_3838);  mul_1587 = unsqueeze_3838 = None
        unsqueeze_3839 = torch.ops.aten.unsqueeze.default(arg750_1, -1);  arg750_1 = None
        unsqueeze_3840 = torch.ops.aten.unsqueeze.default(unsqueeze_3839, -1);  unsqueeze_3839 = None
        add_1368 = torch.ops.aten.add.Tensor(mul_1588, unsqueeze_3840);  mul_1588 = unsqueeze_3840 = None
        relu_423 = torch.ops.aten.relu.default(add_1368);  add_1368 = None
        convolution_475 = torch.ops.aten.convolution.default(relu_423, arg751_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_423 = arg751_1 = None
        add_1369 = torch.ops.aten.add.Tensor(arg753_1, 1e-05);  arg753_1 = None
        sqrt_475 = torch.ops.aten.sqrt.default(add_1369);  add_1369 = None
        reciprocal_475 = torch.ops.aten.reciprocal.default(sqrt_475);  sqrt_475 = None
        mul_1589 = torch.ops.aten.mul.Tensor(reciprocal_475, 1);  reciprocal_475 = None
        unsqueeze_3841 = torch.ops.aten.unsqueeze.default(arg752_1, -1);  arg752_1 = None
        unsqueeze_3842 = torch.ops.aten.unsqueeze.default(unsqueeze_3841, -1);  unsqueeze_3841 = None
        unsqueeze_3843 = torch.ops.aten.unsqueeze.default(mul_1589, -1);  mul_1589 = None
        unsqueeze_3844 = torch.ops.aten.unsqueeze.default(unsqueeze_3843, -1);  unsqueeze_3843 = None
        sub_475 = torch.ops.aten.sub.Tensor(convolution_475, unsqueeze_3842);  convolution_475 = unsqueeze_3842 = None
        mul_1590 = torch.ops.aten.mul.Tensor(sub_475, unsqueeze_3844);  sub_475 = unsqueeze_3844 = None
        unsqueeze_3845 = torch.ops.aten.unsqueeze.default(arg754_1, -1);  arg754_1 = None
        unsqueeze_3846 = torch.ops.aten.unsqueeze.default(unsqueeze_3845, -1);  unsqueeze_3845 = None
        mul_1591 = torch.ops.aten.mul.Tensor(mul_1590, unsqueeze_3846);  mul_1590 = unsqueeze_3846 = None
        unsqueeze_3847 = torch.ops.aten.unsqueeze.default(arg755_1, -1);  arg755_1 = None
        unsqueeze_3848 = torch.ops.aten.unsqueeze.default(unsqueeze_3847, -1);  unsqueeze_3847 = None
        add_1370 = torch.ops.aten.add.Tensor(mul_1591, unsqueeze_3848);  mul_1591 = unsqueeze_3848 = None
        add_1371 = torch.ops.aten.add.Tensor(add_1370, relu_422);  add_1370 = relu_422 = None
        relu_424 = torch.ops.aten.relu.default(add_1371);  add_1371 = None
        convolution_476 = torch.ops.aten.convolution.default(relu_424, arg756_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg756_1 = None
        add_1372 = torch.ops.aten.add.Tensor(arg758_1, 1e-05);  arg758_1 = None
        sqrt_476 = torch.ops.aten.sqrt.default(add_1372);  add_1372 = None
        reciprocal_476 = torch.ops.aten.reciprocal.default(sqrt_476);  sqrt_476 = None
        mul_1592 = torch.ops.aten.mul.Tensor(reciprocal_476, 1);  reciprocal_476 = None
        unsqueeze_3849 = torch.ops.aten.unsqueeze.default(arg757_1, -1);  arg757_1 = None
        unsqueeze_3850 = torch.ops.aten.unsqueeze.default(unsqueeze_3849, -1);  unsqueeze_3849 = None
        unsqueeze_3851 = torch.ops.aten.unsqueeze.default(mul_1592, -1);  mul_1592 = None
        unsqueeze_3852 = torch.ops.aten.unsqueeze.default(unsqueeze_3851, -1);  unsqueeze_3851 = None
        sub_476 = torch.ops.aten.sub.Tensor(convolution_476, unsqueeze_3850);  convolution_476 = unsqueeze_3850 = None
        mul_1593 = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_3852);  sub_476 = unsqueeze_3852 = None
        unsqueeze_3853 = torch.ops.aten.unsqueeze.default(arg759_1, -1);  arg759_1 = None
        unsqueeze_3854 = torch.ops.aten.unsqueeze.default(unsqueeze_3853, -1);  unsqueeze_3853 = None
        mul_1594 = torch.ops.aten.mul.Tensor(mul_1593, unsqueeze_3854);  mul_1593 = unsqueeze_3854 = None
        unsqueeze_3855 = torch.ops.aten.unsqueeze.default(arg760_1, -1);  arg760_1 = None
        unsqueeze_3856 = torch.ops.aten.unsqueeze.default(unsqueeze_3855, -1);  unsqueeze_3855 = None
        add_1373 = torch.ops.aten.add.Tensor(mul_1594, unsqueeze_3856);  mul_1594 = unsqueeze_3856 = None
        relu_425 = torch.ops.aten.relu.default(add_1373);  add_1373 = None
        convolution_477 = torch.ops.aten.convolution.default(relu_425, arg761_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_425 = arg761_1 = None
        add_1374 = torch.ops.aten.add.Tensor(arg763_1, 1e-05);  arg763_1 = None
        sqrt_477 = torch.ops.aten.sqrt.default(add_1374);  add_1374 = None
        reciprocal_477 = torch.ops.aten.reciprocal.default(sqrt_477);  sqrt_477 = None
        mul_1595 = torch.ops.aten.mul.Tensor(reciprocal_477, 1);  reciprocal_477 = None
        unsqueeze_3857 = torch.ops.aten.unsqueeze.default(arg762_1, -1);  arg762_1 = None
        unsqueeze_3858 = torch.ops.aten.unsqueeze.default(unsqueeze_3857, -1);  unsqueeze_3857 = None
        unsqueeze_3859 = torch.ops.aten.unsqueeze.default(mul_1595, -1);  mul_1595 = None
        unsqueeze_3860 = torch.ops.aten.unsqueeze.default(unsqueeze_3859, -1);  unsqueeze_3859 = None
        sub_477 = torch.ops.aten.sub.Tensor(convolution_477, unsqueeze_3858);  convolution_477 = unsqueeze_3858 = None
        mul_1596 = torch.ops.aten.mul.Tensor(sub_477, unsqueeze_3860);  sub_477 = unsqueeze_3860 = None
        unsqueeze_3861 = torch.ops.aten.unsqueeze.default(arg764_1, -1);  arg764_1 = None
        unsqueeze_3862 = torch.ops.aten.unsqueeze.default(unsqueeze_3861, -1);  unsqueeze_3861 = None
        mul_1597 = torch.ops.aten.mul.Tensor(mul_1596, unsqueeze_3862);  mul_1596 = unsqueeze_3862 = None
        unsqueeze_3863 = torch.ops.aten.unsqueeze.default(arg765_1, -1);  arg765_1 = None
        unsqueeze_3864 = torch.ops.aten.unsqueeze.default(unsqueeze_3863, -1);  unsqueeze_3863 = None
        add_1375 = torch.ops.aten.add.Tensor(mul_1597, unsqueeze_3864);  mul_1597 = unsqueeze_3864 = None
        add_1376 = torch.ops.aten.add.Tensor(add_1375, relu_424);  add_1375 = relu_424 = None
        relu_426 = torch.ops.aten.relu.default(add_1376);  add_1376 = None
        convolution_478 = torch.ops.aten.convolution.default(relu_418, arg766_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg766_1 = None
        add_1377 = torch.ops.aten.add.Tensor(arg768_1, 1e-05);  arg768_1 = None
        sqrt_478 = torch.ops.aten.sqrt.default(add_1377);  add_1377 = None
        reciprocal_478 = torch.ops.aten.reciprocal.default(sqrt_478);  sqrt_478 = None
        mul_1598 = torch.ops.aten.mul.Tensor(reciprocal_478, 1);  reciprocal_478 = None
        unsqueeze_3865 = torch.ops.aten.unsqueeze.default(arg767_1, -1);  arg767_1 = None
        unsqueeze_3866 = torch.ops.aten.unsqueeze.default(unsqueeze_3865, -1);  unsqueeze_3865 = None
        unsqueeze_3867 = torch.ops.aten.unsqueeze.default(mul_1598, -1);  mul_1598 = None
        unsqueeze_3868 = torch.ops.aten.unsqueeze.default(unsqueeze_3867, -1);  unsqueeze_3867 = None
        sub_478 = torch.ops.aten.sub.Tensor(convolution_478, unsqueeze_3866);  convolution_478 = unsqueeze_3866 = None
        mul_1599 = torch.ops.aten.mul.Tensor(sub_478, unsqueeze_3868);  sub_478 = unsqueeze_3868 = None
        unsqueeze_3869 = torch.ops.aten.unsqueeze.default(arg769_1, -1);  arg769_1 = None
        unsqueeze_3870 = torch.ops.aten.unsqueeze.default(unsqueeze_3869, -1);  unsqueeze_3869 = None
        mul_1600 = torch.ops.aten.mul.Tensor(mul_1599, unsqueeze_3870);  mul_1599 = unsqueeze_3870 = None
        unsqueeze_3871 = torch.ops.aten.unsqueeze.default(arg770_1, -1);  arg770_1 = None
        unsqueeze_3872 = torch.ops.aten.unsqueeze.default(unsqueeze_3871, -1);  unsqueeze_3871 = None
        add_1378 = torch.ops.aten.add.Tensor(mul_1600, unsqueeze_3872);  mul_1600 = unsqueeze_3872 = None
        iota_82 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1601 = torch.ops.aten.mul.Tensor(iota_82, 1);  iota_82 = None
        add_1379 = torch.ops.aten.add.Tensor(mul_1601, 0);  mul_1601 = None
        convert_element_type_1122 = torch.ops.prims.convert_element_type.default(add_1379, torch.float32);  add_1379 = None
        add_1380 = torch.ops.aten.add.Tensor(convert_element_type_1122, 0.0);  convert_element_type_1122 = None
        mul_1602 = torch.ops.aten.mul.Tensor(add_1380, 0.5);  add_1380 = None
        convert_element_type_1123 = torch.ops.prims.convert_element_type.default(mul_1602, torch.int64);  mul_1602 = None
        unsqueeze_3873 = torch.ops.aten.unsqueeze.default(convert_element_type_1123, -1);  convert_element_type_1123 = None
        iota_83 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1603 = torch.ops.aten.mul.Tensor(iota_83, 1);  iota_83 = None
        add_1381 = torch.ops.aten.add.Tensor(mul_1603, 0);  mul_1603 = None
        convert_element_type_1124 = torch.ops.prims.convert_element_type.default(add_1381, torch.float32);  add_1381 = None
        add_1382 = torch.ops.aten.add.Tensor(convert_element_type_1124, 0.0);  convert_element_type_1124 = None
        mul_1604 = torch.ops.aten.mul.Tensor(add_1382, 0.5);  add_1382 = None
        convert_element_type_1125 = torch.ops.prims.convert_element_type.default(mul_1604, torch.int64);  mul_1604 = None
        _unsafe_index_41 = torch.ops.aten._unsafe_index.Tensor(add_1378, [None, None, unsqueeze_3873, convert_element_type_1125]);  add_1378 = unsqueeze_3873 = convert_element_type_1125 = None
        add_1383 = torch.ops.aten.add.Tensor(relu_410, _unsafe_index_41);  _unsafe_index_41 = None
        convolution_479 = torch.ops.aten.convolution.default(relu_426, arg771_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg771_1 = None
        add_1384 = torch.ops.aten.add.Tensor(arg773_1, 1e-05);  arg773_1 = None
        sqrt_479 = torch.ops.aten.sqrt.default(add_1384);  add_1384 = None
        reciprocal_479 = torch.ops.aten.reciprocal.default(sqrt_479);  sqrt_479 = None
        mul_1605 = torch.ops.aten.mul.Tensor(reciprocal_479, 1);  reciprocal_479 = None
        unsqueeze_3874 = torch.ops.aten.unsqueeze.default(arg772_1, -1);  arg772_1 = None
        unsqueeze_3875 = torch.ops.aten.unsqueeze.default(unsqueeze_3874, -1);  unsqueeze_3874 = None
        unsqueeze_3876 = torch.ops.aten.unsqueeze.default(mul_1605, -1);  mul_1605 = None
        unsqueeze_3877 = torch.ops.aten.unsqueeze.default(unsqueeze_3876, -1);  unsqueeze_3876 = None
        sub_479 = torch.ops.aten.sub.Tensor(convolution_479, unsqueeze_3875);  convolution_479 = unsqueeze_3875 = None
        mul_1606 = torch.ops.aten.mul.Tensor(sub_479, unsqueeze_3877);  sub_479 = unsqueeze_3877 = None
        unsqueeze_3878 = torch.ops.aten.unsqueeze.default(arg774_1, -1);  arg774_1 = None
        unsqueeze_3879 = torch.ops.aten.unsqueeze.default(unsqueeze_3878, -1);  unsqueeze_3878 = None
        mul_1607 = torch.ops.aten.mul.Tensor(mul_1606, unsqueeze_3879);  mul_1606 = unsqueeze_3879 = None
        unsqueeze_3880 = torch.ops.aten.unsqueeze.default(arg775_1, -1);  arg775_1 = None
        unsqueeze_3881 = torch.ops.aten.unsqueeze.default(unsqueeze_3880, -1);  unsqueeze_3880 = None
        add_1385 = torch.ops.aten.add.Tensor(mul_1607, unsqueeze_3881);  mul_1607 = unsqueeze_3881 = None
        iota_84 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1608 = torch.ops.aten.mul.Tensor(iota_84, 1);  iota_84 = None
        add_1386 = torch.ops.aten.add.Tensor(mul_1608, 0);  mul_1608 = None
        convert_element_type_1128 = torch.ops.prims.convert_element_type.default(add_1386, torch.float32);  add_1386 = None
        add_1387 = torch.ops.aten.add.Tensor(convert_element_type_1128, 0.0);  convert_element_type_1128 = None
        mul_1609 = torch.ops.aten.mul.Tensor(add_1387, 0.25);  add_1387 = None
        convert_element_type_1129 = torch.ops.prims.convert_element_type.default(mul_1609, torch.int64);  mul_1609 = None
        unsqueeze_3882 = torch.ops.aten.unsqueeze.default(convert_element_type_1129, -1);  convert_element_type_1129 = None
        iota_85 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1610 = torch.ops.aten.mul.Tensor(iota_85, 1);  iota_85 = None
        add_1388 = torch.ops.aten.add.Tensor(mul_1610, 0);  mul_1610 = None
        convert_element_type_1130 = torch.ops.prims.convert_element_type.default(add_1388, torch.float32);  add_1388 = None
        add_1389 = torch.ops.aten.add.Tensor(convert_element_type_1130, 0.0);  convert_element_type_1130 = None
        mul_1611 = torch.ops.aten.mul.Tensor(add_1389, 0.25);  add_1389 = None
        convert_element_type_1131 = torch.ops.prims.convert_element_type.default(mul_1611, torch.int64);  mul_1611 = None
        _unsafe_index_42 = torch.ops.aten._unsafe_index.Tensor(add_1385, [None, None, unsqueeze_3882, convert_element_type_1131]);  add_1385 = unsqueeze_3882 = convert_element_type_1131 = None
        add_1390 = torch.ops.aten.add.Tensor(add_1383, _unsafe_index_42);  add_1383 = _unsafe_index_42 = None
        relu_427 = torch.ops.aten.relu.default(add_1390);  add_1390 = None
        convolution_480 = torch.ops.aten.convolution.default(relu_410, arg776_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg776_1 = None
        add_1391 = torch.ops.aten.add.Tensor(arg778_1, 1e-05);  arg778_1 = None
        sqrt_480 = torch.ops.aten.sqrt.default(add_1391);  add_1391 = None
        reciprocal_480 = torch.ops.aten.reciprocal.default(sqrt_480);  sqrt_480 = None
        mul_1612 = torch.ops.aten.mul.Tensor(reciprocal_480, 1);  reciprocal_480 = None
        unsqueeze_3883 = torch.ops.aten.unsqueeze.default(arg777_1, -1);  arg777_1 = None
        unsqueeze_3884 = torch.ops.aten.unsqueeze.default(unsqueeze_3883, -1);  unsqueeze_3883 = None
        unsqueeze_3885 = torch.ops.aten.unsqueeze.default(mul_1612, -1);  mul_1612 = None
        unsqueeze_3886 = torch.ops.aten.unsqueeze.default(unsqueeze_3885, -1);  unsqueeze_3885 = None
        sub_480 = torch.ops.aten.sub.Tensor(convolution_480, unsqueeze_3884);  convolution_480 = unsqueeze_3884 = None
        mul_1613 = torch.ops.aten.mul.Tensor(sub_480, unsqueeze_3886);  sub_480 = unsqueeze_3886 = None
        unsqueeze_3887 = torch.ops.aten.unsqueeze.default(arg779_1, -1);  arg779_1 = None
        unsqueeze_3888 = torch.ops.aten.unsqueeze.default(unsqueeze_3887, -1);  unsqueeze_3887 = None
        mul_1614 = torch.ops.aten.mul.Tensor(mul_1613, unsqueeze_3888);  mul_1613 = unsqueeze_3888 = None
        unsqueeze_3889 = torch.ops.aten.unsqueeze.default(arg780_1, -1);  arg780_1 = None
        unsqueeze_3890 = torch.ops.aten.unsqueeze.default(unsqueeze_3889, -1);  unsqueeze_3889 = None
        add_1392 = torch.ops.aten.add.Tensor(mul_1614, unsqueeze_3890);  mul_1614 = unsqueeze_3890 = None
        add_1393 = torch.ops.aten.add.Tensor(add_1392, relu_418);  add_1392 = None
        convolution_481 = torch.ops.aten.convolution.default(relu_426, arg781_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg781_1 = None
        add_1394 = torch.ops.aten.add.Tensor(arg783_1, 1e-05);  arg783_1 = None
        sqrt_481 = torch.ops.aten.sqrt.default(add_1394);  add_1394 = None
        reciprocal_481 = torch.ops.aten.reciprocal.default(sqrt_481);  sqrt_481 = None
        mul_1615 = torch.ops.aten.mul.Tensor(reciprocal_481, 1);  reciprocal_481 = None
        unsqueeze_3891 = torch.ops.aten.unsqueeze.default(arg782_1, -1);  arg782_1 = None
        unsqueeze_3892 = torch.ops.aten.unsqueeze.default(unsqueeze_3891, -1);  unsqueeze_3891 = None
        unsqueeze_3893 = torch.ops.aten.unsqueeze.default(mul_1615, -1);  mul_1615 = None
        unsqueeze_3894 = torch.ops.aten.unsqueeze.default(unsqueeze_3893, -1);  unsqueeze_3893 = None
        sub_481 = torch.ops.aten.sub.Tensor(convolution_481, unsqueeze_3892);  convolution_481 = unsqueeze_3892 = None
        mul_1616 = torch.ops.aten.mul.Tensor(sub_481, unsqueeze_3894);  sub_481 = unsqueeze_3894 = None
        unsqueeze_3895 = torch.ops.aten.unsqueeze.default(arg784_1, -1);  arg784_1 = None
        unsqueeze_3896 = torch.ops.aten.unsqueeze.default(unsqueeze_3895, -1);  unsqueeze_3895 = None
        mul_1617 = torch.ops.aten.mul.Tensor(mul_1616, unsqueeze_3896);  mul_1616 = unsqueeze_3896 = None
        unsqueeze_3897 = torch.ops.aten.unsqueeze.default(arg785_1, -1);  arg785_1 = None
        unsqueeze_3898 = torch.ops.aten.unsqueeze.default(unsqueeze_3897, -1);  unsqueeze_3897 = None
        add_1395 = torch.ops.aten.add.Tensor(mul_1617, unsqueeze_3898);  mul_1617 = unsqueeze_3898 = None
        iota_86 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1618 = torch.ops.aten.mul.Tensor(iota_86, 1);  iota_86 = None
        add_1396 = torch.ops.aten.add.Tensor(mul_1618, 0);  mul_1618 = None
        convert_element_type_1136 = torch.ops.prims.convert_element_type.default(add_1396, torch.float32);  add_1396 = None
        add_1397 = torch.ops.aten.add.Tensor(convert_element_type_1136, 0.0);  convert_element_type_1136 = None
        mul_1619 = torch.ops.aten.mul.Tensor(add_1397, 0.5);  add_1397 = None
        convert_element_type_1137 = torch.ops.prims.convert_element_type.default(mul_1619, torch.int64);  mul_1619 = None
        unsqueeze_3899 = torch.ops.aten.unsqueeze.default(convert_element_type_1137, -1);  convert_element_type_1137 = None
        iota_87 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1620 = torch.ops.aten.mul.Tensor(iota_87, 1);  iota_87 = None
        add_1398 = torch.ops.aten.add.Tensor(mul_1620, 0);  mul_1620 = None
        convert_element_type_1138 = torch.ops.prims.convert_element_type.default(add_1398, torch.float32);  add_1398 = None
        add_1399 = torch.ops.aten.add.Tensor(convert_element_type_1138, 0.0);  convert_element_type_1138 = None
        mul_1621 = torch.ops.aten.mul.Tensor(add_1399, 0.5);  add_1399 = None
        convert_element_type_1139 = torch.ops.prims.convert_element_type.default(mul_1621, torch.int64);  mul_1621 = None
        _unsafe_index_43 = torch.ops.aten._unsafe_index.Tensor(add_1395, [None, None, unsqueeze_3899, convert_element_type_1139]);  add_1395 = unsqueeze_3899 = convert_element_type_1139 = None
        add_1400 = torch.ops.aten.add.Tensor(add_1393, _unsafe_index_43);  add_1393 = _unsafe_index_43 = None
        relu_428 = torch.ops.aten.relu.default(add_1400);  add_1400 = None
        convolution_482 = torch.ops.aten.convolution.default(relu_410, arg786_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_410 = arg786_1 = None
        add_1401 = torch.ops.aten.add.Tensor(arg788_1, 1e-05);  arg788_1 = None
        sqrt_482 = torch.ops.aten.sqrt.default(add_1401);  add_1401 = None
        reciprocal_482 = torch.ops.aten.reciprocal.default(sqrt_482);  sqrt_482 = None
        mul_1622 = torch.ops.aten.mul.Tensor(reciprocal_482, 1);  reciprocal_482 = None
        unsqueeze_3900 = torch.ops.aten.unsqueeze.default(arg787_1, -1);  arg787_1 = None
        unsqueeze_3901 = torch.ops.aten.unsqueeze.default(unsqueeze_3900, -1);  unsqueeze_3900 = None
        unsqueeze_3902 = torch.ops.aten.unsqueeze.default(mul_1622, -1);  mul_1622 = None
        unsqueeze_3903 = torch.ops.aten.unsqueeze.default(unsqueeze_3902, -1);  unsqueeze_3902 = None
        sub_482 = torch.ops.aten.sub.Tensor(convolution_482, unsqueeze_3901);  convolution_482 = unsqueeze_3901 = None
        mul_1623 = torch.ops.aten.mul.Tensor(sub_482, unsqueeze_3903);  sub_482 = unsqueeze_3903 = None
        unsqueeze_3904 = torch.ops.aten.unsqueeze.default(arg789_1, -1);  arg789_1 = None
        unsqueeze_3905 = torch.ops.aten.unsqueeze.default(unsqueeze_3904, -1);  unsqueeze_3904 = None
        mul_1624 = torch.ops.aten.mul.Tensor(mul_1623, unsqueeze_3905);  mul_1623 = unsqueeze_3905 = None
        unsqueeze_3906 = torch.ops.aten.unsqueeze.default(arg790_1, -1);  arg790_1 = None
        unsqueeze_3907 = torch.ops.aten.unsqueeze.default(unsqueeze_3906, -1);  unsqueeze_3906 = None
        add_1402 = torch.ops.aten.add.Tensor(mul_1624, unsqueeze_3907);  mul_1624 = unsqueeze_3907 = None
        relu_429 = torch.ops.aten.relu.default(add_1402);  add_1402 = None
        convolution_483 = torch.ops.aten.convolution.default(relu_429, arg791_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_429 = arg791_1 = None
        add_1403 = torch.ops.aten.add.Tensor(arg793_1, 1e-05);  arg793_1 = None
        sqrt_483 = torch.ops.aten.sqrt.default(add_1403);  add_1403 = None
        reciprocal_483 = torch.ops.aten.reciprocal.default(sqrt_483);  sqrt_483 = None
        mul_1625 = torch.ops.aten.mul.Tensor(reciprocal_483, 1);  reciprocal_483 = None
        unsqueeze_3908 = torch.ops.aten.unsqueeze.default(arg792_1, -1);  arg792_1 = None
        unsqueeze_3909 = torch.ops.aten.unsqueeze.default(unsqueeze_3908, -1);  unsqueeze_3908 = None
        unsqueeze_3910 = torch.ops.aten.unsqueeze.default(mul_1625, -1);  mul_1625 = None
        unsqueeze_3911 = torch.ops.aten.unsqueeze.default(unsqueeze_3910, -1);  unsqueeze_3910 = None
        sub_483 = torch.ops.aten.sub.Tensor(convolution_483, unsqueeze_3909);  convolution_483 = unsqueeze_3909 = None
        mul_1626 = torch.ops.aten.mul.Tensor(sub_483, unsqueeze_3911);  sub_483 = unsqueeze_3911 = None
        unsqueeze_3912 = torch.ops.aten.unsqueeze.default(arg794_1, -1);  arg794_1 = None
        unsqueeze_3913 = torch.ops.aten.unsqueeze.default(unsqueeze_3912, -1);  unsqueeze_3912 = None
        mul_1627 = torch.ops.aten.mul.Tensor(mul_1626, unsqueeze_3913);  mul_1626 = unsqueeze_3913 = None
        unsqueeze_3914 = torch.ops.aten.unsqueeze.default(arg795_1, -1);  arg795_1 = None
        unsqueeze_3915 = torch.ops.aten.unsqueeze.default(unsqueeze_3914, -1);  unsqueeze_3914 = None
        add_1404 = torch.ops.aten.add.Tensor(mul_1627, unsqueeze_3915);  mul_1627 = unsqueeze_3915 = None
        convolution_484 = torch.ops.aten.convolution.default(relu_418, arg796_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_418 = arg796_1 = None
        add_1405 = torch.ops.aten.add.Tensor(arg798_1, 1e-05);  arg798_1 = None
        sqrt_484 = torch.ops.aten.sqrt.default(add_1405);  add_1405 = None
        reciprocal_484 = torch.ops.aten.reciprocal.default(sqrt_484);  sqrt_484 = None
        mul_1628 = torch.ops.aten.mul.Tensor(reciprocal_484, 1);  reciprocal_484 = None
        unsqueeze_3916 = torch.ops.aten.unsqueeze.default(arg797_1, -1);  arg797_1 = None
        unsqueeze_3917 = torch.ops.aten.unsqueeze.default(unsqueeze_3916, -1);  unsqueeze_3916 = None
        unsqueeze_3918 = torch.ops.aten.unsqueeze.default(mul_1628, -1);  mul_1628 = None
        unsqueeze_3919 = torch.ops.aten.unsqueeze.default(unsqueeze_3918, -1);  unsqueeze_3918 = None
        sub_484 = torch.ops.aten.sub.Tensor(convolution_484, unsqueeze_3917);  convolution_484 = unsqueeze_3917 = None
        mul_1629 = torch.ops.aten.mul.Tensor(sub_484, unsqueeze_3919);  sub_484 = unsqueeze_3919 = None
        unsqueeze_3920 = torch.ops.aten.unsqueeze.default(arg799_1, -1);  arg799_1 = None
        unsqueeze_3921 = torch.ops.aten.unsqueeze.default(unsqueeze_3920, -1);  unsqueeze_3920 = None
        mul_1630 = torch.ops.aten.mul.Tensor(mul_1629, unsqueeze_3921);  mul_1629 = unsqueeze_3921 = None
        unsqueeze_3922 = torch.ops.aten.unsqueeze.default(arg800_1, -1);  arg800_1 = None
        unsqueeze_3923 = torch.ops.aten.unsqueeze.default(unsqueeze_3922, -1);  unsqueeze_3922 = None
        add_1406 = torch.ops.aten.add.Tensor(mul_1630, unsqueeze_3923);  mul_1630 = unsqueeze_3923 = None
        add_1407 = torch.ops.aten.add.Tensor(add_1404, add_1406);  add_1404 = add_1406 = None
        add_1408 = torch.ops.aten.add.Tensor(add_1407, relu_426);  add_1407 = relu_426 = None
        relu_430 = torch.ops.aten.relu.default(add_1408);  add_1408 = None
        convolution_485 = torch.ops.aten.convolution.default(relu_430, arg801_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg801_1 = None
        add_1409 = torch.ops.aten.add.Tensor(arg803_1, 1e-05);  arg803_1 = None
        sqrt_485 = torch.ops.aten.sqrt.default(add_1409);  add_1409 = None
        reciprocal_485 = torch.ops.aten.reciprocal.default(sqrt_485);  sqrt_485 = None
        mul_1631 = torch.ops.aten.mul.Tensor(reciprocal_485, 1);  reciprocal_485 = None
        unsqueeze_3924 = torch.ops.aten.unsqueeze.default(arg802_1, -1);  arg802_1 = None
        unsqueeze_3925 = torch.ops.aten.unsqueeze.default(unsqueeze_3924, -1);  unsqueeze_3924 = None
        unsqueeze_3926 = torch.ops.aten.unsqueeze.default(mul_1631, -1);  mul_1631 = None
        unsqueeze_3927 = torch.ops.aten.unsqueeze.default(unsqueeze_3926, -1);  unsqueeze_3926 = None
        sub_485 = torch.ops.aten.sub.Tensor(convolution_485, unsqueeze_3925);  convolution_485 = unsqueeze_3925 = None
        mul_1632 = torch.ops.aten.mul.Tensor(sub_485, unsqueeze_3927);  sub_485 = unsqueeze_3927 = None
        unsqueeze_3928 = torch.ops.aten.unsqueeze.default(arg804_1, -1);  arg804_1 = None
        unsqueeze_3929 = torch.ops.aten.unsqueeze.default(unsqueeze_3928, -1);  unsqueeze_3928 = None
        mul_1633 = torch.ops.aten.mul.Tensor(mul_1632, unsqueeze_3929);  mul_1632 = unsqueeze_3929 = None
        unsqueeze_3930 = torch.ops.aten.unsqueeze.default(arg805_1, -1);  arg805_1 = None
        unsqueeze_3931 = torch.ops.aten.unsqueeze.default(unsqueeze_3930, -1);  unsqueeze_3930 = None
        add_1410 = torch.ops.aten.add.Tensor(mul_1633, unsqueeze_3931);  mul_1633 = unsqueeze_3931 = None
        relu_431 = torch.ops.aten.relu.default(add_1410);  add_1410 = None
        convolution_486 = torch.ops.aten.convolution.default(relu_427, arg806_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg806_1 = None
        add_1411 = torch.ops.aten.add.Tensor(arg808_1, 1e-05);  arg808_1 = None
        sqrt_486 = torch.ops.aten.sqrt.default(add_1411);  add_1411 = None
        reciprocal_486 = torch.ops.aten.reciprocal.default(sqrt_486);  sqrt_486 = None
        mul_1634 = torch.ops.aten.mul.Tensor(reciprocal_486, 1);  reciprocal_486 = None
        unsqueeze_3932 = torch.ops.aten.unsqueeze.default(arg807_1, -1);  arg807_1 = None
        unsqueeze_3933 = torch.ops.aten.unsqueeze.default(unsqueeze_3932, -1);  unsqueeze_3932 = None
        unsqueeze_3934 = torch.ops.aten.unsqueeze.default(mul_1634, -1);  mul_1634 = None
        unsqueeze_3935 = torch.ops.aten.unsqueeze.default(unsqueeze_3934, -1);  unsqueeze_3934 = None
        sub_486 = torch.ops.aten.sub.Tensor(convolution_486, unsqueeze_3933);  convolution_486 = unsqueeze_3933 = None
        mul_1635 = torch.ops.aten.mul.Tensor(sub_486, unsqueeze_3935);  sub_486 = unsqueeze_3935 = None
        unsqueeze_3936 = torch.ops.aten.unsqueeze.default(arg809_1, -1);  arg809_1 = None
        unsqueeze_3937 = torch.ops.aten.unsqueeze.default(unsqueeze_3936, -1);  unsqueeze_3936 = None
        mul_1636 = torch.ops.aten.mul.Tensor(mul_1635, unsqueeze_3937);  mul_1635 = unsqueeze_3937 = None
        unsqueeze_3938 = torch.ops.aten.unsqueeze.default(arg810_1, -1);  arg810_1 = None
        unsqueeze_3939 = torch.ops.aten.unsqueeze.default(unsqueeze_3938, -1);  unsqueeze_3938 = None
        add_1412 = torch.ops.aten.add.Tensor(mul_1636, unsqueeze_3939);  mul_1636 = unsqueeze_3939 = None
        relu_432 = torch.ops.aten.relu.default(add_1412);  add_1412 = None
        convolution_487 = torch.ops.aten.convolution.default(relu_432, arg811_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_432 = arg811_1 = None
        add_1413 = torch.ops.aten.add.Tensor(arg813_1, 1e-05);  arg813_1 = None
        sqrt_487 = torch.ops.aten.sqrt.default(add_1413);  add_1413 = None
        reciprocal_487 = torch.ops.aten.reciprocal.default(sqrt_487);  sqrt_487 = None
        mul_1637 = torch.ops.aten.mul.Tensor(reciprocal_487, 1);  reciprocal_487 = None
        unsqueeze_3940 = torch.ops.aten.unsqueeze.default(arg812_1, -1);  arg812_1 = None
        unsqueeze_3941 = torch.ops.aten.unsqueeze.default(unsqueeze_3940, -1);  unsqueeze_3940 = None
        unsqueeze_3942 = torch.ops.aten.unsqueeze.default(mul_1637, -1);  mul_1637 = None
        unsqueeze_3943 = torch.ops.aten.unsqueeze.default(unsqueeze_3942, -1);  unsqueeze_3942 = None
        sub_487 = torch.ops.aten.sub.Tensor(convolution_487, unsqueeze_3941);  convolution_487 = unsqueeze_3941 = None
        mul_1638 = torch.ops.aten.mul.Tensor(sub_487, unsqueeze_3943);  sub_487 = unsqueeze_3943 = None
        unsqueeze_3944 = torch.ops.aten.unsqueeze.default(arg814_1, -1);  arg814_1 = None
        unsqueeze_3945 = torch.ops.aten.unsqueeze.default(unsqueeze_3944, -1);  unsqueeze_3944 = None
        mul_1639 = torch.ops.aten.mul.Tensor(mul_1638, unsqueeze_3945);  mul_1638 = unsqueeze_3945 = None
        unsqueeze_3946 = torch.ops.aten.unsqueeze.default(arg815_1, -1);  arg815_1 = None
        unsqueeze_3947 = torch.ops.aten.unsqueeze.default(unsqueeze_3946, -1);  unsqueeze_3946 = None
        add_1414 = torch.ops.aten.add.Tensor(mul_1639, unsqueeze_3947);  mul_1639 = unsqueeze_3947 = None
        add_1415 = torch.ops.aten.add.Tensor(add_1414, relu_427);  add_1414 = relu_427 = None
        relu_433 = torch.ops.aten.relu.default(add_1415);  add_1415 = None
        convolution_488 = torch.ops.aten.convolution.default(relu_433, arg816_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg816_1 = None
        add_1416 = torch.ops.aten.add.Tensor(arg818_1, 1e-05);  arg818_1 = None
        sqrt_488 = torch.ops.aten.sqrt.default(add_1416);  add_1416 = None
        reciprocal_488 = torch.ops.aten.reciprocal.default(sqrt_488);  sqrt_488 = None
        mul_1640 = torch.ops.aten.mul.Tensor(reciprocal_488, 1);  reciprocal_488 = None
        unsqueeze_3948 = torch.ops.aten.unsqueeze.default(arg817_1, -1);  arg817_1 = None
        unsqueeze_3949 = torch.ops.aten.unsqueeze.default(unsqueeze_3948, -1);  unsqueeze_3948 = None
        unsqueeze_3950 = torch.ops.aten.unsqueeze.default(mul_1640, -1);  mul_1640 = None
        unsqueeze_3951 = torch.ops.aten.unsqueeze.default(unsqueeze_3950, -1);  unsqueeze_3950 = None
        sub_488 = torch.ops.aten.sub.Tensor(convolution_488, unsqueeze_3949);  convolution_488 = unsqueeze_3949 = None
        mul_1641 = torch.ops.aten.mul.Tensor(sub_488, unsqueeze_3951);  sub_488 = unsqueeze_3951 = None
        unsqueeze_3952 = torch.ops.aten.unsqueeze.default(arg819_1, -1);  arg819_1 = None
        unsqueeze_3953 = torch.ops.aten.unsqueeze.default(unsqueeze_3952, -1);  unsqueeze_3952 = None
        mul_1642 = torch.ops.aten.mul.Tensor(mul_1641, unsqueeze_3953);  mul_1641 = unsqueeze_3953 = None
        unsqueeze_3954 = torch.ops.aten.unsqueeze.default(arg820_1, -1);  arg820_1 = None
        unsqueeze_3955 = torch.ops.aten.unsqueeze.default(unsqueeze_3954, -1);  unsqueeze_3954 = None
        add_1417 = torch.ops.aten.add.Tensor(mul_1642, unsqueeze_3955);  mul_1642 = unsqueeze_3955 = None
        relu_434 = torch.ops.aten.relu.default(add_1417);  add_1417 = None
        convolution_489 = torch.ops.aten.convolution.default(relu_434, arg821_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_434 = arg821_1 = None
        add_1418 = torch.ops.aten.add.Tensor(arg823_1, 1e-05);  arg823_1 = None
        sqrt_489 = torch.ops.aten.sqrt.default(add_1418);  add_1418 = None
        reciprocal_489 = torch.ops.aten.reciprocal.default(sqrt_489);  sqrt_489 = None
        mul_1643 = torch.ops.aten.mul.Tensor(reciprocal_489, 1);  reciprocal_489 = None
        unsqueeze_3956 = torch.ops.aten.unsqueeze.default(arg822_1, -1);  arg822_1 = None
        unsqueeze_3957 = torch.ops.aten.unsqueeze.default(unsqueeze_3956, -1);  unsqueeze_3956 = None
        unsqueeze_3958 = torch.ops.aten.unsqueeze.default(mul_1643, -1);  mul_1643 = None
        unsqueeze_3959 = torch.ops.aten.unsqueeze.default(unsqueeze_3958, -1);  unsqueeze_3958 = None
        sub_489 = torch.ops.aten.sub.Tensor(convolution_489, unsqueeze_3957);  convolution_489 = unsqueeze_3957 = None
        mul_1644 = torch.ops.aten.mul.Tensor(sub_489, unsqueeze_3959);  sub_489 = unsqueeze_3959 = None
        unsqueeze_3960 = torch.ops.aten.unsqueeze.default(arg824_1, -1);  arg824_1 = None
        unsqueeze_3961 = torch.ops.aten.unsqueeze.default(unsqueeze_3960, -1);  unsqueeze_3960 = None
        mul_1645 = torch.ops.aten.mul.Tensor(mul_1644, unsqueeze_3961);  mul_1644 = unsqueeze_3961 = None
        unsqueeze_3962 = torch.ops.aten.unsqueeze.default(arg825_1, -1);  arg825_1 = None
        unsqueeze_3963 = torch.ops.aten.unsqueeze.default(unsqueeze_3962, -1);  unsqueeze_3962 = None
        add_1419 = torch.ops.aten.add.Tensor(mul_1645, unsqueeze_3963);  mul_1645 = unsqueeze_3963 = None
        add_1420 = torch.ops.aten.add.Tensor(add_1419, relu_433);  add_1419 = relu_433 = None
        relu_435 = torch.ops.aten.relu.default(add_1420);  add_1420 = None
        convolution_490 = torch.ops.aten.convolution.default(relu_435, arg826_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg826_1 = None
        add_1421 = torch.ops.aten.add.Tensor(arg828_1, 1e-05);  arg828_1 = None
        sqrt_490 = torch.ops.aten.sqrt.default(add_1421);  add_1421 = None
        reciprocal_490 = torch.ops.aten.reciprocal.default(sqrt_490);  sqrt_490 = None
        mul_1646 = torch.ops.aten.mul.Tensor(reciprocal_490, 1);  reciprocal_490 = None
        unsqueeze_3964 = torch.ops.aten.unsqueeze.default(arg827_1, -1);  arg827_1 = None
        unsqueeze_3965 = torch.ops.aten.unsqueeze.default(unsqueeze_3964, -1);  unsqueeze_3964 = None
        unsqueeze_3966 = torch.ops.aten.unsqueeze.default(mul_1646, -1);  mul_1646 = None
        unsqueeze_3967 = torch.ops.aten.unsqueeze.default(unsqueeze_3966, -1);  unsqueeze_3966 = None
        sub_490 = torch.ops.aten.sub.Tensor(convolution_490, unsqueeze_3965);  convolution_490 = unsqueeze_3965 = None
        mul_1647 = torch.ops.aten.mul.Tensor(sub_490, unsqueeze_3967);  sub_490 = unsqueeze_3967 = None
        unsqueeze_3968 = torch.ops.aten.unsqueeze.default(arg829_1, -1);  arg829_1 = None
        unsqueeze_3969 = torch.ops.aten.unsqueeze.default(unsqueeze_3968, -1);  unsqueeze_3968 = None
        mul_1648 = torch.ops.aten.mul.Tensor(mul_1647, unsqueeze_3969);  mul_1647 = unsqueeze_3969 = None
        unsqueeze_3970 = torch.ops.aten.unsqueeze.default(arg830_1, -1);  arg830_1 = None
        unsqueeze_3971 = torch.ops.aten.unsqueeze.default(unsqueeze_3970, -1);  unsqueeze_3970 = None
        add_1422 = torch.ops.aten.add.Tensor(mul_1648, unsqueeze_3971);  mul_1648 = unsqueeze_3971 = None
        relu_436 = torch.ops.aten.relu.default(add_1422);  add_1422 = None
        convolution_491 = torch.ops.aten.convolution.default(relu_436, arg831_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_436 = arg831_1 = None
        add_1423 = torch.ops.aten.add.Tensor(arg833_1, 1e-05);  arg833_1 = None
        sqrt_491 = torch.ops.aten.sqrt.default(add_1423);  add_1423 = None
        reciprocal_491 = torch.ops.aten.reciprocal.default(sqrt_491);  sqrt_491 = None
        mul_1649 = torch.ops.aten.mul.Tensor(reciprocal_491, 1);  reciprocal_491 = None
        unsqueeze_3972 = torch.ops.aten.unsqueeze.default(arg832_1, -1);  arg832_1 = None
        unsqueeze_3973 = torch.ops.aten.unsqueeze.default(unsqueeze_3972, -1);  unsqueeze_3972 = None
        unsqueeze_3974 = torch.ops.aten.unsqueeze.default(mul_1649, -1);  mul_1649 = None
        unsqueeze_3975 = torch.ops.aten.unsqueeze.default(unsqueeze_3974, -1);  unsqueeze_3974 = None
        sub_491 = torch.ops.aten.sub.Tensor(convolution_491, unsqueeze_3973);  convolution_491 = unsqueeze_3973 = None
        mul_1650 = torch.ops.aten.mul.Tensor(sub_491, unsqueeze_3975);  sub_491 = unsqueeze_3975 = None
        unsqueeze_3976 = torch.ops.aten.unsqueeze.default(arg834_1, -1);  arg834_1 = None
        unsqueeze_3977 = torch.ops.aten.unsqueeze.default(unsqueeze_3976, -1);  unsqueeze_3976 = None
        mul_1651 = torch.ops.aten.mul.Tensor(mul_1650, unsqueeze_3977);  mul_1650 = unsqueeze_3977 = None
        unsqueeze_3978 = torch.ops.aten.unsqueeze.default(arg835_1, -1);  arg835_1 = None
        unsqueeze_3979 = torch.ops.aten.unsqueeze.default(unsqueeze_3978, -1);  unsqueeze_3978 = None
        add_1424 = torch.ops.aten.add.Tensor(mul_1651, unsqueeze_3979);  mul_1651 = unsqueeze_3979 = None
        add_1425 = torch.ops.aten.add.Tensor(add_1424, relu_435);  add_1424 = relu_435 = None
        relu_437 = torch.ops.aten.relu.default(add_1425);  add_1425 = None
        convolution_492 = torch.ops.aten.convolution.default(relu_437, arg836_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg836_1 = None
        add_1426 = torch.ops.aten.add.Tensor(arg838_1, 1e-05);  arg838_1 = None
        sqrt_492 = torch.ops.aten.sqrt.default(add_1426);  add_1426 = None
        reciprocal_492 = torch.ops.aten.reciprocal.default(sqrt_492);  sqrt_492 = None
        mul_1652 = torch.ops.aten.mul.Tensor(reciprocal_492, 1);  reciprocal_492 = None
        unsqueeze_3980 = torch.ops.aten.unsqueeze.default(arg837_1, -1);  arg837_1 = None
        unsqueeze_3981 = torch.ops.aten.unsqueeze.default(unsqueeze_3980, -1);  unsqueeze_3980 = None
        unsqueeze_3982 = torch.ops.aten.unsqueeze.default(mul_1652, -1);  mul_1652 = None
        unsqueeze_3983 = torch.ops.aten.unsqueeze.default(unsqueeze_3982, -1);  unsqueeze_3982 = None
        sub_492 = torch.ops.aten.sub.Tensor(convolution_492, unsqueeze_3981);  convolution_492 = unsqueeze_3981 = None
        mul_1653 = torch.ops.aten.mul.Tensor(sub_492, unsqueeze_3983);  sub_492 = unsqueeze_3983 = None
        unsqueeze_3984 = torch.ops.aten.unsqueeze.default(arg839_1, -1);  arg839_1 = None
        unsqueeze_3985 = torch.ops.aten.unsqueeze.default(unsqueeze_3984, -1);  unsqueeze_3984 = None
        mul_1654 = torch.ops.aten.mul.Tensor(mul_1653, unsqueeze_3985);  mul_1653 = unsqueeze_3985 = None
        unsqueeze_3986 = torch.ops.aten.unsqueeze.default(arg840_1, -1);  arg840_1 = None
        unsqueeze_3987 = torch.ops.aten.unsqueeze.default(unsqueeze_3986, -1);  unsqueeze_3986 = None
        add_1427 = torch.ops.aten.add.Tensor(mul_1654, unsqueeze_3987);  mul_1654 = unsqueeze_3987 = None
        relu_438 = torch.ops.aten.relu.default(add_1427);  add_1427 = None
        convolution_493 = torch.ops.aten.convolution.default(relu_438, arg841_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_438 = arg841_1 = None
        add_1428 = torch.ops.aten.add.Tensor(arg843_1, 1e-05);  arg843_1 = None
        sqrt_493 = torch.ops.aten.sqrt.default(add_1428);  add_1428 = None
        reciprocal_493 = torch.ops.aten.reciprocal.default(sqrt_493);  sqrt_493 = None
        mul_1655 = torch.ops.aten.mul.Tensor(reciprocal_493, 1);  reciprocal_493 = None
        unsqueeze_3988 = torch.ops.aten.unsqueeze.default(arg842_1, -1);  arg842_1 = None
        unsqueeze_3989 = torch.ops.aten.unsqueeze.default(unsqueeze_3988, -1);  unsqueeze_3988 = None
        unsqueeze_3990 = torch.ops.aten.unsqueeze.default(mul_1655, -1);  mul_1655 = None
        unsqueeze_3991 = torch.ops.aten.unsqueeze.default(unsqueeze_3990, -1);  unsqueeze_3990 = None
        sub_493 = torch.ops.aten.sub.Tensor(convolution_493, unsqueeze_3989);  convolution_493 = unsqueeze_3989 = None
        mul_1656 = torch.ops.aten.mul.Tensor(sub_493, unsqueeze_3991);  sub_493 = unsqueeze_3991 = None
        unsqueeze_3992 = torch.ops.aten.unsqueeze.default(arg844_1, -1);  arg844_1 = None
        unsqueeze_3993 = torch.ops.aten.unsqueeze.default(unsqueeze_3992, -1);  unsqueeze_3992 = None
        mul_1657 = torch.ops.aten.mul.Tensor(mul_1656, unsqueeze_3993);  mul_1656 = unsqueeze_3993 = None
        unsqueeze_3994 = torch.ops.aten.unsqueeze.default(arg845_1, -1);  arg845_1 = None
        unsqueeze_3995 = torch.ops.aten.unsqueeze.default(unsqueeze_3994, -1);  unsqueeze_3994 = None
        add_1429 = torch.ops.aten.add.Tensor(mul_1657, unsqueeze_3995);  mul_1657 = unsqueeze_3995 = None
        add_1430 = torch.ops.aten.add.Tensor(add_1429, relu_437);  add_1429 = relu_437 = None
        relu_439 = torch.ops.aten.relu.default(add_1430);  add_1430 = None
        convolution_494 = torch.ops.aten.convolution.default(relu_428, arg846_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg846_1 = None
        add_1431 = torch.ops.aten.add.Tensor(arg848_1, 1e-05);  arg848_1 = None
        sqrt_494 = torch.ops.aten.sqrt.default(add_1431);  add_1431 = None
        reciprocal_494 = torch.ops.aten.reciprocal.default(sqrt_494);  sqrt_494 = None
        mul_1658 = torch.ops.aten.mul.Tensor(reciprocal_494, 1);  reciprocal_494 = None
        unsqueeze_3996 = torch.ops.aten.unsqueeze.default(arg847_1, -1);  arg847_1 = None
        unsqueeze_3997 = torch.ops.aten.unsqueeze.default(unsqueeze_3996, -1);  unsqueeze_3996 = None
        unsqueeze_3998 = torch.ops.aten.unsqueeze.default(mul_1658, -1);  mul_1658 = None
        unsqueeze_3999 = torch.ops.aten.unsqueeze.default(unsqueeze_3998, -1);  unsqueeze_3998 = None
        sub_494 = torch.ops.aten.sub.Tensor(convolution_494, unsqueeze_3997);  convolution_494 = unsqueeze_3997 = None
        mul_1659 = torch.ops.aten.mul.Tensor(sub_494, unsqueeze_3999);  sub_494 = unsqueeze_3999 = None
        unsqueeze_4000 = torch.ops.aten.unsqueeze.default(arg849_1, -1);  arg849_1 = None
        unsqueeze_4001 = torch.ops.aten.unsqueeze.default(unsqueeze_4000, -1);  unsqueeze_4000 = None
        mul_1660 = torch.ops.aten.mul.Tensor(mul_1659, unsqueeze_4001);  mul_1659 = unsqueeze_4001 = None
        unsqueeze_4002 = torch.ops.aten.unsqueeze.default(arg850_1, -1);  arg850_1 = None
        unsqueeze_4003 = torch.ops.aten.unsqueeze.default(unsqueeze_4002, -1);  unsqueeze_4002 = None
        add_1432 = torch.ops.aten.add.Tensor(mul_1660, unsqueeze_4003);  mul_1660 = unsqueeze_4003 = None
        relu_440 = torch.ops.aten.relu.default(add_1432);  add_1432 = None
        convolution_495 = torch.ops.aten.convolution.default(relu_440, arg851_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_440 = arg851_1 = None
        add_1433 = torch.ops.aten.add.Tensor(arg853_1, 1e-05);  arg853_1 = None
        sqrt_495 = torch.ops.aten.sqrt.default(add_1433);  add_1433 = None
        reciprocal_495 = torch.ops.aten.reciprocal.default(sqrt_495);  sqrt_495 = None
        mul_1661 = torch.ops.aten.mul.Tensor(reciprocal_495, 1);  reciprocal_495 = None
        unsqueeze_4004 = torch.ops.aten.unsqueeze.default(arg852_1, -1);  arg852_1 = None
        unsqueeze_4005 = torch.ops.aten.unsqueeze.default(unsqueeze_4004, -1);  unsqueeze_4004 = None
        unsqueeze_4006 = torch.ops.aten.unsqueeze.default(mul_1661, -1);  mul_1661 = None
        unsqueeze_4007 = torch.ops.aten.unsqueeze.default(unsqueeze_4006, -1);  unsqueeze_4006 = None
        sub_495 = torch.ops.aten.sub.Tensor(convolution_495, unsqueeze_4005);  convolution_495 = unsqueeze_4005 = None
        mul_1662 = torch.ops.aten.mul.Tensor(sub_495, unsqueeze_4007);  sub_495 = unsqueeze_4007 = None
        unsqueeze_4008 = torch.ops.aten.unsqueeze.default(arg854_1, -1);  arg854_1 = None
        unsqueeze_4009 = torch.ops.aten.unsqueeze.default(unsqueeze_4008, -1);  unsqueeze_4008 = None
        mul_1663 = torch.ops.aten.mul.Tensor(mul_1662, unsqueeze_4009);  mul_1662 = unsqueeze_4009 = None
        unsqueeze_4010 = torch.ops.aten.unsqueeze.default(arg855_1, -1);  arg855_1 = None
        unsqueeze_4011 = torch.ops.aten.unsqueeze.default(unsqueeze_4010, -1);  unsqueeze_4010 = None
        add_1434 = torch.ops.aten.add.Tensor(mul_1663, unsqueeze_4011);  mul_1663 = unsqueeze_4011 = None
        add_1435 = torch.ops.aten.add.Tensor(add_1434, relu_428);  add_1434 = relu_428 = None
        relu_441 = torch.ops.aten.relu.default(add_1435);  add_1435 = None
        convolution_496 = torch.ops.aten.convolution.default(relu_441, arg856_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg856_1 = None
        add_1436 = torch.ops.aten.add.Tensor(arg858_1, 1e-05);  arg858_1 = None
        sqrt_496 = torch.ops.aten.sqrt.default(add_1436);  add_1436 = None
        reciprocal_496 = torch.ops.aten.reciprocal.default(sqrt_496);  sqrt_496 = None
        mul_1664 = torch.ops.aten.mul.Tensor(reciprocal_496, 1);  reciprocal_496 = None
        unsqueeze_4012 = torch.ops.aten.unsqueeze.default(arg857_1, -1);  arg857_1 = None
        unsqueeze_4013 = torch.ops.aten.unsqueeze.default(unsqueeze_4012, -1);  unsqueeze_4012 = None
        unsqueeze_4014 = torch.ops.aten.unsqueeze.default(mul_1664, -1);  mul_1664 = None
        unsqueeze_4015 = torch.ops.aten.unsqueeze.default(unsqueeze_4014, -1);  unsqueeze_4014 = None
        sub_496 = torch.ops.aten.sub.Tensor(convolution_496, unsqueeze_4013);  convolution_496 = unsqueeze_4013 = None
        mul_1665 = torch.ops.aten.mul.Tensor(sub_496, unsqueeze_4015);  sub_496 = unsqueeze_4015 = None
        unsqueeze_4016 = torch.ops.aten.unsqueeze.default(arg859_1, -1);  arg859_1 = None
        unsqueeze_4017 = torch.ops.aten.unsqueeze.default(unsqueeze_4016, -1);  unsqueeze_4016 = None
        mul_1666 = torch.ops.aten.mul.Tensor(mul_1665, unsqueeze_4017);  mul_1665 = unsqueeze_4017 = None
        unsqueeze_4018 = torch.ops.aten.unsqueeze.default(arg860_1, -1);  arg860_1 = None
        unsqueeze_4019 = torch.ops.aten.unsqueeze.default(unsqueeze_4018, -1);  unsqueeze_4018 = None
        add_1437 = torch.ops.aten.add.Tensor(mul_1666, unsqueeze_4019);  mul_1666 = unsqueeze_4019 = None
        relu_442 = torch.ops.aten.relu.default(add_1437);  add_1437 = None
        convolution_497 = torch.ops.aten.convolution.default(relu_442, arg861_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_442 = arg861_1 = None
        add_1438 = torch.ops.aten.add.Tensor(arg863_1, 1e-05);  arg863_1 = None
        sqrt_497 = torch.ops.aten.sqrt.default(add_1438);  add_1438 = None
        reciprocal_497 = torch.ops.aten.reciprocal.default(sqrt_497);  sqrt_497 = None
        mul_1667 = torch.ops.aten.mul.Tensor(reciprocal_497, 1);  reciprocal_497 = None
        unsqueeze_4020 = torch.ops.aten.unsqueeze.default(arg862_1, -1);  arg862_1 = None
        unsqueeze_4021 = torch.ops.aten.unsqueeze.default(unsqueeze_4020, -1);  unsqueeze_4020 = None
        unsqueeze_4022 = torch.ops.aten.unsqueeze.default(mul_1667, -1);  mul_1667 = None
        unsqueeze_4023 = torch.ops.aten.unsqueeze.default(unsqueeze_4022, -1);  unsqueeze_4022 = None
        sub_497 = torch.ops.aten.sub.Tensor(convolution_497, unsqueeze_4021);  convolution_497 = unsqueeze_4021 = None
        mul_1668 = torch.ops.aten.mul.Tensor(sub_497, unsqueeze_4023);  sub_497 = unsqueeze_4023 = None
        unsqueeze_4024 = torch.ops.aten.unsqueeze.default(arg864_1, -1);  arg864_1 = None
        unsqueeze_4025 = torch.ops.aten.unsqueeze.default(unsqueeze_4024, -1);  unsqueeze_4024 = None
        mul_1669 = torch.ops.aten.mul.Tensor(mul_1668, unsqueeze_4025);  mul_1668 = unsqueeze_4025 = None
        unsqueeze_4026 = torch.ops.aten.unsqueeze.default(arg865_1, -1);  arg865_1 = None
        unsqueeze_4027 = torch.ops.aten.unsqueeze.default(unsqueeze_4026, -1);  unsqueeze_4026 = None
        add_1439 = torch.ops.aten.add.Tensor(mul_1669, unsqueeze_4027);  mul_1669 = unsqueeze_4027 = None
        add_1440 = torch.ops.aten.add.Tensor(add_1439, relu_441);  add_1439 = relu_441 = None
        relu_443 = torch.ops.aten.relu.default(add_1440);  add_1440 = None
        convolution_498 = torch.ops.aten.convolution.default(relu_443, arg866_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg866_1 = None
        add_1441 = torch.ops.aten.add.Tensor(arg868_1, 1e-05);  arg868_1 = None
        sqrt_498 = torch.ops.aten.sqrt.default(add_1441);  add_1441 = None
        reciprocal_498 = torch.ops.aten.reciprocal.default(sqrt_498);  sqrt_498 = None
        mul_1670 = torch.ops.aten.mul.Tensor(reciprocal_498, 1);  reciprocal_498 = None
        unsqueeze_4028 = torch.ops.aten.unsqueeze.default(arg867_1, -1);  arg867_1 = None
        unsqueeze_4029 = torch.ops.aten.unsqueeze.default(unsqueeze_4028, -1);  unsqueeze_4028 = None
        unsqueeze_4030 = torch.ops.aten.unsqueeze.default(mul_1670, -1);  mul_1670 = None
        unsqueeze_4031 = torch.ops.aten.unsqueeze.default(unsqueeze_4030, -1);  unsqueeze_4030 = None
        sub_498 = torch.ops.aten.sub.Tensor(convolution_498, unsqueeze_4029);  convolution_498 = unsqueeze_4029 = None
        mul_1671 = torch.ops.aten.mul.Tensor(sub_498, unsqueeze_4031);  sub_498 = unsqueeze_4031 = None
        unsqueeze_4032 = torch.ops.aten.unsqueeze.default(arg869_1, -1);  arg869_1 = None
        unsqueeze_4033 = torch.ops.aten.unsqueeze.default(unsqueeze_4032, -1);  unsqueeze_4032 = None
        mul_1672 = torch.ops.aten.mul.Tensor(mul_1671, unsqueeze_4033);  mul_1671 = unsqueeze_4033 = None
        unsqueeze_4034 = torch.ops.aten.unsqueeze.default(arg870_1, -1);  arg870_1 = None
        unsqueeze_4035 = torch.ops.aten.unsqueeze.default(unsqueeze_4034, -1);  unsqueeze_4034 = None
        add_1442 = torch.ops.aten.add.Tensor(mul_1672, unsqueeze_4035);  mul_1672 = unsqueeze_4035 = None
        relu_444 = torch.ops.aten.relu.default(add_1442);  add_1442 = None
        convolution_499 = torch.ops.aten.convolution.default(relu_444, arg871_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_444 = arg871_1 = None
        add_1443 = torch.ops.aten.add.Tensor(arg873_1, 1e-05);  arg873_1 = None
        sqrt_499 = torch.ops.aten.sqrt.default(add_1443);  add_1443 = None
        reciprocal_499 = torch.ops.aten.reciprocal.default(sqrt_499);  sqrt_499 = None
        mul_1673 = torch.ops.aten.mul.Tensor(reciprocal_499, 1);  reciprocal_499 = None
        unsqueeze_4036 = torch.ops.aten.unsqueeze.default(arg872_1, -1);  arg872_1 = None
        unsqueeze_4037 = torch.ops.aten.unsqueeze.default(unsqueeze_4036, -1);  unsqueeze_4036 = None
        unsqueeze_4038 = torch.ops.aten.unsqueeze.default(mul_1673, -1);  mul_1673 = None
        unsqueeze_4039 = torch.ops.aten.unsqueeze.default(unsqueeze_4038, -1);  unsqueeze_4038 = None
        sub_499 = torch.ops.aten.sub.Tensor(convolution_499, unsqueeze_4037);  convolution_499 = unsqueeze_4037 = None
        mul_1674 = torch.ops.aten.mul.Tensor(sub_499, unsqueeze_4039);  sub_499 = unsqueeze_4039 = None
        unsqueeze_4040 = torch.ops.aten.unsqueeze.default(arg874_1, -1);  arg874_1 = None
        unsqueeze_4041 = torch.ops.aten.unsqueeze.default(unsqueeze_4040, -1);  unsqueeze_4040 = None
        mul_1675 = torch.ops.aten.mul.Tensor(mul_1674, unsqueeze_4041);  mul_1674 = unsqueeze_4041 = None
        unsqueeze_4042 = torch.ops.aten.unsqueeze.default(arg875_1, -1);  arg875_1 = None
        unsqueeze_4043 = torch.ops.aten.unsqueeze.default(unsqueeze_4042, -1);  unsqueeze_4042 = None
        add_1444 = torch.ops.aten.add.Tensor(mul_1675, unsqueeze_4043);  mul_1675 = unsqueeze_4043 = None
        add_1445 = torch.ops.aten.add.Tensor(add_1444, relu_443);  add_1444 = relu_443 = None
        relu_445 = torch.ops.aten.relu.default(add_1445);  add_1445 = None
        convolution_500 = torch.ops.aten.convolution.default(relu_445, arg876_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg876_1 = None
        add_1446 = torch.ops.aten.add.Tensor(arg878_1, 1e-05);  arg878_1 = None
        sqrt_500 = torch.ops.aten.sqrt.default(add_1446);  add_1446 = None
        reciprocal_500 = torch.ops.aten.reciprocal.default(sqrt_500);  sqrt_500 = None
        mul_1676 = torch.ops.aten.mul.Tensor(reciprocal_500, 1);  reciprocal_500 = None
        unsqueeze_4044 = torch.ops.aten.unsqueeze.default(arg877_1, -1);  arg877_1 = None
        unsqueeze_4045 = torch.ops.aten.unsqueeze.default(unsqueeze_4044, -1);  unsqueeze_4044 = None
        unsqueeze_4046 = torch.ops.aten.unsqueeze.default(mul_1676, -1);  mul_1676 = None
        unsqueeze_4047 = torch.ops.aten.unsqueeze.default(unsqueeze_4046, -1);  unsqueeze_4046 = None
        sub_500 = torch.ops.aten.sub.Tensor(convolution_500, unsqueeze_4045);  convolution_500 = unsqueeze_4045 = None
        mul_1677 = torch.ops.aten.mul.Tensor(sub_500, unsqueeze_4047);  sub_500 = unsqueeze_4047 = None
        unsqueeze_4048 = torch.ops.aten.unsqueeze.default(arg879_1, -1);  arg879_1 = None
        unsqueeze_4049 = torch.ops.aten.unsqueeze.default(unsqueeze_4048, -1);  unsqueeze_4048 = None
        mul_1678 = torch.ops.aten.mul.Tensor(mul_1677, unsqueeze_4049);  mul_1677 = unsqueeze_4049 = None
        unsqueeze_4050 = torch.ops.aten.unsqueeze.default(arg880_1, -1);  arg880_1 = None
        unsqueeze_4051 = torch.ops.aten.unsqueeze.default(unsqueeze_4050, -1);  unsqueeze_4050 = None
        add_1447 = torch.ops.aten.add.Tensor(mul_1678, unsqueeze_4051);  mul_1678 = unsqueeze_4051 = None
        relu_446 = torch.ops.aten.relu.default(add_1447);  add_1447 = None
        convolution_501 = torch.ops.aten.convolution.default(relu_446, arg881_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_446 = arg881_1 = None
        add_1448 = torch.ops.aten.add.Tensor(arg883_1, 1e-05);  arg883_1 = None
        sqrt_501 = torch.ops.aten.sqrt.default(add_1448);  add_1448 = None
        reciprocal_501 = torch.ops.aten.reciprocal.default(sqrt_501);  sqrt_501 = None
        mul_1679 = torch.ops.aten.mul.Tensor(reciprocal_501, 1);  reciprocal_501 = None
        unsqueeze_4052 = torch.ops.aten.unsqueeze.default(arg882_1, -1);  arg882_1 = None
        unsqueeze_4053 = torch.ops.aten.unsqueeze.default(unsqueeze_4052, -1);  unsqueeze_4052 = None
        unsqueeze_4054 = torch.ops.aten.unsqueeze.default(mul_1679, -1);  mul_1679 = None
        unsqueeze_4055 = torch.ops.aten.unsqueeze.default(unsqueeze_4054, -1);  unsqueeze_4054 = None
        sub_501 = torch.ops.aten.sub.Tensor(convolution_501, unsqueeze_4053);  convolution_501 = unsqueeze_4053 = None
        mul_1680 = torch.ops.aten.mul.Tensor(sub_501, unsqueeze_4055);  sub_501 = unsqueeze_4055 = None
        unsqueeze_4056 = torch.ops.aten.unsqueeze.default(arg884_1, -1);  arg884_1 = None
        unsqueeze_4057 = torch.ops.aten.unsqueeze.default(unsqueeze_4056, -1);  unsqueeze_4056 = None
        mul_1681 = torch.ops.aten.mul.Tensor(mul_1680, unsqueeze_4057);  mul_1680 = unsqueeze_4057 = None
        unsqueeze_4058 = torch.ops.aten.unsqueeze.default(arg885_1, -1);  arg885_1 = None
        unsqueeze_4059 = torch.ops.aten.unsqueeze.default(unsqueeze_4058, -1);  unsqueeze_4058 = None
        add_1449 = torch.ops.aten.add.Tensor(mul_1681, unsqueeze_4059);  mul_1681 = unsqueeze_4059 = None
        add_1450 = torch.ops.aten.add.Tensor(add_1449, relu_445);  add_1449 = relu_445 = None
        relu_447 = torch.ops.aten.relu.default(add_1450);  add_1450 = None
        convolution_502 = torch.ops.aten.convolution.default(relu_430, arg886_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg886_1 = None
        add_1451 = torch.ops.aten.add.Tensor(arg888_1, 1e-05);  arg888_1 = None
        sqrt_502 = torch.ops.aten.sqrt.default(add_1451);  add_1451 = None
        reciprocal_502 = torch.ops.aten.reciprocal.default(sqrt_502);  sqrt_502 = None
        mul_1682 = torch.ops.aten.mul.Tensor(reciprocal_502, 1);  reciprocal_502 = None
        unsqueeze_4060 = torch.ops.aten.unsqueeze.default(arg887_1, -1);  arg887_1 = None
        unsqueeze_4061 = torch.ops.aten.unsqueeze.default(unsqueeze_4060, -1);  unsqueeze_4060 = None
        unsqueeze_4062 = torch.ops.aten.unsqueeze.default(mul_1682, -1);  mul_1682 = None
        unsqueeze_4063 = torch.ops.aten.unsqueeze.default(unsqueeze_4062, -1);  unsqueeze_4062 = None
        sub_502 = torch.ops.aten.sub.Tensor(convolution_502, unsqueeze_4061);  convolution_502 = unsqueeze_4061 = None
        mul_1683 = torch.ops.aten.mul.Tensor(sub_502, unsqueeze_4063);  sub_502 = unsqueeze_4063 = None
        unsqueeze_4064 = torch.ops.aten.unsqueeze.default(arg889_1, -1);  arg889_1 = None
        unsqueeze_4065 = torch.ops.aten.unsqueeze.default(unsqueeze_4064, -1);  unsqueeze_4064 = None
        mul_1684 = torch.ops.aten.mul.Tensor(mul_1683, unsqueeze_4065);  mul_1683 = unsqueeze_4065 = None
        unsqueeze_4066 = torch.ops.aten.unsqueeze.default(arg890_1, -1);  arg890_1 = None
        unsqueeze_4067 = torch.ops.aten.unsqueeze.default(unsqueeze_4066, -1);  unsqueeze_4066 = None
        add_1452 = torch.ops.aten.add.Tensor(mul_1684, unsqueeze_4067);  mul_1684 = unsqueeze_4067 = None
        relu_448 = torch.ops.aten.relu.default(add_1452);  add_1452 = None
        convolution_503 = torch.ops.aten.convolution.default(relu_448, arg891_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_448 = arg891_1 = None
        add_1453 = torch.ops.aten.add.Tensor(arg893_1, 1e-05);  arg893_1 = None
        sqrt_503 = torch.ops.aten.sqrt.default(add_1453);  add_1453 = None
        reciprocal_503 = torch.ops.aten.reciprocal.default(sqrt_503);  sqrt_503 = None
        mul_1685 = torch.ops.aten.mul.Tensor(reciprocal_503, 1);  reciprocal_503 = None
        unsqueeze_4068 = torch.ops.aten.unsqueeze.default(arg892_1, -1);  arg892_1 = None
        unsqueeze_4069 = torch.ops.aten.unsqueeze.default(unsqueeze_4068, -1);  unsqueeze_4068 = None
        unsqueeze_4070 = torch.ops.aten.unsqueeze.default(mul_1685, -1);  mul_1685 = None
        unsqueeze_4071 = torch.ops.aten.unsqueeze.default(unsqueeze_4070, -1);  unsqueeze_4070 = None
        sub_503 = torch.ops.aten.sub.Tensor(convolution_503, unsqueeze_4069);  convolution_503 = unsqueeze_4069 = None
        mul_1686 = torch.ops.aten.mul.Tensor(sub_503, unsqueeze_4071);  sub_503 = unsqueeze_4071 = None
        unsqueeze_4072 = torch.ops.aten.unsqueeze.default(arg894_1, -1);  arg894_1 = None
        unsqueeze_4073 = torch.ops.aten.unsqueeze.default(unsqueeze_4072, -1);  unsqueeze_4072 = None
        mul_1687 = torch.ops.aten.mul.Tensor(mul_1686, unsqueeze_4073);  mul_1686 = unsqueeze_4073 = None
        unsqueeze_4074 = torch.ops.aten.unsqueeze.default(arg895_1, -1);  arg895_1 = None
        unsqueeze_4075 = torch.ops.aten.unsqueeze.default(unsqueeze_4074, -1);  unsqueeze_4074 = None
        add_1454 = torch.ops.aten.add.Tensor(mul_1687, unsqueeze_4075);  mul_1687 = unsqueeze_4075 = None
        add_1455 = torch.ops.aten.add.Tensor(add_1454, relu_430);  add_1454 = relu_430 = None
        relu_449 = torch.ops.aten.relu.default(add_1455);  add_1455 = None
        convolution_504 = torch.ops.aten.convolution.default(relu_449, arg896_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg896_1 = None
        add_1456 = torch.ops.aten.add.Tensor(arg898_1, 1e-05);  arg898_1 = None
        sqrt_504 = torch.ops.aten.sqrt.default(add_1456);  add_1456 = None
        reciprocal_504 = torch.ops.aten.reciprocal.default(sqrt_504);  sqrt_504 = None
        mul_1688 = torch.ops.aten.mul.Tensor(reciprocal_504, 1);  reciprocal_504 = None
        unsqueeze_4076 = torch.ops.aten.unsqueeze.default(arg897_1, -1);  arg897_1 = None
        unsqueeze_4077 = torch.ops.aten.unsqueeze.default(unsqueeze_4076, -1);  unsqueeze_4076 = None
        unsqueeze_4078 = torch.ops.aten.unsqueeze.default(mul_1688, -1);  mul_1688 = None
        unsqueeze_4079 = torch.ops.aten.unsqueeze.default(unsqueeze_4078, -1);  unsqueeze_4078 = None
        sub_504 = torch.ops.aten.sub.Tensor(convolution_504, unsqueeze_4077);  convolution_504 = unsqueeze_4077 = None
        mul_1689 = torch.ops.aten.mul.Tensor(sub_504, unsqueeze_4079);  sub_504 = unsqueeze_4079 = None
        unsqueeze_4080 = torch.ops.aten.unsqueeze.default(arg899_1, -1);  arg899_1 = None
        unsqueeze_4081 = torch.ops.aten.unsqueeze.default(unsqueeze_4080, -1);  unsqueeze_4080 = None
        mul_1690 = torch.ops.aten.mul.Tensor(mul_1689, unsqueeze_4081);  mul_1689 = unsqueeze_4081 = None
        unsqueeze_4082 = torch.ops.aten.unsqueeze.default(arg900_1, -1);  arg900_1 = None
        unsqueeze_4083 = torch.ops.aten.unsqueeze.default(unsqueeze_4082, -1);  unsqueeze_4082 = None
        add_1457 = torch.ops.aten.add.Tensor(mul_1690, unsqueeze_4083);  mul_1690 = unsqueeze_4083 = None
        relu_450 = torch.ops.aten.relu.default(add_1457);  add_1457 = None
        convolution_505 = torch.ops.aten.convolution.default(relu_450, arg901_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_450 = arg901_1 = None
        add_1458 = torch.ops.aten.add.Tensor(arg903_1, 1e-05);  arg903_1 = None
        sqrt_505 = torch.ops.aten.sqrt.default(add_1458);  add_1458 = None
        reciprocal_505 = torch.ops.aten.reciprocal.default(sqrt_505);  sqrt_505 = None
        mul_1691 = torch.ops.aten.mul.Tensor(reciprocal_505, 1);  reciprocal_505 = None
        unsqueeze_4084 = torch.ops.aten.unsqueeze.default(arg902_1, -1);  arg902_1 = None
        unsqueeze_4085 = torch.ops.aten.unsqueeze.default(unsqueeze_4084, -1);  unsqueeze_4084 = None
        unsqueeze_4086 = torch.ops.aten.unsqueeze.default(mul_1691, -1);  mul_1691 = None
        unsqueeze_4087 = torch.ops.aten.unsqueeze.default(unsqueeze_4086, -1);  unsqueeze_4086 = None
        sub_505 = torch.ops.aten.sub.Tensor(convolution_505, unsqueeze_4085);  convolution_505 = unsqueeze_4085 = None
        mul_1692 = torch.ops.aten.mul.Tensor(sub_505, unsqueeze_4087);  sub_505 = unsqueeze_4087 = None
        unsqueeze_4088 = torch.ops.aten.unsqueeze.default(arg904_1, -1);  arg904_1 = None
        unsqueeze_4089 = torch.ops.aten.unsqueeze.default(unsqueeze_4088, -1);  unsqueeze_4088 = None
        mul_1693 = torch.ops.aten.mul.Tensor(mul_1692, unsqueeze_4089);  mul_1692 = unsqueeze_4089 = None
        unsqueeze_4090 = torch.ops.aten.unsqueeze.default(arg905_1, -1);  arg905_1 = None
        unsqueeze_4091 = torch.ops.aten.unsqueeze.default(unsqueeze_4090, -1);  unsqueeze_4090 = None
        add_1459 = torch.ops.aten.add.Tensor(mul_1693, unsqueeze_4091);  mul_1693 = unsqueeze_4091 = None
        add_1460 = torch.ops.aten.add.Tensor(add_1459, relu_449);  add_1459 = relu_449 = None
        relu_451 = torch.ops.aten.relu.default(add_1460);  add_1460 = None
        convolution_506 = torch.ops.aten.convolution.default(relu_451, arg906_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg906_1 = None
        add_1461 = torch.ops.aten.add.Tensor(arg908_1, 1e-05);  arg908_1 = None
        sqrt_506 = torch.ops.aten.sqrt.default(add_1461);  add_1461 = None
        reciprocal_506 = torch.ops.aten.reciprocal.default(sqrt_506);  sqrt_506 = None
        mul_1694 = torch.ops.aten.mul.Tensor(reciprocal_506, 1);  reciprocal_506 = None
        unsqueeze_4092 = torch.ops.aten.unsqueeze.default(arg907_1, -1);  arg907_1 = None
        unsqueeze_4093 = torch.ops.aten.unsqueeze.default(unsqueeze_4092, -1);  unsqueeze_4092 = None
        unsqueeze_4094 = torch.ops.aten.unsqueeze.default(mul_1694, -1);  mul_1694 = None
        unsqueeze_4095 = torch.ops.aten.unsqueeze.default(unsqueeze_4094, -1);  unsqueeze_4094 = None
        sub_506 = torch.ops.aten.sub.Tensor(convolution_506, unsqueeze_4093);  convolution_506 = unsqueeze_4093 = None
        mul_1695 = torch.ops.aten.mul.Tensor(sub_506, unsqueeze_4095);  sub_506 = unsqueeze_4095 = None
        unsqueeze_4096 = torch.ops.aten.unsqueeze.default(arg909_1, -1);  arg909_1 = None
        unsqueeze_4097 = torch.ops.aten.unsqueeze.default(unsqueeze_4096, -1);  unsqueeze_4096 = None
        mul_1696 = torch.ops.aten.mul.Tensor(mul_1695, unsqueeze_4097);  mul_1695 = unsqueeze_4097 = None
        unsqueeze_4098 = torch.ops.aten.unsqueeze.default(arg910_1, -1);  arg910_1 = None
        unsqueeze_4099 = torch.ops.aten.unsqueeze.default(unsqueeze_4098, -1);  unsqueeze_4098 = None
        add_1462 = torch.ops.aten.add.Tensor(mul_1696, unsqueeze_4099);  mul_1696 = unsqueeze_4099 = None
        relu_452 = torch.ops.aten.relu.default(add_1462);  add_1462 = None
        convolution_507 = torch.ops.aten.convolution.default(relu_452, arg911_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_452 = arg911_1 = None
        add_1463 = torch.ops.aten.add.Tensor(arg913_1, 1e-05);  arg913_1 = None
        sqrt_507 = torch.ops.aten.sqrt.default(add_1463);  add_1463 = None
        reciprocal_507 = torch.ops.aten.reciprocal.default(sqrt_507);  sqrt_507 = None
        mul_1697 = torch.ops.aten.mul.Tensor(reciprocal_507, 1);  reciprocal_507 = None
        unsqueeze_4100 = torch.ops.aten.unsqueeze.default(arg912_1, -1);  arg912_1 = None
        unsqueeze_4101 = torch.ops.aten.unsqueeze.default(unsqueeze_4100, -1);  unsqueeze_4100 = None
        unsqueeze_4102 = torch.ops.aten.unsqueeze.default(mul_1697, -1);  mul_1697 = None
        unsqueeze_4103 = torch.ops.aten.unsqueeze.default(unsqueeze_4102, -1);  unsqueeze_4102 = None
        sub_507 = torch.ops.aten.sub.Tensor(convolution_507, unsqueeze_4101);  convolution_507 = unsqueeze_4101 = None
        mul_1698 = torch.ops.aten.mul.Tensor(sub_507, unsqueeze_4103);  sub_507 = unsqueeze_4103 = None
        unsqueeze_4104 = torch.ops.aten.unsqueeze.default(arg914_1, -1);  arg914_1 = None
        unsqueeze_4105 = torch.ops.aten.unsqueeze.default(unsqueeze_4104, -1);  unsqueeze_4104 = None
        mul_1699 = torch.ops.aten.mul.Tensor(mul_1698, unsqueeze_4105);  mul_1698 = unsqueeze_4105 = None
        unsqueeze_4106 = torch.ops.aten.unsqueeze.default(arg915_1, -1);  arg915_1 = None
        unsqueeze_4107 = torch.ops.aten.unsqueeze.default(unsqueeze_4106, -1);  unsqueeze_4106 = None
        add_1464 = torch.ops.aten.add.Tensor(mul_1699, unsqueeze_4107);  mul_1699 = unsqueeze_4107 = None
        add_1465 = torch.ops.aten.add.Tensor(add_1464, relu_451);  add_1464 = relu_451 = None
        relu_453 = torch.ops.aten.relu.default(add_1465);  add_1465 = None
        convolution_508 = torch.ops.aten.convolution.default(relu_453, arg916_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg916_1 = None
        add_1466 = torch.ops.aten.add.Tensor(arg918_1, 1e-05);  arg918_1 = None
        sqrt_508 = torch.ops.aten.sqrt.default(add_1466);  add_1466 = None
        reciprocal_508 = torch.ops.aten.reciprocal.default(sqrt_508);  sqrt_508 = None
        mul_1700 = torch.ops.aten.mul.Tensor(reciprocal_508, 1);  reciprocal_508 = None
        unsqueeze_4108 = torch.ops.aten.unsqueeze.default(arg917_1, -1);  arg917_1 = None
        unsqueeze_4109 = torch.ops.aten.unsqueeze.default(unsqueeze_4108, -1);  unsqueeze_4108 = None
        unsqueeze_4110 = torch.ops.aten.unsqueeze.default(mul_1700, -1);  mul_1700 = None
        unsqueeze_4111 = torch.ops.aten.unsqueeze.default(unsqueeze_4110, -1);  unsqueeze_4110 = None
        sub_508 = torch.ops.aten.sub.Tensor(convolution_508, unsqueeze_4109);  convolution_508 = unsqueeze_4109 = None
        mul_1701 = torch.ops.aten.mul.Tensor(sub_508, unsqueeze_4111);  sub_508 = unsqueeze_4111 = None
        unsqueeze_4112 = torch.ops.aten.unsqueeze.default(arg919_1, -1);  arg919_1 = None
        unsqueeze_4113 = torch.ops.aten.unsqueeze.default(unsqueeze_4112, -1);  unsqueeze_4112 = None
        mul_1702 = torch.ops.aten.mul.Tensor(mul_1701, unsqueeze_4113);  mul_1701 = unsqueeze_4113 = None
        unsqueeze_4114 = torch.ops.aten.unsqueeze.default(arg920_1, -1);  arg920_1 = None
        unsqueeze_4115 = torch.ops.aten.unsqueeze.default(unsqueeze_4114, -1);  unsqueeze_4114 = None
        add_1467 = torch.ops.aten.add.Tensor(mul_1702, unsqueeze_4115);  mul_1702 = unsqueeze_4115 = None
        relu_454 = torch.ops.aten.relu.default(add_1467);  add_1467 = None
        convolution_509 = torch.ops.aten.convolution.default(relu_454, arg921_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_454 = arg921_1 = None
        add_1468 = torch.ops.aten.add.Tensor(arg923_1, 1e-05);  arg923_1 = None
        sqrt_509 = torch.ops.aten.sqrt.default(add_1468);  add_1468 = None
        reciprocal_509 = torch.ops.aten.reciprocal.default(sqrt_509);  sqrt_509 = None
        mul_1703 = torch.ops.aten.mul.Tensor(reciprocal_509, 1);  reciprocal_509 = None
        unsqueeze_4116 = torch.ops.aten.unsqueeze.default(arg922_1, -1);  arg922_1 = None
        unsqueeze_4117 = torch.ops.aten.unsqueeze.default(unsqueeze_4116, -1);  unsqueeze_4116 = None
        unsqueeze_4118 = torch.ops.aten.unsqueeze.default(mul_1703, -1);  mul_1703 = None
        unsqueeze_4119 = torch.ops.aten.unsqueeze.default(unsqueeze_4118, -1);  unsqueeze_4118 = None
        sub_509 = torch.ops.aten.sub.Tensor(convolution_509, unsqueeze_4117);  convolution_509 = unsqueeze_4117 = None
        mul_1704 = torch.ops.aten.mul.Tensor(sub_509, unsqueeze_4119);  sub_509 = unsqueeze_4119 = None
        unsqueeze_4120 = torch.ops.aten.unsqueeze.default(arg924_1, -1);  arg924_1 = None
        unsqueeze_4121 = torch.ops.aten.unsqueeze.default(unsqueeze_4120, -1);  unsqueeze_4120 = None
        mul_1705 = torch.ops.aten.mul.Tensor(mul_1704, unsqueeze_4121);  mul_1704 = unsqueeze_4121 = None
        unsqueeze_4122 = torch.ops.aten.unsqueeze.default(arg925_1, -1);  arg925_1 = None
        unsqueeze_4123 = torch.ops.aten.unsqueeze.default(unsqueeze_4122, -1);  unsqueeze_4122 = None
        add_1469 = torch.ops.aten.add.Tensor(mul_1705, unsqueeze_4123);  mul_1705 = unsqueeze_4123 = None
        add_1470 = torch.ops.aten.add.Tensor(add_1469, relu_453);  add_1469 = relu_453 = None
        relu_455 = torch.ops.aten.relu.default(add_1470);  add_1470 = None
        convolution_510 = torch.ops.aten.convolution.default(relu_431, arg926_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg926_1 = None
        add_1471 = torch.ops.aten.add.Tensor(arg928_1, 1e-05);  arg928_1 = None
        sqrt_510 = torch.ops.aten.sqrt.default(add_1471);  add_1471 = None
        reciprocal_510 = torch.ops.aten.reciprocal.default(sqrt_510);  sqrt_510 = None
        mul_1706 = torch.ops.aten.mul.Tensor(reciprocal_510, 1);  reciprocal_510 = None
        unsqueeze_4124 = torch.ops.aten.unsqueeze.default(arg927_1, -1);  arg927_1 = None
        unsqueeze_4125 = torch.ops.aten.unsqueeze.default(unsqueeze_4124, -1);  unsqueeze_4124 = None
        unsqueeze_4126 = torch.ops.aten.unsqueeze.default(mul_1706, -1);  mul_1706 = None
        unsqueeze_4127 = torch.ops.aten.unsqueeze.default(unsqueeze_4126, -1);  unsqueeze_4126 = None
        sub_510 = torch.ops.aten.sub.Tensor(convolution_510, unsqueeze_4125);  convolution_510 = unsqueeze_4125 = None
        mul_1707 = torch.ops.aten.mul.Tensor(sub_510, unsqueeze_4127);  sub_510 = unsqueeze_4127 = None
        unsqueeze_4128 = torch.ops.aten.unsqueeze.default(arg929_1, -1);  arg929_1 = None
        unsqueeze_4129 = torch.ops.aten.unsqueeze.default(unsqueeze_4128, -1);  unsqueeze_4128 = None
        mul_1708 = torch.ops.aten.mul.Tensor(mul_1707, unsqueeze_4129);  mul_1707 = unsqueeze_4129 = None
        unsqueeze_4130 = torch.ops.aten.unsqueeze.default(arg930_1, -1);  arg930_1 = None
        unsqueeze_4131 = torch.ops.aten.unsqueeze.default(unsqueeze_4130, -1);  unsqueeze_4130 = None
        add_1472 = torch.ops.aten.add.Tensor(mul_1708, unsqueeze_4131);  mul_1708 = unsqueeze_4131 = None
        relu_456 = torch.ops.aten.relu.default(add_1472);  add_1472 = None
        convolution_511 = torch.ops.aten.convolution.default(relu_456, arg931_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_456 = arg931_1 = None
        add_1473 = torch.ops.aten.add.Tensor(arg933_1, 1e-05);  arg933_1 = None
        sqrt_511 = torch.ops.aten.sqrt.default(add_1473);  add_1473 = None
        reciprocal_511 = torch.ops.aten.reciprocal.default(sqrt_511);  sqrt_511 = None
        mul_1709 = torch.ops.aten.mul.Tensor(reciprocal_511, 1);  reciprocal_511 = None
        unsqueeze_4132 = torch.ops.aten.unsqueeze.default(arg932_1, -1);  arg932_1 = None
        unsqueeze_4133 = torch.ops.aten.unsqueeze.default(unsqueeze_4132, -1);  unsqueeze_4132 = None
        unsqueeze_4134 = torch.ops.aten.unsqueeze.default(mul_1709, -1);  mul_1709 = None
        unsqueeze_4135 = torch.ops.aten.unsqueeze.default(unsqueeze_4134, -1);  unsqueeze_4134 = None
        sub_511 = torch.ops.aten.sub.Tensor(convolution_511, unsqueeze_4133);  convolution_511 = unsqueeze_4133 = None
        mul_1710 = torch.ops.aten.mul.Tensor(sub_511, unsqueeze_4135);  sub_511 = unsqueeze_4135 = None
        unsqueeze_4136 = torch.ops.aten.unsqueeze.default(arg934_1, -1);  arg934_1 = None
        unsqueeze_4137 = torch.ops.aten.unsqueeze.default(unsqueeze_4136, -1);  unsqueeze_4136 = None
        mul_1711 = torch.ops.aten.mul.Tensor(mul_1710, unsqueeze_4137);  mul_1710 = unsqueeze_4137 = None
        unsqueeze_4138 = torch.ops.aten.unsqueeze.default(arg935_1, -1);  arg935_1 = None
        unsqueeze_4139 = torch.ops.aten.unsqueeze.default(unsqueeze_4138, -1);  unsqueeze_4138 = None
        add_1474 = torch.ops.aten.add.Tensor(mul_1711, unsqueeze_4139);  mul_1711 = unsqueeze_4139 = None
        add_1475 = torch.ops.aten.add.Tensor(add_1474, relu_431);  add_1474 = relu_431 = None
        relu_457 = torch.ops.aten.relu.default(add_1475);  add_1475 = None
        convolution_512 = torch.ops.aten.convolution.default(relu_457, arg936_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg936_1 = None
        add_1476 = torch.ops.aten.add.Tensor(arg938_1, 1e-05);  arg938_1 = None
        sqrt_512 = torch.ops.aten.sqrt.default(add_1476);  add_1476 = None
        reciprocal_512 = torch.ops.aten.reciprocal.default(sqrt_512);  sqrt_512 = None
        mul_1712 = torch.ops.aten.mul.Tensor(reciprocal_512, 1);  reciprocal_512 = None
        unsqueeze_4140 = torch.ops.aten.unsqueeze.default(arg937_1, -1);  arg937_1 = None
        unsqueeze_4141 = torch.ops.aten.unsqueeze.default(unsqueeze_4140, -1);  unsqueeze_4140 = None
        unsqueeze_4142 = torch.ops.aten.unsqueeze.default(mul_1712, -1);  mul_1712 = None
        unsqueeze_4143 = torch.ops.aten.unsqueeze.default(unsqueeze_4142, -1);  unsqueeze_4142 = None
        sub_512 = torch.ops.aten.sub.Tensor(convolution_512, unsqueeze_4141);  convolution_512 = unsqueeze_4141 = None
        mul_1713 = torch.ops.aten.mul.Tensor(sub_512, unsqueeze_4143);  sub_512 = unsqueeze_4143 = None
        unsqueeze_4144 = torch.ops.aten.unsqueeze.default(arg939_1, -1);  arg939_1 = None
        unsqueeze_4145 = torch.ops.aten.unsqueeze.default(unsqueeze_4144, -1);  unsqueeze_4144 = None
        mul_1714 = torch.ops.aten.mul.Tensor(mul_1713, unsqueeze_4145);  mul_1713 = unsqueeze_4145 = None
        unsqueeze_4146 = torch.ops.aten.unsqueeze.default(arg940_1, -1);  arg940_1 = None
        unsqueeze_4147 = torch.ops.aten.unsqueeze.default(unsqueeze_4146, -1);  unsqueeze_4146 = None
        add_1477 = torch.ops.aten.add.Tensor(mul_1714, unsqueeze_4147);  mul_1714 = unsqueeze_4147 = None
        relu_458 = torch.ops.aten.relu.default(add_1477);  add_1477 = None
        convolution_513 = torch.ops.aten.convolution.default(relu_458, arg941_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_458 = arg941_1 = None
        add_1478 = torch.ops.aten.add.Tensor(arg943_1, 1e-05);  arg943_1 = None
        sqrt_513 = torch.ops.aten.sqrt.default(add_1478);  add_1478 = None
        reciprocal_513 = torch.ops.aten.reciprocal.default(sqrt_513);  sqrt_513 = None
        mul_1715 = torch.ops.aten.mul.Tensor(reciprocal_513, 1);  reciprocal_513 = None
        unsqueeze_4148 = torch.ops.aten.unsqueeze.default(arg942_1, -1);  arg942_1 = None
        unsqueeze_4149 = torch.ops.aten.unsqueeze.default(unsqueeze_4148, -1);  unsqueeze_4148 = None
        unsqueeze_4150 = torch.ops.aten.unsqueeze.default(mul_1715, -1);  mul_1715 = None
        unsqueeze_4151 = torch.ops.aten.unsqueeze.default(unsqueeze_4150, -1);  unsqueeze_4150 = None
        sub_513 = torch.ops.aten.sub.Tensor(convolution_513, unsqueeze_4149);  convolution_513 = unsqueeze_4149 = None
        mul_1716 = torch.ops.aten.mul.Tensor(sub_513, unsqueeze_4151);  sub_513 = unsqueeze_4151 = None
        unsqueeze_4152 = torch.ops.aten.unsqueeze.default(arg944_1, -1);  arg944_1 = None
        unsqueeze_4153 = torch.ops.aten.unsqueeze.default(unsqueeze_4152, -1);  unsqueeze_4152 = None
        mul_1717 = torch.ops.aten.mul.Tensor(mul_1716, unsqueeze_4153);  mul_1716 = unsqueeze_4153 = None
        unsqueeze_4154 = torch.ops.aten.unsqueeze.default(arg945_1, -1);  arg945_1 = None
        unsqueeze_4155 = torch.ops.aten.unsqueeze.default(unsqueeze_4154, -1);  unsqueeze_4154 = None
        add_1479 = torch.ops.aten.add.Tensor(mul_1717, unsqueeze_4155);  mul_1717 = unsqueeze_4155 = None
        add_1480 = torch.ops.aten.add.Tensor(add_1479, relu_457);  add_1479 = relu_457 = None
        relu_459 = torch.ops.aten.relu.default(add_1480);  add_1480 = None
        convolution_514 = torch.ops.aten.convolution.default(relu_459, arg946_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg946_1 = None
        add_1481 = torch.ops.aten.add.Tensor(arg948_1, 1e-05);  arg948_1 = None
        sqrt_514 = torch.ops.aten.sqrt.default(add_1481);  add_1481 = None
        reciprocal_514 = torch.ops.aten.reciprocal.default(sqrt_514);  sqrt_514 = None
        mul_1718 = torch.ops.aten.mul.Tensor(reciprocal_514, 1);  reciprocal_514 = None
        unsqueeze_4156 = torch.ops.aten.unsqueeze.default(arg947_1, -1);  arg947_1 = None
        unsqueeze_4157 = torch.ops.aten.unsqueeze.default(unsqueeze_4156, -1);  unsqueeze_4156 = None
        unsqueeze_4158 = torch.ops.aten.unsqueeze.default(mul_1718, -1);  mul_1718 = None
        unsqueeze_4159 = torch.ops.aten.unsqueeze.default(unsqueeze_4158, -1);  unsqueeze_4158 = None
        sub_514 = torch.ops.aten.sub.Tensor(convolution_514, unsqueeze_4157);  convolution_514 = unsqueeze_4157 = None
        mul_1719 = torch.ops.aten.mul.Tensor(sub_514, unsqueeze_4159);  sub_514 = unsqueeze_4159 = None
        unsqueeze_4160 = torch.ops.aten.unsqueeze.default(arg949_1, -1);  arg949_1 = None
        unsqueeze_4161 = torch.ops.aten.unsqueeze.default(unsqueeze_4160, -1);  unsqueeze_4160 = None
        mul_1720 = torch.ops.aten.mul.Tensor(mul_1719, unsqueeze_4161);  mul_1719 = unsqueeze_4161 = None
        unsqueeze_4162 = torch.ops.aten.unsqueeze.default(arg950_1, -1);  arg950_1 = None
        unsqueeze_4163 = torch.ops.aten.unsqueeze.default(unsqueeze_4162, -1);  unsqueeze_4162 = None
        add_1482 = torch.ops.aten.add.Tensor(mul_1720, unsqueeze_4163);  mul_1720 = unsqueeze_4163 = None
        relu_460 = torch.ops.aten.relu.default(add_1482);  add_1482 = None
        convolution_515 = torch.ops.aten.convolution.default(relu_460, arg951_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_460 = arg951_1 = None
        add_1483 = torch.ops.aten.add.Tensor(arg953_1, 1e-05);  arg953_1 = None
        sqrt_515 = torch.ops.aten.sqrt.default(add_1483);  add_1483 = None
        reciprocal_515 = torch.ops.aten.reciprocal.default(sqrt_515);  sqrt_515 = None
        mul_1721 = torch.ops.aten.mul.Tensor(reciprocal_515, 1);  reciprocal_515 = None
        unsqueeze_4164 = torch.ops.aten.unsqueeze.default(arg952_1, -1);  arg952_1 = None
        unsqueeze_4165 = torch.ops.aten.unsqueeze.default(unsqueeze_4164, -1);  unsqueeze_4164 = None
        unsqueeze_4166 = torch.ops.aten.unsqueeze.default(mul_1721, -1);  mul_1721 = None
        unsqueeze_4167 = torch.ops.aten.unsqueeze.default(unsqueeze_4166, -1);  unsqueeze_4166 = None
        sub_515 = torch.ops.aten.sub.Tensor(convolution_515, unsqueeze_4165);  convolution_515 = unsqueeze_4165 = None
        mul_1722 = torch.ops.aten.mul.Tensor(sub_515, unsqueeze_4167);  sub_515 = unsqueeze_4167 = None
        unsqueeze_4168 = torch.ops.aten.unsqueeze.default(arg954_1, -1);  arg954_1 = None
        unsqueeze_4169 = torch.ops.aten.unsqueeze.default(unsqueeze_4168, -1);  unsqueeze_4168 = None
        mul_1723 = torch.ops.aten.mul.Tensor(mul_1722, unsqueeze_4169);  mul_1722 = unsqueeze_4169 = None
        unsqueeze_4170 = torch.ops.aten.unsqueeze.default(arg955_1, -1);  arg955_1 = None
        unsqueeze_4171 = torch.ops.aten.unsqueeze.default(unsqueeze_4170, -1);  unsqueeze_4170 = None
        add_1484 = torch.ops.aten.add.Tensor(mul_1723, unsqueeze_4171);  mul_1723 = unsqueeze_4171 = None
        add_1485 = torch.ops.aten.add.Tensor(add_1484, relu_459);  add_1484 = relu_459 = None
        relu_461 = torch.ops.aten.relu.default(add_1485);  add_1485 = None
        convolution_516 = torch.ops.aten.convolution.default(relu_461, arg956_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg956_1 = None
        add_1486 = torch.ops.aten.add.Tensor(arg958_1, 1e-05);  arg958_1 = None
        sqrt_516 = torch.ops.aten.sqrt.default(add_1486);  add_1486 = None
        reciprocal_516 = torch.ops.aten.reciprocal.default(sqrt_516);  sqrt_516 = None
        mul_1724 = torch.ops.aten.mul.Tensor(reciprocal_516, 1);  reciprocal_516 = None
        unsqueeze_4172 = torch.ops.aten.unsqueeze.default(arg957_1, -1);  arg957_1 = None
        unsqueeze_4173 = torch.ops.aten.unsqueeze.default(unsqueeze_4172, -1);  unsqueeze_4172 = None
        unsqueeze_4174 = torch.ops.aten.unsqueeze.default(mul_1724, -1);  mul_1724 = None
        unsqueeze_4175 = torch.ops.aten.unsqueeze.default(unsqueeze_4174, -1);  unsqueeze_4174 = None
        sub_516 = torch.ops.aten.sub.Tensor(convolution_516, unsqueeze_4173);  convolution_516 = unsqueeze_4173 = None
        mul_1725 = torch.ops.aten.mul.Tensor(sub_516, unsqueeze_4175);  sub_516 = unsqueeze_4175 = None
        unsqueeze_4176 = torch.ops.aten.unsqueeze.default(arg959_1, -1);  arg959_1 = None
        unsqueeze_4177 = torch.ops.aten.unsqueeze.default(unsqueeze_4176, -1);  unsqueeze_4176 = None
        mul_1726 = torch.ops.aten.mul.Tensor(mul_1725, unsqueeze_4177);  mul_1725 = unsqueeze_4177 = None
        unsqueeze_4178 = torch.ops.aten.unsqueeze.default(arg960_1, -1);  arg960_1 = None
        unsqueeze_4179 = torch.ops.aten.unsqueeze.default(unsqueeze_4178, -1);  unsqueeze_4178 = None
        add_1487 = torch.ops.aten.add.Tensor(mul_1726, unsqueeze_4179);  mul_1726 = unsqueeze_4179 = None
        relu_462 = torch.ops.aten.relu.default(add_1487);  add_1487 = None
        convolution_517 = torch.ops.aten.convolution.default(relu_462, arg961_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_462 = arg961_1 = None
        add_1488 = torch.ops.aten.add.Tensor(arg963_1, 1e-05);  arg963_1 = None
        sqrt_517 = torch.ops.aten.sqrt.default(add_1488);  add_1488 = None
        reciprocal_517 = torch.ops.aten.reciprocal.default(sqrt_517);  sqrt_517 = None
        mul_1727 = torch.ops.aten.mul.Tensor(reciprocal_517, 1);  reciprocal_517 = None
        unsqueeze_4180 = torch.ops.aten.unsqueeze.default(arg962_1, -1);  arg962_1 = None
        unsqueeze_4181 = torch.ops.aten.unsqueeze.default(unsqueeze_4180, -1);  unsqueeze_4180 = None
        unsqueeze_4182 = torch.ops.aten.unsqueeze.default(mul_1727, -1);  mul_1727 = None
        unsqueeze_4183 = torch.ops.aten.unsqueeze.default(unsqueeze_4182, -1);  unsqueeze_4182 = None
        sub_517 = torch.ops.aten.sub.Tensor(convolution_517, unsqueeze_4181);  convolution_517 = unsqueeze_4181 = None
        mul_1728 = torch.ops.aten.mul.Tensor(sub_517, unsqueeze_4183);  sub_517 = unsqueeze_4183 = None
        unsqueeze_4184 = torch.ops.aten.unsqueeze.default(arg964_1, -1);  arg964_1 = None
        unsqueeze_4185 = torch.ops.aten.unsqueeze.default(unsqueeze_4184, -1);  unsqueeze_4184 = None
        mul_1729 = torch.ops.aten.mul.Tensor(mul_1728, unsqueeze_4185);  mul_1728 = unsqueeze_4185 = None
        unsqueeze_4186 = torch.ops.aten.unsqueeze.default(arg965_1, -1);  arg965_1 = None
        unsqueeze_4187 = torch.ops.aten.unsqueeze.default(unsqueeze_4186, -1);  unsqueeze_4186 = None
        add_1489 = torch.ops.aten.add.Tensor(mul_1729, unsqueeze_4187);  mul_1729 = unsqueeze_4187 = None
        add_1490 = torch.ops.aten.add.Tensor(add_1489, relu_461);  add_1489 = relu_461 = None
        relu_463 = torch.ops.aten.relu.default(add_1490);  add_1490 = None
        convolution_518 = torch.ops.aten.convolution.default(relu_447, arg966_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg966_1 = None
        add_1491 = torch.ops.aten.add.Tensor(arg968_1, 1e-05);  arg968_1 = None
        sqrt_518 = torch.ops.aten.sqrt.default(add_1491);  add_1491 = None
        reciprocal_518 = torch.ops.aten.reciprocal.default(sqrt_518);  sqrt_518 = None
        mul_1730 = torch.ops.aten.mul.Tensor(reciprocal_518, 1);  reciprocal_518 = None
        unsqueeze_4188 = torch.ops.aten.unsqueeze.default(arg967_1, -1);  arg967_1 = None
        unsqueeze_4189 = torch.ops.aten.unsqueeze.default(unsqueeze_4188, -1);  unsqueeze_4188 = None
        unsqueeze_4190 = torch.ops.aten.unsqueeze.default(mul_1730, -1);  mul_1730 = None
        unsqueeze_4191 = torch.ops.aten.unsqueeze.default(unsqueeze_4190, -1);  unsqueeze_4190 = None
        sub_518 = torch.ops.aten.sub.Tensor(convolution_518, unsqueeze_4189);  convolution_518 = unsqueeze_4189 = None
        mul_1731 = torch.ops.aten.mul.Tensor(sub_518, unsqueeze_4191);  sub_518 = unsqueeze_4191 = None
        unsqueeze_4192 = torch.ops.aten.unsqueeze.default(arg969_1, -1);  arg969_1 = None
        unsqueeze_4193 = torch.ops.aten.unsqueeze.default(unsqueeze_4192, -1);  unsqueeze_4192 = None
        mul_1732 = torch.ops.aten.mul.Tensor(mul_1731, unsqueeze_4193);  mul_1731 = unsqueeze_4193 = None
        unsqueeze_4194 = torch.ops.aten.unsqueeze.default(arg970_1, -1);  arg970_1 = None
        unsqueeze_4195 = torch.ops.aten.unsqueeze.default(unsqueeze_4194, -1);  unsqueeze_4194 = None
        add_1492 = torch.ops.aten.add.Tensor(mul_1732, unsqueeze_4195);  mul_1732 = unsqueeze_4195 = None
        iota_88 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1733 = torch.ops.aten.mul.Tensor(iota_88, 1);  iota_88 = None
        add_1493 = torch.ops.aten.add.Tensor(mul_1733, 0);  mul_1733 = None
        convert_element_type_1214 = torch.ops.prims.convert_element_type.default(add_1493, torch.float32);  add_1493 = None
        add_1494 = torch.ops.aten.add.Tensor(convert_element_type_1214, 0.0);  convert_element_type_1214 = None
        mul_1734 = torch.ops.aten.mul.Tensor(add_1494, 0.5);  add_1494 = None
        convert_element_type_1215 = torch.ops.prims.convert_element_type.default(mul_1734, torch.int64);  mul_1734 = None
        unsqueeze_4196 = torch.ops.aten.unsqueeze.default(convert_element_type_1215, -1);  convert_element_type_1215 = None
        iota_89 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1735 = torch.ops.aten.mul.Tensor(iota_89, 1);  iota_89 = None
        add_1495 = torch.ops.aten.add.Tensor(mul_1735, 0);  mul_1735 = None
        convert_element_type_1216 = torch.ops.prims.convert_element_type.default(add_1495, torch.float32);  add_1495 = None
        add_1496 = torch.ops.aten.add.Tensor(convert_element_type_1216, 0.0);  convert_element_type_1216 = None
        mul_1736 = torch.ops.aten.mul.Tensor(add_1496, 0.5);  add_1496 = None
        convert_element_type_1217 = torch.ops.prims.convert_element_type.default(mul_1736, torch.int64);  mul_1736 = None
        _unsafe_index_44 = torch.ops.aten._unsafe_index.Tensor(add_1492, [None, None, unsqueeze_4196, convert_element_type_1217]);  add_1492 = unsqueeze_4196 = convert_element_type_1217 = None
        add_1497 = torch.ops.aten.add.Tensor(relu_439, _unsafe_index_44);  _unsafe_index_44 = None
        convolution_519 = torch.ops.aten.convolution.default(relu_455, arg971_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg971_1 = None
        add_1498 = torch.ops.aten.add.Tensor(arg973_1, 1e-05);  arg973_1 = None
        sqrt_519 = torch.ops.aten.sqrt.default(add_1498);  add_1498 = None
        reciprocal_519 = torch.ops.aten.reciprocal.default(sqrt_519);  sqrt_519 = None
        mul_1737 = torch.ops.aten.mul.Tensor(reciprocal_519, 1);  reciprocal_519 = None
        unsqueeze_4197 = torch.ops.aten.unsqueeze.default(arg972_1, -1);  arg972_1 = None
        unsqueeze_4198 = torch.ops.aten.unsqueeze.default(unsqueeze_4197, -1);  unsqueeze_4197 = None
        unsqueeze_4199 = torch.ops.aten.unsqueeze.default(mul_1737, -1);  mul_1737 = None
        unsqueeze_4200 = torch.ops.aten.unsqueeze.default(unsqueeze_4199, -1);  unsqueeze_4199 = None
        sub_519 = torch.ops.aten.sub.Tensor(convolution_519, unsqueeze_4198);  convolution_519 = unsqueeze_4198 = None
        mul_1738 = torch.ops.aten.mul.Tensor(sub_519, unsqueeze_4200);  sub_519 = unsqueeze_4200 = None
        unsqueeze_4201 = torch.ops.aten.unsqueeze.default(arg974_1, -1);  arg974_1 = None
        unsqueeze_4202 = torch.ops.aten.unsqueeze.default(unsqueeze_4201, -1);  unsqueeze_4201 = None
        mul_1739 = torch.ops.aten.mul.Tensor(mul_1738, unsqueeze_4202);  mul_1738 = unsqueeze_4202 = None
        unsqueeze_4203 = torch.ops.aten.unsqueeze.default(arg975_1, -1);  arg975_1 = None
        unsqueeze_4204 = torch.ops.aten.unsqueeze.default(unsqueeze_4203, -1);  unsqueeze_4203 = None
        add_1499 = torch.ops.aten.add.Tensor(mul_1739, unsqueeze_4204);  mul_1739 = unsqueeze_4204 = None
        iota_90 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1740 = torch.ops.aten.mul.Tensor(iota_90, 1);  iota_90 = None
        add_1500 = torch.ops.aten.add.Tensor(mul_1740, 0);  mul_1740 = None
        convert_element_type_1220 = torch.ops.prims.convert_element_type.default(add_1500, torch.float32);  add_1500 = None
        add_1501 = torch.ops.aten.add.Tensor(convert_element_type_1220, 0.0);  convert_element_type_1220 = None
        mul_1741 = torch.ops.aten.mul.Tensor(add_1501, 0.25);  add_1501 = None
        convert_element_type_1221 = torch.ops.prims.convert_element_type.default(mul_1741, torch.int64);  mul_1741 = None
        unsqueeze_4205 = torch.ops.aten.unsqueeze.default(convert_element_type_1221, -1);  convert_element_type_1221 = None
        iota_91 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1742 = torch.ops.aten.mul.Tensor(iota_91, 1);  iota_91 = None
        add_1502 = torch.ops.aten.add.Tensor(mul_1742, 0);  mul_1742 = None
        convert_element_type_1222 = torch.ops.prims.convert_element_type.default(add_1502, torch.float32);  add_1502 = None
        add_1503 = torch.ops.aten.add.Tensor(convert_element_type_1222, 0.0);  convert_element_type_1222 = None
        mul_1743 = torch.ops.aten.mul.Tensor(add_1503, 0.25);  add_1503 = None
        convert_element_type_1223 = torch.ops.prims.convert_element_type.default(mul_1743, torch.int64);  mul_1743 = None
        _unsafe_index_45 = torch.ops.aten._unsafe_index.Tensor(add_1499, [None, None, unsqueeze_4205, convert_element_type_1223]);  add_1499 = unsqueeze_4205 = convert_element_type_1223 = None
        add_1504 = torch.ops.aten.add.Tensor(add_1497, _unsafe_index_45);  add_1497 = _unsafe_index_45 = None
        convolution_520 = torch.ops.aten.convolution.default(relu_463, arg976_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg976_1 = None
        add_1505 = torch.ops.aten.add.Tensor(arg978_1, 1e-05);  arg978_1 = None
        sqrt_520 = torch.ops.aten.sqrt.default(add_1505);  add_1505 = None
        reciprocal_520 = torch.ops.aten.reciprocal.default(sqrt_520);  sqrt_520 = None
        mul_1744 = torch.ops.aten.mul.Tensor(reciprocal_520, 1);  reciprocal_520 = None
        unsqueeze_4206 = torch.ops.aten.unsqueeze.default(arg977_1, -1);  arg977_1 = None
        unsqueeze_4207 = torch.ops.aten.unsqueeze.default(unsqueeze_4206, -1);  unsqueeze_4206 = None
        unsqueeze_4208 = torch.ops.aten.unsqueeze.default(mul_1744, -1);  mul_1744 = None
        unsqueeze_4209 = torch.ops.aten.unsqueeze.default(unsqueeze_4208, -1);  unsqueeze_4208 = None
        sub_520 = torch.ops.aten.sub.Tensor(convolution_520, unsqueeze_4207);  convolution_520 = unsqueeze_4207 = None
        mul_1745 = torch.ops.aten.mul.Tensor(sub_520, unsqueeze_4209);  sub_520 = unsqueeze_4209 = None
        unsqueeze_4210 = torch.ops.aten.unsqueeze.default(arg979_1, -1);  arg979_1 = None
        unsqueeze_4211 = torch.ops.aten.unsqueeze.default(unsqueeze_4210, -1);  unsqueeze_4210 = None
        mul_1746 = torch.ops.aten.mul.Tensor(mul_1745, unsqueeze_4211);  mul_1745 = unsqueeze_4211 = None
        unsqueeze_4212 = torch.ops.aten.unsqueeze.default(arg980_1, -1);  arg980_1 = None
        unsqueeze_4213 = torch.ops.aten.unsqueeze.default(unsqueeze_4212, -1);  unsqueeze_4212 = None
        add_1506 = torch.ops.aten.add.Tensor(mul_1746, unsqueeze_4213);  mul_1746 = unsqueeze_4213 = None
        iota_92 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1747 = torch.ops.aten.mul.Tensor(iota_92, 1);  iota_92 = None
        add_1507 = torch.ops.aten.add.Tensor(mul_1747, 0);  mul_1747 = None
        convert_element_type_1226 = torch.ops.prims.convert_element_type.default(add_1507, torch.float32);  add_1507 = None
        add_1508 = torch.ops.aten.add.Tensor(convert_element_type_1226, 0.0);  convert_element_type_1226 = None
        mul_1748 = torch.ops.aten.mul.Tensor(add_1508, 0.125);  add_1508 = None
        convert_element_type_1227 = torch.ops.prims.convert_element_type.default(mul_1748, torch.int64);  mul_1748 = None
        unsqueeze_4214 = torch.ops.aten.unsqueeze.default(convert_element_type_1227, -1);  convert_element_type_1227 = None
        iota_93 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1749 = torch.ops.aten.mul.Tensor(iota_93, 1);  iota_93 = None
        add_1509 = torch.ops.aten.add.Tensor(mul_1749, 0);  mul_1749 = None
        convert_element_type_1228 = torch.ops.prims.convert_element_type.default(add_1509, torch.float32);  add_1509 = None
        add_1510 = torch.ops.aten.add.Tensor(convert_element_type_1228, 0.0);  convert_element_type_1228 = None
        mul_1750 = torch.ops.aten.mul.Tensor(add_1510, 0.125);  add_1510 = None
        convert_element_type_1229 = torch.ops.prims.convert_element_type.default(mul_1750, torch.int64);  mul_1750 = None
        _unsafe_index_46 = torch.ops.aten._unsafe_index.Tensor(add_1506, [None, None, unsqueeze_4214, convert_element_type_1229]);  add_1506 = unsqueeze_4214 = convert_element_type_1229 = None
        add_1511 = torch.ops.aten.add.Tensor(add_1504, _unsafe_index_46);  add_1504 = _unsafe_index_46 = None
        relu_464 = torch.ops.aten.relu.default(add_1511);  add_1511 = None
        convolution_521 = torch.ops.aten.convolution.default(relu_439, arg981_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg981_1 = None
        add_1512 = torch.ops.aten.add.Tensor(arg983_1, 1e-05);  arg983_1 = None
        sqrt_521 = torch.ops.aten.sqrt.default(add_1512);  add_1512 = None
        reciprocal_521 = torch.ops.aten.reciprocal.default(sqrt_521);  sqrt_521 = None
        mul_1751 = torch.ops.aten.mul.Tensor(reciprocal_521, 1);  reciprocal_521 = None
        unsqueeze_4215 = torch.ops.aten.unsqueeze.default(arg982_1, -1);  arg982_1 = None
        unsqueeze_4216 = torch.ops.aten.unsqueeze.default(unsqueeze_4215, -1);  unsqueeze_4215 = None
        unsqueeze_4217 = torch.ops.aten.unsqueeze.default(mul_1751, -1);  mul_1751 = None
        unsqueeze_4218 = torch.ops.aten.unsqueeze.default(unsqueeze_4217, -1);  unsqueeze_4217 = None
        sub_521 = torch.ops.aten.sub.Tensor(convolution_521, unsqueeze_4216);  convolution_521 = unsqueeze_4216 = None
        mul_1752 = torch.ops.aten.mul.Tensor(sub_521, unsqueeze_4218);  sub_521 = unsqueeze_4218 = None
        unsqueeze_4219 = torch.ops.aten.unsqueeze.default(arg984_1, -1);  arg984_1 = None
        unsqueeze_4220 = torch.ops.aten.unsqueeze.default(unsqueeze_4219, -1);  unsqueeze_4219 = None
        mul_1753 = torch.ops.aten.mul.Tensor(mul_1752, unsqueeze_4220);  mul_1752 = unsqueeze_4220 = None
        unsqueeze_4221 = torch.ops.aten.unsqueeze.default(arg985_1, -1);  arg985_1 = None
        unsqueeze_4222 = torch.ops.aten.unsqueeze.default(unsqueeze_4221, -1);  unsqueeze_4221 = None
        add_1513 = torch.ops.aten.add.Tensor(mul_1753, unsqueeze_4222);  mul_1753 = unsqueeze_4222 = None
        add_1514 = torch.ops.aten.add.Tensor(add_1513, relu_447);  add_1513 = None
        convolution_522 = torch.ops.aten.convolution.default(relu_455, arg986_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg986_1 = None
        add_1515 = torch.ops.aten.add.Tensor(arg988_1, 1e-05);  arg988_1 = None
        sqrt_522 = torch.ops.aten.sqrt.default(add_1515);  add_1515 = None
        reciprocal_522 = torch.ops.aten.reciprocal.default(sqrt_522);  sqrt_522 = None
        mul_1754 = torch.ops.aten.mul.Tensor(reciprocal_522, 1);  reciprocal_522 = None
        unsqueeze_4223 = torch.ops.aten.unsqueeze.default(arg987_1, -1);  arg987_1 = None
        unsqueeze_4224 = torch.ops.aten.unsqueeze.default(unsqueeze_4223, -1);  unsqueeze_4223 = None
        unsqueeze_4225 = torch.ops.aten.unsqueeze.default(mul_1754, -1);  mul_1754 = None
        unsqueeze_4226 = torch.ops.aten.unsqueeze.default(unsqueeze_4225, -1);  unsqueeze_4225 = None
        sub_522 = torch.ops.aten.sub.Tensor(convolution_522, unsqueeze_4224);  convolution_522 = unsqueeze_4224 = None
        mul_1755 = torch.ops.aten.mul.Tensor(sub_522, unsqueeze_4226);  sub_522 = unsqueeze_4226 = None
        unsqueeze_4227 = torch.ops.aten.unsqueeze.default(arg989_1, -1);  arg989_1 = None
        unsqueeze_4228 = torch.ops.aten.unsqueeze.default(unsqueeze_4227, -1);  unsqueeze_4227 = None
        mul_1756 = torch.ops.aten.mul.Tensor(mul_1755, unsqueeze_4228);  mul_1755 = unsqueeze_4228 = None
        unsqueeze_4229 = torch.ops.aten.unsqueeze.default(arg990_1, -1);  arg990_1 = None
        unsqueeze_4230 = torch.ops.aten.unsqueeze.default(unsqueeze_4229, -1);  unsqueeze_4229 = None
        add_1516 = torch.ops.aten.add.Tensor(mul_1756, unsqueeze_4230);  mul_1756 = unsqueeze_4230 = None
        iota_94 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1757 = torch.ops.aten.mul.Tensor(iota_94, 1);  iota_94 = None
        add_1517 = torch.ops.aten.add.Tensor(mul_1757, 0);  mul_1757 = None
        convert_element_type_1234 = torch.ops.prims.convert_element_type.default(add_1517, torch.float32);  add_1517 = None
        add_1518 = torch.ops.aten.add.Tensor(convert_element_type_1234, 0.0);  convert_element_type_1234 = None
        mul_1758 = torch.ops.aten.mul.Tensor(add_1518, 0.5);  add_1518 = None
        convert_element_type_1235 = torch.ops.prims.convert_element_type.default(mul_1758, torch.int64);  mul_1758 = None
        unsqueeze_4231 = torch.ops.aten.unsqueeze.default(convert_element_type_1235, -1);  convert_element_type_1235 = None
        iota_95 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1759 = torch.ops.aten.mul.Tensor(iota_95, 1);  iota_95 = None
        add_1519 = torch.ops.aten.add.Tensor(mul_1759, 0);  mul_1759 = None
        convert_element_type_1236 = torch.ops.prims.convert_element_type.default(add_1519, torch.float32);  add_1519 = None
        add_1520 = torch.ops.aten.add.Tensor(convert_element_type_1236, 0.0);  convert_element_type_1236 = None
        mul_1760 = torch.ops.aten.mul.Tensor(add_1520, 0.5);  add_1520 = None
        convert_element_type_1237 = torch.ops.prims.convert_element_type.default(mul_1760, torch.int64);  mul_1760 = None
        _unsafe_index_47 = torch.ops.aten._unsafe_index.Tensor(add_1516, [None, None, unsqueeze_4231, convert_element_type_1237]);  add_1516 = unsqueeze_4231 = convert_element_type_1237 = None
        add_1521 = torch.ops.aten.add.Tensor(add_1514, _unsafe_index_47);  add_1514 = _unsafe_index_47 = None
        convolution_523 = torch.ops.aten.convolution.default(relu_463, arg991_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg991_1 = None
        add_1522 = torch.ops.aten.add.Tensor(arg993_1, 1e-05);  arg993_1 = None
        sqrt_523 = torch.ops.aten.sqrt.default(add_1522);  add_1522 = None
        reciprocal_523 = torch.ops.aten.reciprocal.default(sqrt_523);  sqrt_523 = None
        mul_1761 = torch.ops.aten.mul.Tensor(reciprocal_523, 1);  reciprocal_523 = None
        unsqueeze_4232 = torch.ops.aten.unsqueeze.default(arg992_1, -1);  arg992_1 = None
        unsqueeze_4233 = torch.ops.aten.unsqueeze.default(unsqueeze_4232, -1);  unsqueeze_4232 = None
        unsqueeze_4234 = torch.ops.aten.unsqueeze.default(mul_1761, -1);  mul_1761 = None
        unsqueeze_4235 = torch.ops.aten.unsqueeze.default(unsqueeze_4234, -1);  unsqueeze_4234 = None
        sub_523 = torch.ops.aten.sub.Tensor(convolution_523, unsqueeze_4233);  convolution_523 = unsqueeze_4233 = None
        mul_1762 = torch.ops.aten.mul.Tensor(sub_523, unsqueeze_4235);  sub_523 = unsqueeze_4235 = None
        unsqueeze_4236 = torch.ops.aten.unsqueeze.default(arg994_1, -1);  arg994_1 = None
        unsqueeze_4237 = torch.ops.aten.unsqueeze.default(unsqueeze_4236, -1);  unsqueeze_4236 = None
        mul_1763 = torch.ops.aten.mul.Tensor(mul_1762, unsqueeze_4237);  mul_1762 = unsqueeze_4237 = None
        unsqueeze_4238 = torch.ops.aten.unsqueeze.default(arg995_1, -1);  arg995_1 = None
        unsqueeze_4239 = torch.ops.aten.unsqueeze.default(unsqueeze_4238, -1);  unsqueeze_4238 = None
        add_1523 = torch.ops.aten.add.Tensor(mul_1763, unsqueeze_4239);  mul_1763 = unsqueeze_4239 = None
        iota_96 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1764 = torch.ops.aten.mul.Tensor(iota_96, 1);  iota_96 = None
        add_1524 = torch.ops.aten.add.Tensor(mul_1764, 0);  mul_1764 = None
        convert_element_type_1240 = torch.ops.prims.convert_element_type.default(add_1524, torch.float32);  add_1524 = None
        add_1525 = torch.ops.aten.add.Tensor(convert_element_type_1240, 0.0);  convert_element_type_1240 = None
        mul_1765 = torch.ops.aten.mul.Tensor(add_1525, 0.25);  add_1525 = None
        convert_element_type_1241 = torch.ops.prims.convert_element_type.default(mul_1765, torch.int64);  mul_1765 = None
        unsqueeze_4240 = torch.ops.aten.unsqueeze.default(convert_element_type_1241, -1);  convert_element_type_1241 = None
        iota_97 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1766 = torch.ops.aten.mul.Tensor(iota_97, 1);  iota_97 = None
        add_1526 = torch.ops.aten.add.Tensor(mul_1766, 0);  mul_1766 = None
        convert_element_type_1242 = torch.ops.prims.convert_element_type.default(add_1526, torch.float32);  add_1526 = None
        add_1527 = torch.ops.aten.add.Tensor(convert_element_type_1242, 0.0);  convert_element_type_1242 = None
        mul_1767 = torch.ops.aten.mul.Tensor(add_1527, 0.25);  add_1527 = None
        convert_element_type_1243 = torch.ops.prims.convert_element_type.default(mul_1767, torch.int64);  mul_1767 = None
        _unsafe_index_48 = torch.ops.aten._unsafe_index.Tensor(add_1523, [None, None, unsqueeze_4240, convert_element_type_1243]);  add_1523 = unsqueeze_4240 = convert_element_type_1243 = None
        add_1528 = torch.ops.aten.add.Tensor(add_1521, _unsafe_index_48);  add_1521 = _unsafe_index_48 = None
        relu_465 = torch.ops.aten.relu.default(add_1528);  add_1528 = None
        convolution_524 = torch.ops.aten.convolution.default(relu_439, arg996_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg996_1 = None
        add_1529 = torch.ops.aten.add.Tensor(arg998_1, 1e-05);  arg998_1 = None
        sqrt_524 = torch.ops.aten.sqrt.default(add_1529);  add_1529 = None
        reciprocal_524 = torch.ops.aten.reciprocal.default(sqrt_524);  sqrt_524 = None
        mul_1768 = torch.ops.aten.mul.Tensor(reciprocal_524, 1);  reciprocal_524 = None
        unsqueeze_4241 = torch.ops.aten.unsqueeze.default(arg997_1, -1);  arg997_1 = None
        unsqueeze_4242 = torch.ops.aten.unsqueeze.default(unsqueeze_4241, -1);  unsqueeze_4241 = None
        unsqueeze_4243 = torch.ops.aten.unsqueeze.default(mul_1768, -1);  mul_1768 = None
        unsqueeze_4244 = torch.ops.aten.unsqueeze.default(unsqueeze_4243, -1);  unsqueeze_4243 = None
        sub_524 = torch.ops.aten.sub.Tensor(convolution_524, unsqueeze_4242);  convolution_524 = unsqueeze_4242 = None
        mul_1769 = torch.ops.aten.mul.Tensor(sub_524, unsqueeze_4244);  sub_524 = unsqueeze_4244 = None
        unsqueeze_4245 = torch.ops.aten.unsqueeze.default(arg999_1, -1);  arg999_1 = None
        unsqueeze_4246 = torch.ops.aten.unsqueeze.default(unsqueeze_4245, -1);  unsqueeze_4245 = None
        mul_1770 = torch.ops.aten.mul.Tensor(mul_1769, unsqueeze_4246);  mul_1769 = unsqueeze_4246 = None
        unsqueeze_4247 = torch.ops.aten.unsqueeze.default(arg1000_1, -1);  arg1000_1 = None
        unsqueeze_4248 = torch.ops.aten.unsqueeze.default(unsqueeze_4247, -1);  unsqueeze_4247 = None
        add_1530 = torch.ops.aten.add.Tensor(mul_1770, unsqueeze_4248);  mul_1770 = unsqueeze_4248 = None
        relu_466 = torch.ops.aten.relu.default(add_1530);  add_1530 = None
        convolution_525 = torch.ops.aten.convolution.default(relu_466, arg1001_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_466 = arg1001_1 = None
        add_1531 = torch.ops.aten.add.Tensor(arg1003_1, 1e-05);  arg1003_1 = None
        sqrt_525 = torch.ops.aten.sqrt.default(add_1531);  add_1531 = None
        reciprocal_525 = torch.ops.aten.reciprocal.default(sqrt_525);  sqrt_525 = None
        mul_1771 = torch.ops.aten.mul.Tensor(reciprocal_525, 1);  reciprocal_525 = None
        unsqueeze_4249 = torch.ops.aten.unsqueeze.default(arg1002_1, -1);  arg1002_1 = None
        unsqueeze_4250 = torch.ops.aten.unsqueeze.default(unsqueeze_4249, -1);  unsqueeze_4249 = None
        unsqueeze_4251 = torch.ops.aten.unsqueeze.default(mul_1771, -1);  mul_1771 = None
        unsqueeze_4252 = torch.ops.aten.unsqueeze.default(unsqueeze_4251, -1);  unsqueeze_4251 = None
        sub_525 = torch.ops.aten.sub.Tensor(convolution_525, unsqueeze_4250);  convolution_525 = unsqueeze_4250 = None
        mul_1772 = torch.ops.aten.mul.Tensor(sub_525, unsqueeze_4252);  sub_525 = unsqueeze_4252 = None
        unsqueeze_4253 = torch.ops.aten.unsqueeze.default(arg1004_1, -1);  arg1004_1 = None
        unsqueeze_4254 = torch.ops.aten.unsqueeze.default(unsqueeze_4253, -1);  unsqueeze_4253 = None
        mul_1773 = torch.ops.aten.mul.Tensor(mul_1772, unsqueeze_4254);  mul_1772 = unsqueeze_4254 = None
        unsqueeze_4255 = torch.ops.aten.unsqueeze.default(arg1005_1, -1);  arg1005_1 = None
        unsqueeze_4256 = torch.ops.aten.unsqueeze.default(unsqueeze_4255, -1);  unsqueeze_4255 = None
        add_1532 = torch.ops.aten.add.Tensor(mul_1773, unsqueeze_4256);  mul_1773 = unsqueeze_4256 = None
        convolution_526 = torch.ops.aten.convolution.default(relu_447, arg1006_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1006_1 = None
        add_1533 = torch.ops.aten.add.Tensor(arg1008_1, 1e-05);  arg1008_1 = None
        sqrt_526 = torch.ops.aten.sqrt.default(add_1533);  add_1533 = None
        reciprocal_526 = torch.ops.aten.reciprocal.default(sqrt_526);  sqrt_526 = None
        mul_1774 = torch.ops.aten.mul.Tensor(reciprocal_526, 1);  reciprocal_526 = None
        unsqueeze_4257 = torch.ops.aten.unsqueeze.default(arg1007_1, -1);  arg1007_1 = None
        unsqueeze_4258 = torch.ops.aten.unsqueeze.default(unsqueeze_4257, -1);  unsqueeze_4257 = None
        unsqueeze_4259 = torch.ops.aten.unsqueeze.default(mul_1774, -1);  mul_1774 = None
        unsqueeze_4260 = torch.ops.aten.unsqueeze.default(unsqueeze_4259, -1);  unsqueeze_4259 = None
        sub_526 = torch.ops.aten.sub.Tensor(convolution_526, unsqueeze_4258);  convolution_526 = unsqueeze_4258 = None
        mul_1775 = torch.ops.aten.mul.Tensor(sub_526, unsqueeze_4260);  sub_526 = unsqueeze_4260 = None
        unsqueeze_4261 = torch.ops.aten.unsqueeze.default(arg1009_1, -1);  arg1009_1 = None
        unsqueeze_4262 = torch.ops.aten.unsqueeze.default(unsqueeze_4261, -1);  unsqueeze_4261 = None
        mul_1776 = torch.ops.aten.mul.Tensor(mul_1775, unsqueeze_4262);  mul_1775 = unsqueeze_4262 = None
        unsqueeze_4263 = torch.ops.aten.unsqueeze.default(arg1010_1, -1);  arg1010_1 = None
        unsqueeze_4264 = torch.ops.aten.unsqueeze.default(unsqueeze_4263, -1);  unsqueeze_4263 = None
        add_1534 = torch.ops.aten.add.Tensor(mul_1776, unsqueeze_4264);  mul_1776 = unsqueeze_4264 = None
        add_1535 = torch.ops.aten.add.Tensor(add_1532, add_1534);  add_1532 = add_1534 = None
        add_1536 = torch.ops.aten.add.Tensor(add_1535, relu_455);  add_1535 = None
        convolution_527 = torch.ops.aten.convolution.default(relu_463, arg1011_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1011_1 = None
        add_1537 = torch.ops.aten.add.Tensor(arg1013_1, 1e-05);  arg1013_1 = None
        sqrt_527 = torch.ops.aten.sqrt.default(add_1537);  add_1537 = None
        reciprocal_527 = torch.ops.aten.reciprocal.default(sqrt_527);  sqrt_527 = None
        mul_1777 = torch.ops.aten.mul.Tensor(reciprocal_527, 1);  reciprocal_527 = None
        unsqueeze_4265 = torch.ops.aten.unsqueeze.default(arg1012_1, -1);  arg1012_1 = None
        unsqueeze_4266 = torch.ops.aten.unsqueeze.default(unsqueeze_4265, -1);  unsqueeze_4265 = None
        unsqueeze_4267 = torch.ops.aten.unsqueeze.default(mul_1777, -1);  mul_1777 = None
        unsqueeze_4268 = torch.ops.aten.unsqueeze.default(unsqueeze_4267, -1);  unsqueeze_4267 = None
        sub_527 = torch.ops.aten.sub.Tensor(convolution_527, unsqueeze_4266);  convolution_527 = unsqueeze_4266 = None
        mul_1778 = torch.ops.aten.mul.Tensor(sub_527, unsqueeze_4268);  sub_527 = unsqueeze_4268 = None
        unsqueeze_4269 = torch.ops.aten.unsqueeze.default(arg1014_1, -1);  arg1014_1 = None
        unsqueeze_4270 = torch.ops.aten.unsqueeze.default(unsqueeze_4269, -1);  unsqueeze_4269 = None
        mul_1779 = torch.ops.aten.mul.Tensor(mul_1778, unsqueeze_4270);  mul_1778 = unsqueeze_4270 = None
        unsqueeze_4271 = torch.ops.aten.unsqueeze.default(arg1015_1, -1);  arg1015_1 = None
        unsqueeze_4272 = torch.ops.aten.unsqueeze.default(unsqueeze_4271, -1);  unsqueeze_4271 = None
        add_1538 = torch.ops.aten.add.Tensor(mul_1779, unsqueeze_4272);  mul_1779 = unsqueeze_4272 = None
        iota_98 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1780 = torch.ops.aten.mul.Tensor(iota_98, 1);  iota_98 = None
        add_1539 = torch.ops.aten.add.Tensor(mul_1780, 0);  mul_1780 = None
        convert_element_type_1252 = torch.ops.prims.convert_element_type.default(add_1539, torch.float32);  add_1539 = None
        add_1540 = torch.ops.aten.add.Tensor(convert_element_type_1252, 0.0);  convert_element_type_1252 = None
        mul_1781 = torch.ops.aten.mul.Tensor(add_1540, 0.5);  add_1540 = None
        convert_element_type_1253 = torch.ops.prims.convert_element_type.default(mul_1781, torch.int64);  mul_1781 = None
        unsqueeze_4273 = torch.ops.aten.unsqueeze.default(convert_element_type_1253, -1);  convert_element_type_1253 = None
        iota_99 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1782 = torch.ops.aten.mul.Tensor(iota_99, 1);  iota_99 = None
        add_1541 = torch.ops.aten.add.Tensor(mul_1782, 0);  mul_1782 = None
        convert_element_type_1254 = torch.ops.prims.convert_element_type.default(add_1541, torch.float32);  add_1541 = None
        add_1542 = torch.ops.aten.add.Tensor(convert_element_type_1254, 0.0);  convert_element_type_1254 = None
        mul_1783 = torch.ops.aten.mul.Tensor(add_1542, 0.5);  add_1542 = None
        convert_element_type_1255 = torch.ops.prims.convert_element_type.default(mul_1783, torch.int64);  mul_1783 = None
        _unsafe_index_49 = torch.ops.aten._unsafe_index.Tensor(add_1538, [None, None, unsqueeze_4273, convert_element_type_1255]);  add_1538 = unsqueeze_4273 = convert_element_type_1255 = None
        add_1543 = torch.ops.aten.add.Tensor(add_1536, _unsafe_index_49);  add_1536 = _unsafe_index_49 = None
        relu_467 = torch.ops.aten.relu.default(add_1543);  add_1543 = None
        convolution_528 = torch.ops.aten.convolution.default(relu_439, arg1016_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_439 = arg1016_1 = None
        add_1544 = torch.ops.aten.add.Tensor(arg1018_1, 1e-05);  arg1018_1 = None
        sqrt_528 = torch.ops.aten.sqrt.default(add_1544);  add_1544 = None
        reciprocal_528 = torch.ops.aten.reciprocal.default(sqrt_528);  sqrt_528 = None
        mul_1784 = torch.ops.aten.mul.Tensor(reciprocal_528, 1);  reciprocal_528 = None
        unsqueeze_4274 = torch.ops.aten.unsqueeze.default(arg1017_1, -1);  arg1017_1 = None
        unsqueeze_4275 = torch.ops.aten.unsqueeze.default(unsqueeze_4274, -1);  unsqueeze_4274 = None
        unsqueeze_4276 = torch.ops.aten.unsqueeze.default(mul_1784, -1);  mul_1784 = None
        unsqueeze_4277 = torch.ops.aten.unsqueeze.default(unsqueeze_4276, -1);  unsqueeze_4276 = None
        sub_528 = torch.ops.aten.sub.Tensor(convolution_528, unsqueeze_4275);  convolution_528 = unsqueeze_4275 = None
        mul_1785 = torch.ops.aten.mul.Tensor(sub_528, unsqueeze_4277);  sub_528 = unsqueeze_4277 = None
        unsqueeze_4278 = torch.ops.aten.unsqueeze.default(arg1019_1, -1);  arg1019_1 = None
        unsqueeze_4279 = torch.ops.aten.unsqueeze.default(unsqueeze_4278, -1);  unsqueeze_4278 = None
        mul_1786 = torch.ops.aten.mul.Tensor(mul_1785, unsqueeze_4279);  mul_1785 = unsqueeze_4279 = None
        unsqueeze_4280 = torch.ops.aten.unsqueeze.default(arg1020_1, -1);  arg1020_1 = None
        unsqueeze_4281 = torch.ops.aten.unsqueeze.default(unsqueeze_4280, -1);  unsqueeze_4280 = None
        add_1545 = torch.ops.aten.add.Tensor(mul_1786, unsqueeze_4281);  mul_1786 = unsqueeze_4281 = None
        relu_468 = torch.ops.aten.relu.default(add_1545);  add_1545 = None
        convolution_529 = torch.ops.aten.convolution.default(relu_468, arg1021_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_468 = arg1021_1 = None
        add_1546 = torch.ops.aten.add.Tensor(arg1023_1, 1e-05);  arg1023_1 = None
        sqrt_529 = torch.ops.aten.sqrt.default(add_1546);  add_1546 = None
        reciprocal_529 = torch.ops.aten.reciprocal.default(sqrt_529);  sqrt_529 = None
        mul_1787 = torch.ops.aten.mul.Tensor(reciprocal_529, 1);  reciprocal_529 = None
        unsqueeze_4282 = torch.ops.aten.unsqueeze.default(arg1022_1, -1);  arg1022_1 = None
        unsqueeze_4283 = torch.ops.aten.unsqueeze.default(unsqueeze_4282, -1);  unsqueeze_4282 = None
        unsqueeze_4284 = torch.ops.aten.unsqueeze.default(mul_1787, -1);  mul_1787 = None
        unsqueeze_4285 = torch.ops.aten.unsqueeze.default(unsqueeze_4284, -1);  unsqueeze_4284 = None
        sub_529 = torch.ops.aten.sub.Tensor(convolution_529, unsqueeze_4283);  convolution_529 = unsqueeze_4283 = None
        mul_1788 = torch.ops.aten.mul.Tensor(sub_529, unsqueeze_4285);  sub_529 = unsqueeze_4285 = None
        unsqueeze_4286 = torch.ops.aten.unsqueeze.default(arg1024_1, -1);  arg1024_1 = None
        unsqueeze_4287 = torch.ops.aten.unsqueeze.default(unsqueeze_4286, -1);  unsqueeze_4286 = None
        mul_1789 = torch.ops.aten.mul.Tensor(mul_1788, unsqueeze_4287);  mul_1788 = unsqueeze_4287 = None
        unsqueeze_4288 = torch.ops.aten.unsqueeze.default(arg1025_1, -1);  arg1025_1 = None
        unsqueeze_4289 = torch.ops.aten.unsqueeze.default(unsqueeze_4288, -1);  unsqueeze_4288 = None
        add_1547 = torch.ops.aten.add.Tensor(mul_1789, unsqueeze_4289);  mul_1789 = unsqueeze_4289 = None
        relu_469 = torch.ops.aten.relu.default(add_1547);  add_1547 = None
        convolution_530 = torch.ops.aten.convolution.default(relu_469, arg1026_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_469 = arg1026_1 = None
        add_1548 = torch.ops.aten.add.Tensor(arg1028_1, 1e-05);  arg1028_1 = None
        sqrt_530 = torch.ops.aten.sqrt.default(add_1548);  add_1548 = None
        reciprocal_530 = torch.ops.aten.reciprocal.default(sqrt_530);  sqrt_530 = None
        mul_1790 = torch.ops.aten.mul.Tensor(reciprocal_530, 1);  reciprocal_530 = None
        unsqueeze_4290 = torch.ops.aten.unsqueeze.default(arg1027_1, -1);  arg1027_1 = None
        unsqueeze_4291 = torch.ops.aten.unsqueeze.default(unsqueeze_4290, -1);  unsqueeze_4290 = None
        unsqueeze_4292 = torch.ops.aten.unsqueeze.default(mul_1790, -1);  mul_1790 = None
        unsqueeze_4293 = torch.ops.aten.unsqueeze.default(unsqueeze_4292, -1);  unsqueeze_4292 = None
        sub_530 = torch.ops.aten.sub.Tensor(convolution_530, unsqueeze_4291);  convolution_530 = unsqueeze_4291 = None
        mul_1791 = torch.ops.aten.mul.Tensor(sub_530, unsqueeze_4293);  sub_530 = unsqueeze_4293 = None
        unsqueeze_4294 = torch.ops.aten.unsqueeze.default(arg1029_1, -1);  arg1029_1 = None
        unsqueeze_4295 = torch.ops.aten.unsqueeze.default(unsqueeze_4294, -1);  unsqueeze_4294 = None
        mul_1792 = torch.ops.aten.mul.Tensor(mul_1791, unsqueeze_4295);  mul_1791 = unsqueeze_4295 = None
        unsqueeze_4296 = torch.ops.aten.unsqueeze.default(arg1030_1, -1);  arg1030_1 = None
        unsqueeze_4297 = torch.ops.aten.unsqueeze.default(unsqueeze_4296, -1);  unsqueeze_4296 = None
        add_1549 = torch.ops.aten.add.Tensor(mul_1792, unsqueeze_4297);  mul_1792 = unsqueeze_4297 = None
        convolution_531 = torch.ops.aten.convolution.default(relu_447, arg1031_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_447 = arg1031_1 = None
        add_1550 = torch.ops.aten.add.Tensor(arg1033_1, 1e-05);  arg1033_1 = None
        sqrt_531 = torch.ops.aten.sqrt.default(add_1550);  add_1550 = None
        reciprocal_531 = torch.ops.aten.reciprocal.default(sqrt_531);  sqrt_531 = None
        mul_1793 = torch.ops.aten.mul.Tensor(reciprocal_531, 1);  reciprocal_531 = None
        unsqueeze_4298 = torch.ops.aten.unsqueeze.default(arg1032_1, -1);  arg1032_1 = None
        unsqueeze_4299 = torch.ops.aten.unsqueeze.default(unsqueeze_4298, -1);  unsqueeze_4298 = None
        unsqueeze_4300 = torch.ops.aten.unsqueeze.default(mul_1793, -1);  mul_1793 = None
        unsqueeze_4301 = torch.ops.aten.unsqueeze.default(unsqueeze_4300, -1);  unsqueeze_4300 = None
        sub_531 = torch.ops.aten.sub.Tensor(convolution_531, unsqueeze_4299);  convolution_531 = unsqueeze_4299 = None
        mul_1794 = torch.ops.aten.mul.Tensor(sub_531, unsqueeze_4301);  sub_531 = unsqueeze_4301 = None
        unsqueeze_4302 = torch.ops.aten.unsqueeze.default(arg1034_1, -1);  arg1034_1 = None
        unsqueeze_4303 = torch.ops.aten.unsqueeze.default(unsqueeze_4302, -1);  unsqueeze_4302 = None
        mul_1795 = torch.ops.aten.mul.Tensor(mul_1794, unsqueeze_4303);  mul_1794 = unsqueeze_4303 = None
        unsqueeze_4304 = torch.ops.aten.unsqueeze.default(arg1035_1, -1);  arg1035_1 = None
        unsqueeze_4305 = torch.ops.aten.unsqueeze.default(unsqueeze_4304, -1);  unsqueeze_4304 = None
        add_1551 = torch.ops.aten.add.Tensor(mul_1795, unsqueeze_4305);  mul_1795 = unsqueeze_4305 = None
        relu_470 = torch.ops.aten.relu.default(add_1551);  add_1551 = None
        convolution_532 = torch.ops.aten.convolution.default(relu_470, arg1036_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_470 = arg1036_1 = None
        add_1552 = torch.ops.aten.add.Tensor(arg1038_1, 1e-05);  arg1038_1 = None
        sqrt_532 = torch.ops.aten.sqrt.default(add_1552);  add_1552 = None
        reciprocal_532 = torch.ops.aten.reciprocal.default(sqrt_532);  sqrt_532 = None
        mul_1796 = torch.ops.aten.mul.Tensor(reciprocal_532, 1);  reciprocal_532 = None
        unsqueeze_4306 = torch.ops.aten.unsqueeze.default(arg1037_1, -1);  arg1037_1 = None
        unsqueeze_4307 = torch.ops.aten.unsqueeze.default(unsqueeze_4306, -1);  unsqueeze_4306 = None
        unsqueeze_4308 = torch.ops.aten.unsqueeze.default(mul_1796, -1);  mul_1796 = None
        unsqueeze_4309 = torch.ops.aten.unsqueeze.default(unsqueeze_4308, -1);  unsqueeze_4308 = None
        sub_532 = torch.ops.aten.sub.Tensor(convolution_532, unsqueeze_4307);  convolution_532 = unsqueeze_4307 = None
        mul_1797 = torch.ops.aten.mul.Tensor(sub_532, unsqueeze_4309);  sub_532 = unsqueeze_4309 = None
        unsqueeze_4310 = torch.ops.aten.unsqueeze.default(arg1039_1, -1);  arg1039_1 = None
        unsqueeze_4311 = torch.ops.aten.unsqueeze.default(unsqueeze_4310, -1);  unsqueeze_4310 = None
        mul_1798 = torch.ops.aten.mul.Tensor(mul_1797, unsqueeze_4311);  mul_1797 = unsqueeze_4311 = None
        unsqueeze_4312 = torch.ops.aten.unsqueeze.default(arg1040_1, -1);  arg1040_1 = None
        unsqueeze_4313 = torch.ops.aten.unsqueeze.default(unsqueeze_4312, -1);  unsqueeze_4312 = None
        add_1553 = torch.ops.aten.add.Tensor(mul_1798, unsqueeze_4313);  mul_1798 = unsqueeze_4313 = None
        add_1554 = torch.ops.aten.add.Tensor(add_1549, add_1553);  add_1549 = add_1553 = None
        convolution_533 = torch.ops.aten.convolution.default(relu_455, arg1041_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_455 = arg1041_1 = None
        add_1555 = torch.ops.aten.add.Tensor(arg1043_1, 1e-05);  arg1043_1 = None
        sqrt_533 = torch.ops.aten.sqrt.default(add_1555);  add_1555 = None
        reciprocal_533 = torch.ops.aten.reciprocal.default(sqrt_533);  sqrt_533 = None
        mul_1799 = torch.ops.aten.mul.Tensor(reciprocal_533, 1);  reciprocal_533 = None
        unsqueeze_4314 = torch.ops.aten.unsqueeze.default(arg1042_1, -1);  arg1042_1 = None
        unsqueeze_4315 = torch.ops.aten.unsqueeze.default(unsqueeze_4314, -1);  unsqueeze_4314 = None
        unsqueeze_4316 = torch.ops.aten.unsqueeze.default(mul_1799, -1);  mul_1799 = None
        unsqueeze_4317 = torch.ops.aten.unsqueeze.default(unsqueeze_4316, -1);  unsqueeze_4316 = None
        sub_533 = torch.ops.aten.sub.Tensor(convolution_533, unsqueeze_4315);  convolution_533 = unsqueeze_4315 = None
        mul_1800 = torch.ops.aten.mul.Tensor(sub_533, unsqueeze_4317);  sub_533 = unsqueeze_4317 = None
        unsqueeze_4318 = torch.ops.aten.unsqueeze.default(arg1044_1, -1);  arg1044_1 = None
        unsqueeze_4319 = torch.ops.aten.unsqueeze.default(unsqueeze_4318, -1);  unsqueeze_4318 = None
        mul_1801 = torch.ops.aten.mul.Tensor(mul_1800, unsqueeze_4319);  mul_1800 = unsqueeze_4319 = None
        unsqueeze_4320 = torch.ops.aten.unsqueeze.default(arg1045_1, -1);  arg1045_1 = None
        unsqueeze_4321 = torch.ops.aten.unsqueeze.default(unsqueeze_4320, -1);  unsqueeze_4320 = None
        add_1556 = torch.ops.aten.add.Tensor(mul_1801, unsqueeze_4321);  mul_1801 = unsqueeze_4321 = None
        add_1557 = torch.ops.aten.add.Tensor(add_1554, add_1556);  add_1554 = add_1556 = None
        add_1558 = torch.ops.aten.add.Tensor(add_1557, relu_463);  add_1557 = relu_463 = None
        relu_471 = torch.ops.aten.relu.default(add_1558);  add_1558 = None
        convolution_534 = torch.ops.aten.convolution.default(relu_464, arg1046_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1046_1 = None
        add_1559 = torch.ops.aten.add.Tensor(arg1048_1, 1e-05);  arg1048_1 = None
        sqrt_534 = torch.ops.aten.sqrt.default(add_1559);  add_1559 = None
        reciprocal_534 = torch.ops.aten.reciprocal.default(sqrt_534);  sqrt_534 = None
        mul_1802 = torch.ops.aten.mul.Tensor(reciprocal_534, 1);  reciprocal_534 = None
        unsqueeze_4322 = torch.ops.aten.unsqueeze.default(arg1047_1, -1);  arg1047_1 = None
        unsqueeze_4323 = torch.ops.aten.unsqueeze.default(unsqueeze_4322, -1);  unsqueeze_4322 = None
        unsqueeze_4324 = torch.ops.aten.unsqueeze.default(mul_1802, -1);  mul_1802 = None
        unsqueeze_4325 = torch.ops.aten.unsqueeze.default(unsqueeze_4324, -1);  unsqueeze_4324 = None
        sub_534 = torch.ops.aten.sub.Tensor(convolution_534, unsqueeze_4323);  convolution_534 = unsqueeze_4323 = None
        mul_1803 = torch.ops.aten.mul.Tensor(sub_534, unsqueeze_4325);  sub_534 = unsqueeze_4325 = None
        unsqueeze_4326 = torch.ops.aten.unsqueeze.default(arg1049_1, -1);  arg1049_1 = None
        unsqueeze_4327 = torch.ops.aten.unsqueeze.default(unsqueeze_4326, -1);  unsqueeze_4326 = None
        mul_1804 = torch.ops.aten.mul.Tensor(mul_1803, unsqueeze_4327);  mul_1803 = unsqueeze_4327 = None
        unsqueeze_4328 = torch.ops.aten.unsqueeze.default(arg1050_1, -1);  arg1050_1 = None
        unsqueeze_4329 = torch.ops.aten.unsqueeze.default(unsqueeze_4328, -1);  unsqueeze_4328 = None
        add_1560 = torch.ops.aten.add.Tensor(mul_1804, unsqueeze_4329);  mul_1804 = unsqueeze_4329 = None
        relu_472 = torch.ops.aten.relu.default(add_1560);  add_1560 = None
        convolution_535 = torch.ops.aten.convolution.default(relu_472, arg1051_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_472 = arg1051_1 = None
        add_1561 = torch.ops.aten.add.Tensor(arg1053_1, 1e-05);  arg1053_1 = None
        sqrt_535 = torch.ops.aten.sqrt.default(add_1561);  add_1561 = None
        reciprocal_535 = torch.ops.aten.reciprocal.default(sqrt_535);  sqrt_535 = None
        mul_1805 = torch.ops.aten.mul.Tensor(reciprocal_535, 1);  reciprocal_535 = None
        unsqueeze_4330 = torch.ops.aten.unsqueeze.default(arg1052_1, -1);  arg1052_1 = None
        unsqueeze_4331 = torch.ops.aten.unsqueeze.default(unsqueeze_4330, -1);  unsqueeze_4330 = None
        unsqueeze_4332 = torch.ops.aten.unsqueeze.default(mul_1805, -1);  mul_1805 = None
        unsqueeze_4333 = torch.ops.aten.unsqueeze.default(unsqueeze_4332, -1);  unsqueeze_4332 = None
        sub_535 = torch.ops.aten.sub.Tensor(convolution_535, unsqueeze_4331);  convolution_535 = unsqueeze_4331 = None
        mul_1806 = torch.ops.aten.mul.Tensor(sub_535, unsqueeze_4333);  sub_535 = unsqueeze_4333 = None
        unsqueeze_4334 = torch.ops.aten.unsqueeze.default(arg1054_1, -1);  arg1054_1 = None
        unsqueeze_4335 = torch.ops.aten.unsqueeze.default(unsqueeze_4334, -1);  unsqueeze_4334 = None
        mul_1807 = torch.ops.aten.mul.Tensor(mul_1806, unsqueeze_4335);  mul_1806 = unsqueeze_4335 = None
        unsqueeze_4336 = torch.ops.aten.unsqueeze.default(arg1055_1, -1);  arg1055_1 = None
        unsqueeze_4337 = torch.ops.aten.unsqueeze.default(unsqueeze_4336, -1);  unsqueeze_4336 = None
        add_1562 = torch.ops.aten.add.Tensor(mul_1807, unsqueeze_4337);  mul_1807 = unsqueeze_4337 = None
        add_1563 = torch.ops.aten.add.Tensor(add_1562, relu_464);  add_1562 = relu_464 = None
        relu_473 = torch.ops.aten.relu.default(add_1563);  add_1563 = None
        convolution_536 = torch.ops.aten.convolution.default(relu_473, arg1056_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1056_1 = None
        add_1564 = torch.ops.aten.add.Tensor(arg1058_1, 1e-05);  arg1058_1 = None
        sqrt_536 = torch.ops.aten.sqrt.default(add_1564);  add_1564 = None
        reciprocal_536 = torch.ops.aten.reciprocal.default(sqrt_536);  sqrt_536 = None
        mul_1808 = torch.ops.aten.mul.Tensor(reciprocal_536, 1);  reciprocal_536 = None
        unsqueeze_4338 = torch.ops.aten.unsqueeze.default(arg1057_1, -1);  arg1057_1 = None
        unsqueeze_4339 = torch.ops.aten.unsqueeze.default(unsqueeze_4338, -1);  unsqueeze_4338 = None
        unsqueeze_4340 = torch.ops.aten.unsqueeze.default(mul_1808, -1);  mul_1808 = None
        unsqueeze_4341 = torch.ops.aten.unsqueeze.default(unsqueeze_4340, -1);  unsqueeze_4340 = None
        sub_536 = torch.ops.aten.sub.Tensor(convolution_536, unsqueeze_4339);  convolution_536 = unsqueeze_4339 = None
        mul_1809 = torch.ops.aten.mul.Tensor(sub_536, unsqueeze_4341);  sub_536 = unsqueeze_4341 = None
        unsqueeze_4342 = torch.ops.aten.unsqueeze.default(arg1059_1, -1);  arg1059_1 = None
        unsqueeze_4343 = torch.ops.aten.unsqueeze.default(unsqueeze_4342, -1);  unsqueeze_4342 = None
        mul_1810 = torch.ops.aten.mul.Tensor(mul_1809, unsqueeze_4343);  mul_1809 = unsqueeze_4343 = None
        unsqueeze_4344 = torch.ops.aten.unsqueeze.default(arg1060_1, -1);  arg1060_1 = None
        unsqueeze_4345 = torch.ops.aten.unsqueeze.default(unsqueeze_4344, -1);  unsqueeze_4344 = None
        add_1565 = torch.ops.aten.add.Tensor(mul_1810, unsqueeze_4345);  mul_1810 = unsqueeze_4345 = None
        relu_474 = torch.ops.aten.relu.default(add_1565);  add_1565 = None
        convolution_537 = torch.ops.aten.convolution.default(relu_474, arg1061_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_474 = arg1061_1 = None
        add_1566 = torch.ops.aten.add.Tensor(arg1063_1, 1e-05);  arg1063_1 = None
        sqrt_537 = torch.ops.aten.sqrt.default(add_1566);  add_1566 = None
        reciprocal_537 = torch.ops.aten.reciprocal.default(sqrt_537);  sqrt_537 = None
        mul_1811 = torch.ops.aten.mul.Tensor(reciprocal_537, 1);  reciprocal_537 = None
        unsqueeze_4346 = torch.ops.aten.unsqueeze.default(arg1062_1, -1);  arg1062_1 = None
        unsqueeze_4347 = torch.ops.aten.unsqueeze.default(unsqueeze_4346, -1);  unsqueeze_4346 = None
        unsqueeze_4348 = torch.ops.aten.unsqueeze.default(mul_1811, -1);  mul_1811 = None
        unsqueeze_4349 = torch.ops.aten.unsqueeze.default(unsqueeze_4348, -1);  unsqueeze_4348 = None
        sub_537 = torch.ops.aten.sub.Tensor(convolution_537, unsqueeze_4347);  convolution_537 = unsqueeze_4347 = None
        mul_1812 = torch.ops.aten.mul.Tensor(sub_537, unsqueeze_4349);  sub_537 = unsqueeze_4349 = None
        unsqueeze_4350 = torch.ops.aten.unsqueeze.default(arg1064_1, -1);  arg1064_1 = None
        unsqueeze_4351 = torch.ops.aten.unsqueeze.default(unsqueeze_4350, -1);  unsqueeze_4350 = None
        mul_1813 = torch.ops.aten.mul.Tensor(mul_1812, unsqueeze_4351);  mul_1812 = unsqueeze_4351 = None
        unsqueeze_4352 = torch.ops.aten.unsqueeze.default(arg1065_1, -1);  arg1065_1 = None
        unsqueeze_4353 = torch.ops.aten.unsqueeze.default(unsqueeze_4352, -1);  unsqueeze_4352 = None
        add_1567 = torch.ops.aten.add.Tensor(mul_1813, unsqueeze_4353);  mul_1813 = unsqueeze_4353 = None
        add_1568 = torch.ops.aten.add.Tensor(add_1567, relu_473);  add_1567 = relu_473 = None
        relu_475 = torch.ops.aten.relu.default(add_1568);  add_1568 = None
        convolution_538 = torch.ops.aten.convolution.default(relu_475, arg1066_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1066_1 = None
        add_1569 = torch.ops.aten.add.Tensor(arg1068_1, 1e-05);  arg1068_1 = None
        sqrt_538 = torch.ops.aten.sqrt.default(add_1569);  add_1569 = None
        reciprocal_538 = torch.ops.aten.reciprocal.default(sqrt_538);  sqrt_538 = None
        mul_1814 = torch.ops.aten.mul.Tensor(reciprocal_538, 1);  reciprocal_538 = None
        unsqueeze_4354 = torch.ops.aten.unsqueeze.default(arg1067_1, -1);  arg1067_1 = None
        unsqueeze_4355 = torch.ops.aten.unsqueeze.default(unsqueeze_4354, -1);  unsqueeze_4354 = None
        unsqueeze_4356 = torch.ops.aten.unsqueeze.default(mul_1814, -1);  mul_1814 = None
        unsqueeze_4357 = torch.ops.aten.unsqueeze.default(unsqueeze_4356, -1);  unsqueeze_4356 = None
        sub_538 = torch.ops.aten.sub.Tensor(convolution_538, unsqueeze_4355);  convolution_538 = unsqueeze_4355 = None
        mul_1815 = torch.ops.aten.mul.Tensor(sub_538, unsqueeze_4357);  sub_538 = unsqueeze_4357 = None
        unsqueeze_4358 = torch.ops.aten.unsqueeze.default(arg1069_1, -1);  arg1069_1 = None
        unsqueeze_4359 = torch.ops.aten.unsqueeze.default(unsqueeze_4358, -1);  unsqueeze_4358 = None
        mul_1816 = torch.ops.aten.mul.Tensor(mul_1815, unsqueeze_4359);  mul_1815 = unsqueeze_4359 = None
        unsqueeze_4360 = torch.ops.aten.unsqueeze.default(arg1070_1, -1);  arg1070_1 = None
        unsqueeze_4361 = torch.ops.aten.unsqueeze.default(unsqueeze_4360, -1);  unsqueeze_4360 = None
        add_1570 = torch.ops.aten.add.Tensor(mul_1816, unsqueeze_4361);  mul_1816 = unsqueeze_4361 = None
        relu_476 = torch.ops.aten.relu.default(add_1570);  add_1570 = None
        convolution_539 = torch.ops.aten.convolution.default(relu_476, arg1071_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_476 = arg1071_1 = None
        add_1571 = torch.ops.aten.add.Tensor(arg1073_1, 1e-05);  arg1073_1 = None
        sqrt_539 = torch.ops.aten.sqrt.default(add_1571);  add_1571 = None
        reciprocal_539 = torch.ops.aten.reciprocal.default(sqrt_539);  sqrt_539 = None
        mul_1817 = torch.ops.aten.mul.Tensor(reciprocal_539, 1);  reciprocal_539 = None
        unsqueeze_4362 = torch.ops.aten.unsqueeze.default(arg1072_1, -1);  arg1072_1 = None
        unsqueeze_4363 = torch.ops.aten.unsqueeze.default(unsqueeze_4362, -1);  unsqueeze_4362 = None
        unsqueeze_4364 = torch.ops.aten.unsqueeze.default(mul_1817, -1);  mul_1817 = None
        unsqueeze_4365 = torch.ops.aten.unsqueeze.default(unsqueeze_4364, -1);  unsqueeze_4364 = None
        sub_539 = torch.ops.aten.sub.Tensor(convolution_539, unsqueeze_4363);  convolution_539 = unsqueeze_4363 = None
        mul_1818 = torch.ops.aten.mul.Tensor(sub_539, unsqueeze_4365);  sub_539 = unsqueeze_4365 = None
        unsqueeze_4366 = torch.ops.aten.unsqueeze.default(arg1074_1, -1);  arg1074_1 = None
        unsqueeze_4367 = torch.ops.aten.unsqueeze.default(unsqueeze_4366, -1);  unsqueeze_4366 = None
        mul_1819 = torch.ops.aten.mul.Tensor(mul_1818, unsqueeze_4367);  mul_1818 = unsqueeze_4367 = None
        unsqueeze_4368 = torch.ops.aten.unsqueeze.default(arg1075_1, -1);  arg1075_1 = None
        unsqueeze_4369 = torch.ops.aten.unsqueeze.default(unsqueeze_4368, -1);  unsqueeze_4368 = None
        add_1572 = torch.ops.aten.add.Tensor(mul_1819, unsqueeze_4369);  mul_1819 = unsqueeze_4369 = None
        add_1573 = torch.ops.aten.add.Tensor(add_1572, relu_475);  add_1572 = relu_475 = None
        relu_477 = torch.ops.aten.relu.default(add_1573);  add_1573 = None
        convolution_540 = torch.ops.aten.convolution.default(relu_477, arg1076_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1076_1 = None
        add_1574 = torch.ops.aten.add.Tensor(arg1078_1, 1e-05);  arg1078_1 = None
        sqrt_540 = torch.ops.aten.sqrt.default(add_1574);  add_1574 = None
        reciprocal_540 = torch.ops.aten.reciprocal.default(sqrt_540);  sqrt_540 = None
        mul_1820 = torch.ops.aten.mul.Tensor(reciprocal_540, 1);  reciprocal_540 = None
        unsqueeze_4370 = torch.ops.aten.unsqueeze.default(arg1077_1, -1);  arg1077_1 = None
        unsqueeze_4371 = torch.ops.aten.unsqueeze.default(unsqueeze_4370, -1);  unsqueeze_4370 = None
        unsqueeze_4372 = torch.ops.aten.unsqueeze.default(mul_1820, -1);  mul_1820 = None
        unsqueeze_4373 = torch.ops.aten.unsqueeze.default(unsqueeze_4372, -1);  unsqueeze_4372 = None
        sub_540 = torch.ops.aten.sub.Tensor(convolution_540, unsqueeze_4371);  convolution_540 = unsqueeze_4371 = None
        mul_1821 = torch.ops.aten.mul.Tensor(sub_540, unsqueeze_4373);  sub_540 = unsqueeze_4373 = None
        unsqueeze_4374 = torch.ops.aten.unsqueeze.default(arg1079_1, -1);  arg1079_1 = None
        unsqueeze_4375 = torch.ops.aten.unsqueeze.default(unsqueeze_4374, -1);  unsqueeze_4374 = None
        mul_1822 = torch.ops.aten.mul.Tensor(mul_1821, unsqueeze_4375);  mul_1821 = unsqueeze_4375 = None
        unsqueeze_4376 = torch.ops.aten.unsqueeze.default(arg1080_1, -1);  arg1080_1 = None
        unsqueeze_4377 = torch.ops.aten.unsqueeze.default(unsqueeze_4376, -1);  unsqueeze_4376 = None
        add_1575 = torch.ops.aten.add.Tensor(mul_1822, unsqueeze_4377);  mul_1822 = unsqueeze_4377 = None
        relu_478 = torch.ops.aten.relu.default(add_1575);  add_1575 = None
        convolution_541 = torch.ops.aten.convolution.default(relu_478, arg1081_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_478 = arg1081_1 = None
        add_1576 = torch.ops.aten.add.Tensor(arg1083_1, 1e-05);  arg1083_1 = None
        sqrt_541 = torch.ops.aten.sqrt.default(add_1576);  add_1576 = None
        reciprocal_541 = torch.ops.aten.reciprocal.default(sqrt_541);  sqrt_541 = None
        mul_1823 = torch.ops.aten.mul.Tensor(reciprocal_541, 1);  reciprocal_541 = None
        unsqueeze_4378 = torch.ops.aten.unsqueeze.default(arg1082_1, -1);  arg1082_1 = None
        unsqueeze_4379 = torch.ops.aten.unsqueeze.default(unsqueeze_4378, -1);  unsqueeze_4378 = None
        unsqueeze_4380 = torch.ops.aten.unsqueeze.default(mul_1823, -1);  mul_1823 = None
        unsqueeze_4381 = torch.ops.aten.unsqueeze.default(unsqueeze_4380, -1);  unsqueeze_4380 = None
        sub_541 = torch.ops.aten.sub.Tensor(convolution_541, unsqueeze_4379);  convolution_541 = unsqueeze_4379 = None
        mul_1824 = torch.ops.aten.mul.Tensor(sub_541, unsqueeze_4381);  sub_541 = unsqueeze_4381 = None
        unsqueeze_4382 = torch.ops.aten.unsqueeze.default(arg1084_1, -1);  arg1084_1 = None
        unsqueeze_4383 = torch.ops.aten.unsqueeze.default(unsqueeze_4382, -1);  unsqueeze_4382 = None
        mul_1825 = torch.ops.aten.mul.Tensor(mul_1824, unsqueeze_4383);  mul_1824 = unsqueeze_4383 = None
        unsqueeze_4384 = torch.ops.aten.unsqueeze.default(arg1085_1, -1);  arg1085_1 = None
        unsqueeze_4385 = torch.ops.aten.unsqueeze.default(unsqueeze_4384, -1);  unsqueeze_4384 = None
        add_1577 = torch.ops.aten.add.Tensor(mul_1825, unsqueeze_4385);  mul_1825 = unsqueeze_4385 = None
        add_1578 = torch.ops.aten.add.Tensor(add_1577, relu_477);  add_1577 = relu_477 = None
        relu_479 = torch.ops.aten.relu.default(add_1578);  add_1578 = None
        convolution_542 = torch.ops.aten.convolution.default(relu_465, arg1086_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1086_1 = None
        add_1579 = torch.ops.aten.add.Tensor(arg1088_1, 1e-05);  arg1088_1 = None
        sqrt_542 = torch.ops.aten.sqrt.default(add_1579);  add_1579 = None
        reciprocal_542 = torch.ops.aten.reciprocal.default(sqrt_542);  sqrt_542 = None
        mul_1826 = torch.ops.aten.mul.Tensor(reciprocal_542, 1);  reciprocal_542 = None
        unsqueeze_4386 = torch.ops.aten.unsqueeze.default(arg1087_1, -1);  arg1087_1 = None
        unsqueeze_4387 = torch.ops.aten.unsqueeze.default(unsqueeze_4386, -1);  unsqueeze_4386 = None
        unsqueeze_4388 = torch.ops.aten.unsqueeze.default(mul_1826, -1);  mul_1826 = None
        unsqueeze_4389 = torch.ops.aten.unsqueeze.default(unsqueeze_4388, -1);  unsqueeze_4388 = None
        sub_542 = torch.ops.aten.sub.Tensor(convolution_542, unsqueeze_4387);  convolution_542 = unsqueeze_4387 = None
        mul_1827 = torch.ops.aten.mul.Tensor(sub_542, unsqueeze_4389);  sub_542 = unsqueeze_4389 = None
        unsqueeze_4390 = torch.ops.aten.unsqueeze.default(arg1089_1, -1);  arg1089_1 = None
        unsqueeze_4391 = torch.ops.aten.unsqueeze.default(unsqueeze_4390, -1);  unsqueeze_4390 = None
        mul_1828 = torch.ops.aten.mul.Tensor(mul_1827, unsqueeze_4391);  mul_1827 = unsqueeze_4391 = None
        unsqueeze_4392 = torch.ops.aten.unsqueeze.default(arg1090_1, -1);  arg1090_1 = None
        unsqueeze_4393 = torch.ops.aten.unsqueeze.default(unsqueeze_4392, -1);  unsqueeze_4392 = None
        add_1580 = torch.ops.aten.add.Tensor(mul_1828, unsqueeze_4393);  mul_1828 = unsqueeze_4393 = None
        relu_480 = torch.ops.aten.relu.default(add_1580);  add_1580 = None
        convolution_543 = torch.ops.aten.convolution.default(relu_480, arg1091_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_480 = arg1091_1 = None
        add_1581 = torch.ops.aten.add.Tensor(arg1093_1, 1e-05);  arg1093_1 = None
        sqrt_543 = torch.ops.aten.sqrt.default(add_1581);  add_1581 = None
        reciprocal_543 = torch.ops.aten.reciprocal.default(sqrt_543);  sqrt_543 = None
        mul_1829 = torch.ops.aten.mul.Tensor(reciprocal_543, 1);  reciprocal_543 = None
        unsqueeze_4394 = torch.ops.aten.unsqueeze.default(arg1092_1, -1);  arg1092_1 = None
        unsqueeze_4395 = torch.ops.aten.unsqueeze.default(unsqueeze_4394, -1);  unsqueeze_4394 = None
        unsqueeze_4396 = torch.ops.aten.unsqueeze.default(mul_1829, -1);  mul_1829 = None
        unsqueeze_4397 = torch.ops.aten.unsqueeze.default(unsqueeze_4396, -1);  unsqueeze_4396 = None
        sub_543 = torch.ops.aten.sub.Tensor(convolution_543, unsqueeze_4395);  convolution_543 = unsqueeze_4395 = None
        mul_1830 = torch.ops.aten.mul.Tensor(sub_543, unsqueeze_4397);  sub_543 = unsqueeze_4397 = None
        unsqueeze_4398 = torch.ops.aten.unsqueeze.default(arg1094_1, -1);  arg1094_1 = None
        unsqueeze_4399 = torch.ops.aten.unsqueeze.default(unsqueeze_4398, -1);  unsqueeze_4398 = None
        mul_1831 = torch.ops.aten.mul.Tensor(mul_1830, unsqueeze_4399);  mul_1830 = unsqueeze_4399 = None
        unsqueeze_4400 = torch.ops.aten.unsqueeze.default(arg1095_1, -1);  arg1095_1 = None
        unsqueeze_4401 = torch.ops.aten.unsqueeze.default(unsqueeze_4400, -1);  unsqueeze_4400 = None
        add_1582 = torch.ops.aten.add.Tensor(mul_1831, unsqueeze_4401);  mul_1831 = unsqueeze_4401 = None
        add_1583 = torch.ops.aten.add.Tensor(add_1582, relu_465);  add_1582 = relu_465 = None
        relu_481 = torch.ops.aten.relu.default(add_1583);  add_1583 = None
        convolution_544 = torch.ops.aten.convolution.default(relu_481, arg1096_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1096_1 = None
        add_1584 = torch.ops.aten.add.Tensor(arg1098_1, 1e-05);  arg1098_1 = None
        sqrt_544 = torch.ops.aten.sqrt.default(add_1584);  add_1584 = None
        reciprocal_544 = torch.ops.aten.reciprocal.default(sqrt_544);  sqrt_544 = None
        mul_1832 = torch.ops.aten.mul.Tensor(reciprocal_544, 1);  reciprocal_544 = None
        unsqueeze_4402 = torch.ops.aten.unsqueeze.default(arg1097_1, -1);  arg1097_1 = None
        unsqueeze_4403 = torch.ops.aten.unsqueeze.default(unsqueeze_4402, -1);  unsqueeze_4402 = None
        unsqueeze_4404 = torch.ops.aten.unsqueeze.default(mul_1832, -1);  mul_1832 = None
        unsqueeze_4405 = torch.ops.aten.unsqueeze.default(unsqueeze_4404, -1);  unsqueeze_4404 = None
        sub_544 = torch.ops.aten.sub.Tensor(convolution_544, unsqueeze_4403);  convolution_544 = unsqueeze_4403 = None
        mul_1833 = torch.ops.aten.mul.Tensor(sub_544, unsqueeze_4405);  sub_544 = unsqueeze_4405 = None
        unsqueeze_4406 = torch.ops.aten.unsqueeze.default(arg1099_1, -1);  arg1099_1 = None
        unsqueeze_4407 = torch.ops.aten.unsqueeze.default(unsqueeze_4406, -1);  unsqueeze_4406 = None
        mul_1834 = torch.ops.aten.mul.Tensor(mul_1833, unsqueeze_4407);  mul_1833 = unsqueeze_4407 = None
        unsqueeze_4408 = torch.ops.aten.unsqueeze.default(arg1100_1, -1);  arg1100_1 = None
        unsqueeze_4409 = torch.ops.aten.unsqueeze.default(unsqueeze_4408, -1);  unsqueeze_4408 = None
        add_1585 = torch.ops.aten.add.Tensor(mul_1834, unsqueeze_4409);  mul_1834 = unsqueeze_4409 = None
        relu_482 = torch.ops.aten.relu.default(add_1585);  add_1585 = None
        convolution_545 = torch.ops.aten.convolution.default(relu_482, arg1101_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_482 = arg1101_1 = None
        add_1586 = torch.ops.aten.add.Tensor(arg1103_1, 1e-05);  arg1103_1 = None
        sqrt_545 = torch.ops.aten.sqrt.default(add_1586);  add_1586 = None
        reciprocal_545 = torch.ops.aten.reciprocal.default(sqrt_545);  sqrt_545 = None
        mul_1835 = torch.ops.aten.mul.Tensor(reciprocal_545, 1);  reciprocal_545 = None
        unsqueeze_4410 = torch.ops.aten.unsqueeze.default(arg1102_1, -1);  arg1102_1 = None
        unsqueeze_4411 = torch.ops.aten.unsqueeze.default(unsqueeze_4410, -1);  unsqueeze_4410 = None
        unsqueeze_4412 = torch.ops.aten.unsqueeze.default(mul_1835, -1);  mul_1835 = None
        unsqueeze_4413 = torch.ops.aten.unsqueeze.default(unsqueeze_4412, -1);  unsqueeze_4412 = None
        sub_545 = torch.ops.aten.sub.Tensor(convolution_545, unsqueeze_4411);  convolution_545 = unsqueeze_4411 = None
        mul_1836 = torch.ops.aten.mul.Tensor(sub_545, unsqueeze_4413);  sub_545 = unsqueeze_4413 = None
        unsqueeze_4414 = torch.ops.aten.unsqueeze.default(arg1104_1, -1);  arg1104_1 = None
        unsqueeze_4415 = torch.ops.aten.unsqueeze.default(unsqueeze_4414, -1);  unsqueeze_4414 = None
        mul_1837 = torch.ops.aten.mul.Tensor(mul_1836, unsqueeze_4415);  mul_1836 = unsqueeze_4415 = None
        unsqueeze_4416 = torch.ops.aten.unsqueeze.default(arg1105_1, -1);  arg1105_1 = None
        unsqueeze_4417 = torch.ops.aten.unsqueeze.default(unsqueeze_4416, -1);  unsqueeze_4416 = None
        add_1587 = torch.ops.aten.add.Tensor(mul_1837, unsqueeze_4417);  mul_1837 = unsqueeze_4417 = None
        add_1588 = torch.ops.aten.add.Tensor(add_1587, relu_481);  add_1587 = relu_481 = None
        relu_483 = torch.ops.aten.relu.default(add_1588);  add_1588 = None
        convolution_546 = torch.ops.aten.convolution.default(relu_483, arg1106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1106_1 = None
        add_1589 = torch.ops.aten.add.Tensor(arg1108_1, 1e-05);  arg1108_1 = None
        sqrt_546 = torch.ops.aten.sqrt.default(add_1589);  add_1589 = None
        reciprocal_546 = torch.ops.aten.reciprocal.default(sqrt_546);  sqrt_546 = None
        mul_1838 = torch.ops.aten.mul.Tensor(reciprocal_546, 1);  reciprocal_546 = None
        unsqueeze_4418 = torch.ops.aten.unsqueeze.default(arg1107_1, -1);  arg1107_1 = None
        unsqueeze_4419 = torch.ops.aten.unsqueeze.default(unsqueeze_4418, -1);  unsqueeze_4418 = None
        unsqueeze_4420 = torch.ops.aten.unsqueeze.default(mul_1838, -1);  mul_1838 = None
        unsqueeze_4421 = torch.ops.aten.unsqueeze.default(unsqueeze_4420, -1);  unsqueeze_4420 = None
        sub_546 = torch.ops.aten.sub.Tensor(convolution_546, unsqueeze_4419);  convolution_546 = unsqueeze_4419 = None
        mul_1839 = torch.ops.aten.mul.Tensor(sub_546, unsqueeze_4421);  sub_546 = unsqueeze_4421 = None
        unsqueeze_4422 = torch.ops.aten.unsqueeze.default(arg1109_1, -1);  arg1109_1 = None
        unsqueeze_4423 = torch.ops.aten.unsqueeze.default(unsqueeze_4422, -1);  unsqueeze_4422 = None
        mul_1840 = torch.ops.aten.mul.Tensor(mul_1839, unsqueeze_4423);  mul_1839 = unsqueeze_4423 = None
        unsqueeze_4424 = torch.ops.aten.unsqueeze.default(arg1110_1, -1);  arg1110_1 = None
        unsqueeze_4425 = torch.ops.aten.unsqueeze.default(unsqueeze_4424, -1);  unsqueeze_4424 = None
        add_1590 = torch.ops.aten.add.Tensor(mul_1840, unsqueeze_4425);  mul_1840 = unsqueeze_4425 = None
        relu_484 = torch.ops.aten.relu.default(add_1590);  add_1590 = None
        convolution_547 = torch.ops.aten.convolution.default(relu_484, arg1111_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_484 = arg1111_1 = None
        add_1591 = torch.ops.aten.add.Tensor(arg1113_1, 1e-05);  arg1113_1 = None
        sqrt_547 = torch.ops.aten.sqrt.default(add_1591);  add_1591 = None
        reciprocal_547 = torch.ops.aten.reciprocal.default(sqrt_547);  sqrt_547 = None
        mul_1841 = torch.ops.aten.mul.Tensor(reciprocal_547, 1);  reciprocal_547 = None
        unsqueeze_4426 = torch.ops.aten.unsqueeze.default(arg1112_1, -1);  arg1112_1 = None
        unsqueeze_4427 = torch.ops.aten.unsqueeze.default(unsqueeze_4426, -1);  unsqueeze_4426 = None
        unsqueeze_4428 = torch.ops.aten.unsqueeze.default(mul_1841, -1);  mul_1841 = None
        unsqueeze_4429 = torch.ops.aten.unsqueeze.default(unsqueeze_4428, -1);  unsqueeze_4428 = None
        sub_547 = torch.ops.aten.sub.Tensor(convolution_547, unsqueeze_4427);  convolution_547 = unsqueeze_4427 = None
        mul_1842 = torch.ops.aten.mul.Tensor(sub_547, unsqueeze_4429);  sub_547 = unsqueeze_4429 = None
        unsqueeze_4430 = torch.ops.aten.unsqueeze.default(arg1114_1, -1);  arg1114_1 = None
        unsqueeze_4431 = torch.ops.aten.unsqueeze.default(unsqueeze_4430, -1);  unsqueeze_4430 = None
        mul_1843 = torch.ops.aten.mul.Tensor(mul_1842, unsqueeze_4431);  mul_1842 = unsqueeze_4431 = None
        unsqueeze_4432 = torch.ops.aten.unsqueeze.default(arg1115_1, -1);  arg1115_1 = None
        unsqueeze_4433 = torch.ops.aten.unsqueeze.default(unsqueeze_4432, -1);  unsqueeze_4432 = None
        add_1592 = torch.ops.aten.add.Tensor(mul_1843, unsqueeze_4433);  mul_1843 = unsqueeze_4433 = None
        add_1593 = torch.ops.aten.add.Tensor(add_1592, relu_483);  add_1592 = relu_483 = None
        relu_485 = torch.ops.aten.relu.default(add_1593);  add_1593 = None
        convolution_548 = torch.ops.aten.convolution.default(relu_485, arg1116_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1116_1 = None
        add_1594 = torch.ops.aten.add.Tensor(arg1118_1, 1e-05);  arg1118_1 = None
        sqrt_548 = torch.ops.aten.sqrt.default(add_1594);  add_1594 = None
        reciprocal_548 = torch.ops.aten.reciprocal.default(sqrt_548);  sqrt_548 = None
        mul_1844 = torch.ops.aten.mul.Tensor(reciprocal_548, 1);  reciprocal_548 = None
        unsqueeze_4434 = torch.ops.aten.unsqueeze.default(arg1117_1, -1);  arg1117_1 = None
        unsqueeze_4435 = torch.ops.aten.unsqueeze.default(unsqueeze_4434, -1);  unsqueeze_4434 = None
        unsqueeze_4436 = torch.ops.aten.unsqueeze.default(mul_1844, -1);  mul_1844 = None
        unsqueeze_4437 = torch.ops.aten.unsqueeze.default(unsqueeze_4436, -1);  unsqueeze_4436 = None
        sub_548 = torch.ops.aten.sub.Tensor(convolution_548, unsqueeze_4435);  convolution_548 = unsqueeze_4435 = None
        mul_1845 = torch.ops.aten.mul.Tensor(sub_548, unsqueeze_4437);  sub_548 = unsqueeze_4437 = None
        unsqueeze_4438 = torch.ops.aten.unsqueeze.default(arg1119_1, -1);  arg1119_1 = None
        unsqueeze_4439 = torch.ops.aten.unsqueeze.default(unsqueeze_4438, -1);  unsqueeze_4438 = None
        mul_1846 = torch.ops.aten.mul.Tensor(mul_1845, unsqueeze_4439);  mul_1845 = unsqueeze_4439 = None
        unsqueeze_4440 = torch.ops.aten.unsqueeze.default(arg1120_1, -1);  arg1120_1 = None
        unsqueeze_4441 = torch.ops.aten.unsqueeze.default(unsqueeze_4440, -1);  unsqueeze_4440 = None
        add_1595 = torch.ops.aten.add.Tensor(mul_1846, unsqueeze_4441);  mul_1846 = unsqueeze_4441 = None
        relu_486 = torch.ops.aten.relu.default(add_1595);  add_1595 = None
        convolution_549 = torch.ops.aten.convolution.default(relu_486, arg1121_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_486 = arg1121_1 = None
        add_1596 = torch.ops.aten.add.Tensor(arg1123_1, 1e-05);  arg1123_1 = None
        sqrt_549 = torch.ops.aten.sqrt.default(add_1596);  add_1596 = None
        reciprocal_549 = torch.ops.aten.reciprocal.default(sqrt_549);  sqrt_549 = None
        mul_1847 = torch.ops.aten.mul.Tensor(reciprocal_549, 1);  reciprocal_549 = None
        unsqueeze_4442 = torch.ops.aten.unsqueeze.default(arg1122_1, -1);  arg1122_1 = None
        unsqueeze_4443 = torch.ops.aten.unsqueeze.default(unsqueeze_4442, -1);  unsqueeze_4442 = None
        unsqueeze_4444 = torch.ops.aten.unsqueeze.default(mul_1847, -1);  mul_1847 = None
        unsqueeze_4445 = torch.ops.aten.unsqueeze.default(unsqueeze_4444, -1);  unsqueeze_4444 = None
        sub_549 = torch.ops.aten.sub.Tensor(convolution_549, unsqueeze_4443);  convolution_549 = unsqueeze_4443 = None
        mul_1848 = torch.ops.aten.mul.Tensor(sub_549, unsqueeze_4445);  sub_549 = unsqueeze_4445 = None
        unsqueeze_4446 = torch.ops.aten.unsqueeze.default(arg1124_1, -1);  arg1124_1 = None
        unsqueeze_4447 = torch.ops.aten.unsqueeze.default(unsqueeze_4446, -1);  unsqueeze_4446 = None
        mul_1849 = torch.ops.aten.mul.Tensor(mul_1848, unsqueeze_4447);  mul_1848 = unsqueeze_4447 = None
        unsqueeze_4448 = torch.ops.aten.unsqueeze.default(arg1125_1, -1);  arg1125_1 = None
        unsqueeze_4449 = torch.ops.aten.unsqueeze.default(unsqueeze_4448, -1);  unsqueeze_4448 = None
        add_1597 = torch.ops.aten.add.Tensor(mul_1849, unsqueeze_4449);  mul_1849 = unsqueeze_4449 = None
        add_1598 = torch.ops.aten.add.Tensor(add_1597, relu_485);  add_1597 = relu_485 = None
        relu_487 = torch.ops.aten.relu.default(add_1598);  add_1598 = None
        convolution_550 = torch.ops.aten.convolution.default(relu_467, arg1126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1126_1 = None
        add_1599 = torch.ops.aten.add.Tensor(arg1128_1, 1e-05);  arg1128_1 = None
        sqrt_550 = torch.ops.aten.sqrt.default(add_1599);  add_1599 = None
        reciprocal_550 = torch.ops.aten.reciprocal.default(sqrt_550);  sqrt_550 = None
        mul_1850 = torch.ops.aten.mul.Tensor(reciprocal_550, 1);  reciprocal_550 = None
        unsqueeze_4450 = torch.ops.aten.unsqueeze.default(arg1127_1, -1);  arg1127_1 = None
        unsqueeze_4451 = torch.ops.aten.unsqueeze.default(unsqueeze_4450, -1);  unsqueeze_4450 = None
        unsqueeze_4452 = torch.ops.aten.unsqueeze.default(mul_1850, -1);  mul_1850 = None
        unsqueeze_4453 = torch.ops.aten.unsqueeze.default(unsqueeze_4452, -1);  unsqueeze_4452 = None
        sub_550 = torch.ops.aten.sub.Tensor(convolution_550, unsqueeze_4451);  convolution_550 = unsqueeze_4451 = None
        mul_1851 = torch.ops.aten.mul.Tensor(sub_550, unsqueeze_4453);  sub_550 = unsqueeze_4453 = None
        unsqueeze_4454 = torch.ops.aten.unsqueeze.default(arg1129_1, -1);  arg1129_1 = None
        unsqueeze_4455 = torch.ops.aten.unsqueeze.default(unsqueeze_4454, -1);  unsqueeze_4454 = None
        mul_1852 = torch.ops.aten.mul.Tensor(mul_1851, unsqueeze_4455);  mul_1851 = unsqueeze_4455 = None
        unsqueeze_4456 = torch.ops.aten.unsqueeze.default(arg1130_1, -1);  arg1130_1 = None
        unsqueeze_4457 = torch.ops.aten.unsqueeze.default(unsqueeze_4456, -1);  unsqueeze_4456 = None
        add_1600 = torch.ops.aten.add.Tensor(mul_1852, unsqueeze_4457);  mul_1852 = unsqueeze_4457 = None
        relu_488 = torch.ops.aten.relu.default(add_1600);  add_1600 = None
        convolution_551 = torch.ops.aten.convolution.default(relu_488, arg1131_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_488 = arg1131_1 = None
        add_1601 = torch.ops.aten.add.Tensor(arg1133_1, 1e-05);  arg1133_1 = None
        sqrt_551 = torch.ops.aten.sqrt.default(add_1601);  add_1601 = None
        reciprocal_551 = torch.ops.aten.reciprocal.default(sqrt_551);  sqrt_551 = None
        mul_1853 = torch.ops.aten.mul.Tensor(reciprocal_551, 1);  reciprocal_551 = None
        unsqueeze_4458 = torch.ops.aten.unsqueeze.default(arg1132_1, -1);  arg1132_1 = None
        unsqueeze_4459 = torch.ops.aten.unsqueeze.default(unsqueeze_4458, -1);  unsqueeze_4458 = None
        unsqueeze_4460 = torch.ops.aten.unsqueeze.default(mul_1853, -1);  mul_1853 = None
        unsqueeze_4461 = torch.ops.aten.unsqueeze.default(unsqueeze_4460, -1);  unsqueeze_4460 = None
        sub_551 = torch.ops.aten.sub.Tensor(convolution_551, unsqueeze_4459);  convolution_551 = unsqueeze_4459 = None
        mul_1854 = torch.ops.aten.mul.Tensor(sub_551, unsqueeze_4461);  sub_551 = unsqueeze_4461 = None
        unsqueeze_4462 = torch.ops.aten.unsqueeze.default(arg1134_1, -1);  arg1134_1 = None
        unsqueeze_4463 = torch.ops.aten.unsqueeze.default(unsqueeze_4462, -1);  unsqueeze_4462 = None
        mul_1855 = torch.ops.aten.mul.Tensor(mul_1854, unsqueeze_4463);  mul_1854 = unsqueeze_4463 = None
        unsqueeze_4464 = torch.ops.aten.unsqueeze.default(arg1135_1, -1);  arg1135_1 = None
        unsqueeze_4465 = torch.ops.aten.unsqueeze.default(unsqueeze_4464, -1);  unsqueeze_4464 = None
        add_1602 = torch.ops.aten.add.Tensor(mul_1855, unsqueeze_4465);  mul_1855 = unsqueeze_4465 = None
        add_1603 = torch.ops.aten.add.Tensor(add_1602, relu_467);  add_1602 = relu_467 = None
        relu_489 = torch.ops.aten.relu.default(add_1603);  add_1603 = None
        convolution_552 = torch.ops.aten.convolution.default(relu_489, arg1136_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1136_1 = None
        add_1604 = torch.ops.aten.add.Tensor(arg1138_1, 1e-05);  arg1138_1 = None
        sqrt_552 = torch.ops.aten.sqrt.default(add_1604);  add_1604 = None
        reciprocal_552 = torch.ops.aten.reciprocal.default(sqrt_552);  sqrt_552 = None
        mul_1856 = torch.ops.aten.mul.Tensor(reciprocal_552, 1);  reciprocal_552 = None
        unsqueeze_4466 = torch.ops.aten.unsqueeze.default(arg1137_1, -1);  arg1137_1 = None
        unsqueeze_4467 = torch.ops.aten.unsqueeze.default(unsqueeze_4466, -1);  unsqueeze_4466 = None
        unsqueeze_4468 = torch.ops.aten.unsqueeze.default(mul_1856, -1);  mul_1856 = None
        unsqueeze_4469 = torch.ops.aten.unsqueeze.default(unsqueeze_4468, -1);  unsqueeze_4468 = None
        sub_552 = torch.ops.aten.sub.Tensor(convolution_552, unsqueeze_4467);  convolution_552 = unsqueeze_4467 = None
        mul_1857 = torch.ops.aten.mul.Tensor(sub_552, unsqueeze_4469);  sub_552 = unsqueeze_4469 = None
        unsqueeze_4470 = torch.ops.aten.unsqueeze.default(arg1139_1, -1);  arg1139_1 = None
        unsqueeze_4471 = torch.ops.aten.unsqueeze.default(unsqueeze_4470, -1);  unsqueeze_4470 = None
        mul_1858 = torch.ops.aten.mul.Tensor(mul_1857, unsqueeze_4471);  mul_1857 = unsqueeze_4471 = None
        unsqueeze_4472 = torch.ops.aten.unsqueeze.default(arg1140_1, -1);  arg1140_1 = None
        unsqueeze_4473 = torch.ops.aten.unsqueeze.default(unsqueeze_4472, -1);  unsqueeze_4472 = None
        add_1605 = torch.ops.aten.add.Tensor(mul_1858, unsqueeze_4473);  mul_1858 = unsqueeze_4473 = None
        relu_490 = torch.ops.aten.relu.default(add_1605);  add_1605 = None
        convolution_553 = torch.ops.aten.convolution.default(relu_490, arg1141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_490 = arg1141_1 = None
        add_1606 = torch.ops.aten.add.Tensor(arg1143_1, 1e-05);  arg1143_1 = None
        sqrt_553 = torch.ops.aten.sqrt.default(add_1606);  add_1606 = None
        reciprocal_553 = torch.ops.aten.reciprocal.default(sqrt_553);  sqrt_553 = None
        mul_1859 = torch.ops.aten.mul.Tensor(reciprocal_553, 1);  reciprocal_553 = None
        unsqueeze_4474 = torch.ops.aten.unsqueeze.default(arg1142_1, -1);  arg1142_1 = None
        unsqueeze_4475 = torch.ops.aten.unsqueeze.default(unsqueeze_4474, -1);  unsqueeze_4474 = None
        unsqueeze_4476 = torch.ops.aten.unsqueeze.default(mul_1859, -1);  mul_1859 = None
        unsqueeze_4477 = torch.ops.aten.unsqueeze.default(unsqueeze_4476, -1);  unsqueeze_4476 = None
        sub_553 = torch.ops.aten.sub.Tensor(convolution_553, unsqueeze_4475);  convolution_553 = unsqueeze_4475 = None
        mul_1860 = torch.ops.aten.mul.Tensor(sub_553, unsqueeze_4477);  sub_553 = unsqueeze_4477 = None
        unsqueeze_4478 = torch.ops.aten.unsqueeze.default(arg1144_1, -1);  arg1144_1 = None
        unsqueeze_4479 = torch.ops.aten.unsqueeze.default(unsqueeze_4478, -1);  unsqueeze_4478 = None
        mul_1861 = torch.ops.aten.mul.Tensor(mul_1860, unsqueeze_4479);  mul_1860 = unsqueeze_4479 = None
        unsqueeze_4480 = torch.ops.aten.unsqueeze.default(arg1145_1, -1);  arg1145_1 = None
        unsqueeze_4481 = torch.ops.aten.unsqueeze.default(unsqueeze_4480, -1);  unsqueeze_4480 = None
        add_1607 = torch.ops.aten.add.Tensor(mul_1861, unsqueeze_4481);  mul_1861 = unsqueeze_4481 = None
        add_1608 = torch.ops.aten.add.Tensor(add_1607, relu_489);  add_1607 = relu_489 = None
        relu_491 = torch.ops.aten.relu.default(add_1608);  add_1608 = None
        convolution_554 = torch.ops.aten.convolution.default(relu_491, arg1146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1146_1 = None
        add_1609 = torch.ops.aten.add.Tensor(arg1148_1, 1e-05);  arg1148_1 = None
        sqrt_554 = torch.ops.aten.sqrt.default(add_1609);  add_1609 = None
        reciprocal_554 = torch.ops.aten.reciprocal.default(sqrt_554);  sqrt_554 = None
        mul_1862 = torch.ops.aten.mul.Tensor(reciprocal_554, 1);  reciprocal_554 = None
        unsqueeze_4482 = torch.ops.aten.unsqueeze.default(arg1147_1, -1);  arg1147_1 = None
        unsqueeze_4483 = torch.ops.aten.unsqueeze.default(unsqueeze_4482, -1);  unsqueeze_4482 = None
        unsqueeze_4484 = torch.ops.aten.unsqueeze.default(mul_1862, -1);  mul_1862 = None
        unsqueeze_4485 = torch.ops.aten.unsqueeze.default(unsqueeze_4484, -1);  unsqueeze_4484 = None
        sub_554 = torch.ops.aten.sub.Tensor(convolution_554, unsqueeze_4483);  convolution_554 = unsqueeze_4483 = None
        mul_1863 = torch.ops.aten.mul.Tensor(sub_554, unsqueeze_4485);  sub_554 = unsqueeze_4485 = None
        unsqueeze_4486 = torch.ops.aten.unsqueeze.default(arg1149_1, -1);  arg1149_1 = None
        unsqueeze_4487 = torch.ops.aten.unsqueeze.default(unsqueeze_4486, -1);  unsqueeze_4486 = None
        mul_1864 = torch.ops.aten.mul.Tensor(mul_1863, unsqueeze_4487);  mul_1863 = unsqueeze_4487 = None
        unsqueeze_4488 = torch.ops.aten.unsqueeze.default(arg1150_1, -1);  arg1150_1 = None
        unsqueeze_4489 = torch.ops.aten.unsqueeze.default(unsqueeze_4488, -1);  unsqueeze_4488 = None
        add_1610 = torch.ops.aten.add.Tensor(mul_1864, unsqueeze_4489);  mul_1864 = unsqueeze_4489 = None
        relu_492 = torch.ops.aten.relu.default(add_1610);  add_1610 = None
        convolution_555 = torch.ops.aten.convolution.default(relu_492, arg1151_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_492 = arg1151_1 = None
        add_1611 = torch.ops.aten.add.Tensor(arg1153_1, 1e-05);  arg1153_1 = None
        sqrt_555 = torch.ops.aten.sqrt.default(add_1611);  add_1611 = None
        reciprocal_555 = torch.ops.aten.reciprocal.default(sqrt_555);  sqrt_555 = None
        mul_1865 = torch.ops.aten.mul.Tensor(reciprocal_555, 1);  reciprocal_555 = None
        unsqueeze_4490 = torch.ops.aten.unsqueeze.default(arg1152_1, -1);  arg1152_1 = None
        unsqueeze_4491 = torch.ops.aten.unsqueeze.default(unsqueeze_4490, -1);  unsqueeze_4490 = None
        unsqueeze_4492 = torch.ops.aten.unsqueeze.default(mul_1865, -1);  mul_1865 = None
        unsqueeze_4493 = torch.ops.aten.unsqueeze.default(unsqueeze_4492, -1);  unsqueeze_4492 = None
        sub_555 = torch.ops.aten.sub.Tensor(convolution_555, unsqueeze_4491);  convolution_555 = unsqueeze_4491 = None
        mul_1866 = torch.ops.aten.mul.Tensor(sub_555, unsqueeze_4493);  sub_555 = unsqueeze_4493 = None
        unsqueeze_4494 = torch.ops.aten.unsqueeze.default(arg1154_1, -1);  arg1154_1 = None
        unsqueeze_4495 = torch.ops.aten.unsqueeze.default(unsqueeze_4494, -1);  unsqueeze_4494 = None
        mul_1867 = torch.ops.aten.mul.Tensor(mul_1866, unsqueeze_4495);  mul_1866 = unsqueeze_4495 = None
        unsqueeze_4496 = torch.ops.aten.unsqueeze.default(arg1155_1, -1);  arg1155_1 = None
        unsqueeze_4497 = torch.ops.aten.unsqueeze.default(unsqueeze_4496, -1);  unsqueeze_4496 = None
        add_1612 = torch.ops.aten.add.Tensor(mul_1867, unsqueeze_4497);  mul_1867 = unsqueeze_4497 = None
        add_1613 = torch.ops.aten.add.Tensor(add_1612, relu_491);  add_1612 = relu_491 = None
        relu_493 = torch.ops.aten.relu.default(add_1613);  add_1613 = None
        convolution_556 = torch.ops.aten.convolution.default(relu_493, arg1156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1156_1 = None
        add_1614 = torch.ops.aten.add.Tensor(arg1158_1, 1e-05);  arg1158_1 = None
        sqrt_556 = torch.ops.aten.sqrt.default(add_1614);  add_1614 = None
        reciprocal_556 = torch.ops.aten.reciprocal.default(sqrt_556);  sqrt_556 = None
        mul_1868 = torch.ops.aten.mul.Tensor(reciprocal_556, 1);  reciprocal_556 = None
        unsqueeze_4498 = torch.ops.aten.unsqueeze.default(arg1157_1, -1);  arg1157_1 = None
        unsqueeze_4499 = torch.ops.aten.unsqueeze.default(unsqueeze_4498, -1);  unsqueeze_4498 = None
        unsqueeze_4500 = torch.ops.aten.unsqueeze.default(mul_1868, -1);  mul_1868 = None
        unsqueeze_4501 = torch.ops.aten.unsqueeze.default(unsqueeze_4500, -1);  unsqueeze_4500 = None
        sub_556 = torch.ops.aten.sub.Tensor(convolution_556, unsqueeze_4499);  convolution_556 = unsqueeze_4499 = None
        mul_1869 = torch.ops.aten.mul.Tensor(sub_556, unsqueeze_4501);  sub_556 = unsqueeze_4501 = None
        unsqueeze_4502 = torch.ops.aten.unsqueeze.default(arg1159_1, -1);  arg1159_1 = None
        unsqueeze_4503 = torch.ops.aten.unsqueeze.default(unsqueeze_4502, -1);  unsqueeze_4502 = None
        mul_1870 = torch.ops.aten.mul.Tensor(mul_1869, unsqueeze_4503);  mul_1869 = unsqueeze_4503 = None
        unsqueeze_4504 = torch.ops.aten.unsqueeze.default(arg1160_1, -1);  arg1160_1 = None
        unsqueeze_4505 = torch.ops.aten.unsqueeze.default(unsqueeze_4504, -1);  unsqueeze_4504 = None
        add_1615 = torch.ops.aten.add.Tensor(mul_1870, unsqueeze_4505);  mul_1870 = unsqueeze_4505 = None
        relu_494 = torch.ops.aten.relu.default(add_1615);  add_1615 = None
        convolution_557 = torch.ops.aten.convolution.default(relu_494, arg1161_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_494 = arg1161_1 = None
        add_1616 = torch.ops.aten.add.Tensor(arg1163_1, 1e-05);  arg1163_1 = None
        sqrt_557 = torch.ops.aten.sqrt.default(add_1616);  add_1616 = None
        reciprocal_557 = torch.ops.aten.reciprocal.default(sqrt_557);  sqrt_557 = None
        mul_1871 = torch.ops.aten.mul.Tensor(reciprocal_557, 1);  reciprocal_557 = None
        unsqueeze_4506 = torch.ops.aten.unsqueeze.default(arg1162_1, -1);  arg1162_1 = None
        unsqueeze_4507 = torch.ops.aten.unsqueeze.default(unsqueeze_4506, -1);  unsqueeze_4506 = None
        unsqueeze_4508 = torch.ops.aten.unsqueeze.default(mul_1871, -1);  mul_1871 = None
        unsqueeze_4509 = torch.ops.aten.unsqueeze.default(unsqueeze_4508, -1);  unsqueeze_4508 = None
        sub_557 = torch.ops.aten.sub.Tensor(convolution_557, unsqueeze_4507);  convolution_557 = unsqueeze_4507 = None
        mul_1872 = torch.ops.aten.mul.Tensor(sub_557, unsqueeze_4509);  sub_557 = unsqueeze_4509 = None
        unsqueeze_4510 = torch.ops.aten.unsqueeze.default(arg1164_1, -1);  arg1164_1 = None
        unsqueeze_4511 = torch.ops.aten.unsqueeze.default(unsqueeze_4510, -1);  unsqueeze_4510 = None
        mul_1873 = torch.ops.aten.mul.Tensor(mul_1872, unsqueeze_4511);  mul_1872 = unsqueeze_4511 = None
        unsqueeze_4512 = torch.ops.aten.unsqueeze.default(arg1165_1, -1);  arg1165_1 = None
        unsqueeze_4513 = torch.ops.aten.unsqueeze.default(unsqueeze_4512, -1);  unsqueeze_4512 = None
        add_1617 = torch.ops.aten.add.Tensor(mul_1873, unsqueeze_4513);  mul_1873 = unsqueeze_4513 = None
        add_1618 = torch.ops.aten.add.Tensor(add_1617, relu_493);  add_1617 = relu_493 = None
        relu_495 = torch.ops.aten.relu.default(add_1618);  add_1618 = None
        convolution_558 = torch.ops.aten.convolution.default(relu_471, arg1166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1166_1 = None
        add_1619 = torch.ops.aten.add.Tensor(arg1168_1, 1e-05);  arg1168_1 = None
        sqrt_558 = torch.ops.aten.sqrt.default(add_1619);  add_1619 = None
        reciprocal_558 = torch.ops.aten.reciprocal.default(sqrt_558);  sqrt_558 = None
        mul_1874 = torch.ops.aten.mul.Tensor(reciprocal_558, 1);  reciprocal_558 = None
        unsqueeze_4514 = torch.ops.aten.unsqueeze.default(arg1167_1, -1);  arg1167_1 = None
        unsqueeze_4515 = torch.ops.aten.unsqueeze.default(unsqueeze_4514, -1);  unsqueeze_4514 = None
        unsqueeze_4516 = torch.ops.aten.unsqueeze.default(mul_1874, -1);  mul_1874 = None
        unsqueeze_4517 = torch.ops.aten.unsqueeze.default(unsqueeze_4516, -1);  unsqueeze_4516 = None
        sub_558 = torch.ops.aten.sub.Tensor(convolution_558, unsqueeze_4515);  convolution_558 = unsqueeze_4515 = None
        mul_1875 = torch.ops.aten.mul.Tensor(sub_558, unsqueeze_4517);  sub_558 = unsqueeze_4517 = None
        unsqueeze_4518 = torch.ops.aten.unsqueeze.default(arg1169_1, -1);  arg1169_1 = None
        unsqueeze_4519 = torch.ops.aten.unsqueeze.default(unsqueeze_4518, -1);  unsqueeze_4518 = None
        mul_1876 = torch.ops.aten.mul.Tensor(mul_1875, unsqueeze_4519);  mul_1875 = unsqueeze_4519 = None
        unsqueeze_4520 = torch.ops.aten.unsqueeze.default(arg1170_1, -1);  arg1170_1 = None
        unsqueeze_4521 = torch.ops.aten.unsqueeze.default(unsqueeze_4520, -1);  unsqueeze_4520 = None
        add_1620 = torch.ops.aten.add.Tensor(mul_1876, unsqueeze_4521);  mul_1876 = unsqueeze_4521 = None
        relu_496 = torch.ops.aten.relu.default(add_1620);  add_1620 = None
        convolution_559 = torch.ops.aten.convolution.default(relu_496, arg1171_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_496 = arg1171_1 = None
        add_1621 = torch.ops.aten.add.Tensor(arg1173_1, 1e-05);  arg1173_1 = None
        sqrt_559 = torch.ops.aten.sqrt.default(add_1621);  add_1621 = None
        reciprocal_559 = torch.ops.aten.reciprocal.default(sqrt_559);  sqrt_559 = None
        mul_1877 = torch.ops.aten.mul.Tensor(reciprocal_559, 1);  reciprocal_559 = None
        unsqueeze_4522 = torch.ops.aten.unsqueeze.default(arg1172_1, -1);  arg1172_1 = None
        unsqueeze_4523 = torch.ops.aten.unsqueeze.default(unsqueeze_4522, -1);  unsqueeze_4522 = None
        unsqueeze_4524 = torch.ops.aten.unsqueeze.default(mul_1877, -1);  mul_1877 = None
        unsqueeze_4525 = torch.ops.aten.unsqueeze.default(unsqueeze_4524, -1);  unsqueeze_4524 = None
        sub_559 = torch.ops.aten.sub.Tensor(convolution_559, unsqueeze_4523);  convolution_559 = unsqueeze_4523 = None
        mul_1878 = torch.ops.aten.mul.Tensor(sub_559, unsqueeze_4525);  sub_559 = unsqueeze_4525 = None
        unsqueeze_4526 = torch.ops.aten.unsqueeze.default(arg1174_1, -1);  arg1174_1 = None
        unsqueeze_4527 = torch.ops.aten.unsqueeze.default(unsqueeze_4526, -1);  unsqueeze_4526 = None
        mul_1879 = torch.ops.aten.mul.Tensor(mul_1878, unsqueeze_4527);  mul_1878 = unsqueeze_4527 = None
        unsqueeze_4528 = torch.ops.aten.unsqueeze.default(arg1175_1, -1);  arg1175_1 = None
        unsqueeze_4529 = torch.ops.aten.unsqueeze.default(unsqueeze_4528, -1);  unsqueeze_4528 = None
        add_1622 = torch.ops.aten.add.Tensor(mul_1879, unsqueeze_4529);  mul_1879 = unsqueeze_4529 = None
        add_1623 = torch.ops.aten.add.Tensor(add_1622, relu_471);  add_1622 = relu_471 = None
        relu_497 = torch.ops.aten.relu.default(add_1623);  add_1623 = None
        convolution_560 = torch.ops.aten.convolution.default(relu_497, arg1176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1176_1 = None
        add_1624 = torch.ops.aten.add.Tensor(arg1178_1, 1e-05);  arg1178_1 = None
        sqrt_560 = torch.ops.aten.sqrt.default(add_1624);  add_1624 = None
        reciprocal_560 = torch.ops.aten.reciprocal.default(sqrt_560);  sqrt_560 = None
        mul_1880 = torch.ops.aten.mul.Tensor(reciprocal_560, 1);  reciprocal_560 = None
        unsqueeze_4530 = torch.ops.aten.unsqueeze.default(arg1177_1, -1);  arg1177_1 = None
        unsqueeze_4531 = torch.ops.aten.unsqueeze.default(unsqueeze_4530, -1);  unsqueeze_4530 = None
        unsqueeze_4532 = torch.ops.aten.unsqueeze.default(mul_1880, -1);  mul_1880 = None
        unsqueeze_4533 = torch.ops.aten.unsqueeze.default(unsqueeze_4532, -1);  unsqueeze_4532 = None
        sub_560 = torch.ops.aten.sub.Tensor(convolution_560, unsqueeze_4531);  convolution_560 = unsqueeze_4531 = None
        mul_1881 = torch.ops.aten.mul.Tensor(sub_560, unsqueeze_4533);  sub_560 = unsqueeze_4533 = None
        unsqueeze_4534 = torch.ops.aten.unsqueeze.default(arg1179_1, -1);  arg1179_1 = None
        unsqueeze_4535 = torch.ops.aten.unsqueeze.default(unsqueeze_4534, -1);  unsqueeze_4534 = None
        mul_1882 = torch.ops.aten.mul.Tensor(mul_1881, unsqueeze_4535);  mul_1881 = unsqueeze_4535 = None
        unsqueeze_4536 = torch.ops.aten.unsqueeze.default(arg1180_1, -1);  arg1180_1 = None
        unsqueeze_4537 = torch.ops.aten.unsqueeze.default(unsqueeze_4536, -1);  unsqueeze_4536 = None
        add_1625 = torch.ops.aten.add.Tensor(mul_1882, unsqueeze_4537);  mul_1882 = unsqueeze_4537 = None
        relu_498 = torch.ops.aten.relu.default(add_1625);  add_1625 = None
        convolution_561 = torch.ops.aten.convolution.default(relu_498, arg1181_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_498 = arg1181_1 = None
        add_1626 = torch.ops.aten.add.Tensor(arg1183_1, 1e-05);  arg1183_1 = None
        sqrt_561 = torch.ops.aten.sqrt.default(add_1626);  add_1626 = None
        reciprocal_561 = torch.ops.aten.reciprocal.default(sqrt_561);  sqrt_561 = None
        mul_1883 = torch.ops.aten.mul.Tensor(reciprocal_561, 1);  reciprocal_561 = None
        unsqueeze_4538 = torch.ops.aten.unsqueeze.default(arg1182_1, -1);  arg1182_1 = None
        unsqueeze_4539 = torch.ops.aten.unsqueeze.default(unsqueeze_4538, -1);  unsqueeze_4538 = None
        unsqueeze_4540 = torch.ops.aten.unsqueeze.default(mul_1883, -1);  mul_1883 = None
        unsqueeze_4541 = torch.ops.aten.unsqueeze.default(unsqueeze_4540, -1);  unsqueeze_4540 = None
        sub_561 = torch.ops.aten.sub.Tensor(convolution_561, unsqueeze_4539);  convolution_561 = unsqueeze_4539 = None
        mul_1884 = torch.ops.aten.mul.Tensor(sub_561, unsqueeze_4541);  sub_561 = unsqueeze_4541 = None
        unsqueeze_4542 = torch.ops.aten.unsqueeze.default(arg1184_1, -1);  arg1184_1 = None
        unsqueeze_4543 = torch.ops.aten.unsqueeze.default(unsqueeze_4542, -1);  unsqueeze_4542 = None
        mul_1885 = torch.ops.aten.mul.Tensor(mul_1884, unsqueeze_4543);  mul_1884 = unsqueeze_4543 = None
        unsqueeze_4544 = torch.ops.aten.unsqueeze.default(arg1185_1, -1);  arg1185_1 = None
        unsqueeze_4545 = torch.ops.aten.unsqueeze.default(unsqueeze_4544, -1);  unsqueeze_4544 = None
        add_1627 = torch.ops.aten.add.Tensor(mul_1885, unsqueeze_4545);  mul_1885 = unsqueeze_4545 = None
        add_1628 = torch.ops.aten.add.Tensor(add_1627, relu_497);  add_1627 = relu_497 = None
        relu_499 = torch.ops.aten.relu.default(add_1628);  add_1628 = None
        convolution_562 = torch.ops.aten.convolution.default(relu_499, arg1186_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1186_1 = None
        add_1629 = torch.ops.aten.add.Tensor(arg1188_1, 1e-05);  arg1188_1 = None
        sqrt_562 = torch.ops.aten.sqrt.default(add_1629);  add_1629 = None
        reciprocal_562 = torch.ops.aten.reciprocal.default(sqrt_562);  sqrt_562 = None
        mul_1886 = torch.ops.aten.mul.Tensor(reciprocal_562, 1);  reciprocal_562 = None
        unsqueeze_4546 = torch.ops.aten.unsqueeze.default(arg1187_1, -1);  arg1187_1 = None
        unsqueeze_4547 = torch.ops.aten.unsqueeze.default(unsqueeze_4546, -1);  unsqueeze_4546 = None
        unsqueeze_4548 = torch.ops.aten.unsqueeze.default(mul_1886, -1);  mul_1886 = None
        unsqueeze_4549 = torch.ops.aten.unsqueeze.default(unsqueeze_4548, -1);  unsqueeze_4548 = None
        sub_562 = torch.ops.aten.sub.Tensor(convolution_562, unsqueeze_4547);  convolution_562 = unsqueeze_4547 = None
        mul_1887 = torch.ops.aten.mul.Tensor(sub_562, unsqueeze_4549);  sub_562 = unsqueeze_4549 = None
        unsqueeze_4550 = torch.ops.aten.unsqueeze.default(arg1189_1, -1);  arg1189_1 = None
        unsqueeze_4551 = torch.ops.aten.unsqueeze.default(unsqueeze_4550, -1);  unsqueeze_4550 = None
        mul_1888 = torch.ops.aten.mul.Tensor(mul_1887, unsqueeze_4551);  mul_1887 = unsqueeze_4551 = None
        unsqueeze_4552 = torch.ops.aten.unsqueeze.default(arg1190_1, -1);  arg1190_1 = None
        unsqueeze_4553 = torch.ops.aten.unsqueeze.default(unsqueeze_4552, -1);  unsqueeze_4552 = None
        add_1630 = torch.ops.aten.add.Tensor(mul_1888, unsqueeze_4553);  mul_1888 = unsqueeze_4553 = None
        relu_500 = torch.ops.aten.relu.default(add_1630);  add_1630 = None
        convolution_563 = torch.ops.aten.convolution.default(relu_500, arg1191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_500 = arg1191_1 = None
        add_1631 = torch.ops.aten.add.Tensor(arg1193_1, 1e-05);  arg1193_1 = None
        sqrt_563 = torch.ops.aten.sqrt.default(add_1631);  add_1631 = None
        reciprocal_563 = torch.ops.aten.reciprocal.default(sqrt_563);  sqrt_563 = None
        mul_1889 = torch.ops.aten.mul.Tensor(reciprocal_563, 1);  reciprocal_563 = None
        unsqueeze_4554 = torch.ops.aten.unsqueeze.default(arg1192_1, -1);  arg1192_1 = None
        unsqueeze_4555 = torch.ops.aten.unsqueeze.default(unsqueeze_4554, -1);  unsqueeze_4554 = None
        unsqueeze_4556 = torch.ops.aten.unsqueeze.default(mul_1889, -1);  mul_1889 = None
        unsqueeze_4557 = torch.ops.aten.unsqueeze.default(unsqueeze_4556, -1);  unsqueeze_4556 = None
        sub_563 = torch.ops.aten.sub.Tensor(convolution_563, unsqueeze_4555);  convolution_563 = unsqueeze_4555 = None
        mul_1890 = torch.ops.aten.mul.Tensor(sub_563, unsqueeze_4557);  sub_563 = unsqueeze_4557 = None
        unsqueeze_4558 = torch.ops.aten.unsqueeze.default(arg1194_1, -1);  arg1194_1 = None
        unsqueeze_4559 = torch.ops.aten.unsqueeze.default(unsqueeze_4558, -1);  unsqueeze_4558 = None
        mul_1891 = torch.ops.aten.mul.Tensor(mul_1890, unsqueeze_4559);  mul_1890 = unsqueeze_4559 = None
        unsqueeze_4560 = torch.ops.aten.unsqueeze.default(arg1195_1, -1);  arg1195_1 = None
        unsqueeze_4561 = torch.ops.aten.unsqueeze.default(unsqueeze_4560, -1);  unsqueeze_4560 = None
        add_1632 = torch.ops.aten.add.Tensor(mul_1891, unsqueeze_4561);  mul_1891 = unsqueeze_4561 = None
        add_1633 = torch.ops.aten.add.Tensor(add_1632, relu_499);  add_1632 = relu_499 = None
        relu_501 = torch.ops.aten.relu.default(add_1633);  add_1633 = None
        convolution_564 = torch.ops.aten.convolution.default(relu_501, arg1196_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1196_1 = None
        add_1634 = torch.ops.aten.add.Tensor(arg1198_1, 1e-05);  arg1198_1 = None
        sqrt_564 = torch.ops.aten.sqrt.default(add_1634);  add_1634 = None
        reciprocal_564 = torch.ops.aten.reciprocal.default(sqrt_564);  sqrt_564 = None
        mul_1892 = torch.ops.aten.mul.Tensor(reciprocal_564, 1);  reciprocal_564 = None
        unsqueeze_4562 = torch.ops.aten.unsqueeze.default(arg1197_1, -1);  arg1197_1 = None
        unsqueeze_4563 = torch.ops.aten.unsqueeze.default(unsqueeze_4562, -1);  unsqueeze_4562 = None
        unsqueeze_4564 = torch.ops.aten.unsqueeze.default(mul_1892, -1);  mul_1892 = None
        unsqueeze_4565 = torch.ops.aten.unsqueeze.default(unsqueeze_4564, -1);  unsqueeze_4564 = None
        sub_564 = torch.ops.aten.sub.Tensor(convolution_564, unsqueeze_4563);  convolution_564 = unsqueeze_4563 = None
        mul_1893 = torch.ops.aten.mul.Tensor(sub_564, unsqueeze_4565);  sub_564 = unsqueeze_4565 = None
        unsqueeze_4566 = torch.ops.aten.unsqueeze.default(arg1199_1, -1);  arg1199_1 = None
        unsqueeze_4567 = torch.ops.aten.unsqueeze.default(unsqueeze_4566, -1);  unsqueeze_4566 = None
        mul_1894 = torch.ops.aten.mul.Tensor(mul_1893, unsqueeze_4567);  mul_1893 = unsqueeze_4567 = None
        unsqueeze_4568 = torch.ops.aten.unsqueeze.default(arg1200_1, -1);  arg1200_1 = None
        unsqueeze_4569 = torch.ops.aten.unsqueeze.default(unsqueeze_4568, -1);  unsqueeze_4568 = None
        add_1635 = torch.ops.aten.add.Tensor(mul_1894, unsqueeze_4569);  mul_1894 = unsqueeze_4569 = None
        relu_502 = torch.ops.aten.relu.default(add_1635);  add_1635 = None
        convolution_565 = torch.ops.aten.convolution.default(relu_502, arg1201_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_502 = arg1201_1 = None
        add_1636 = torch.ops.aten.add.Tensor(arg1203_1, 1e-05);  arg1203_1 = None
        sqrt_565 = torch.ops.aten.sqrt.default(add_1636);  add_1636 = None
        reciprocal_565 = torch.ops.aten.reciprocal.default(sqrt_565);  sqrt_565 = None
        mul_1895 = torch.ops.aten.mul.Tensor(reciprocal_565, 1);  reciprocal_565 = None
        unsqueeze_4570 = torch.ops.aten.unsqueeze.default(arg1202_1, -1);  arg1202_1 = None
        unsqueeze_4571 = torch.ops.aten.unsqueeze.default(unsqueeze_4570, -1);  unsqueeze_4570 = None
        unsqueeze_4572 = torch.ops.aten.unsqueeze.default(mul_1895, -1);  mul_1895 = None
        unsqueeze_4573 = torch.ops.aten.unsqueeze.default(unsqueeze_4572, -1);  unsqueeze_4572 = None
        sub_565 = torch.ops.aten.sub.Tensor(convolution_565, unsqueeze_4571);  convolution_565 = unsqueeze_4571 = None
        mul_1896 = torch.ops.aten.mul.Tensor(sub_565, unsqueeze_4573);  sub_565 = unsqueeze_4573 = None
        unsqueeze_4574 = torch.ops.aten.unsqueeze.default(arg1204_1, -1);  arg1204_1 = None
        unsqueeze_4575 = torch.ops.aten.unsqueeze.default(unsqueeze_4574, -1);  unsqueeze_4574 = None
        mul_1897 = torch.ops.aten.mul.Tensor(mul_1896, unsqueeze_4575);  mul_1896 = unsqueeze_4575 = None
        unsqueeze_4576 = torch.ops.aten.unsqueeze.default(arg1205_1, -1);  arg1205_1 = None
        unsqueeze_4577 = torch.ops.aten.unsqueeze.default(unsqueeze_4576, -1);  unsqueeze_4576 = None
        add_1637 = torch.ops.aten.add.Tensor(mul_1897, unsqueeze_4577);  mul_1897 = unsqueeze_4577 = None
        add_1638 = torch.ops.aten.add.Tensor(add_1637, relu_501);  add_1637 = relu_501 = None
        relu_503 = torch.ops.aten.relu.default(add_1638);  add_1638 = None
        convolution_566 = torch.ops.aten.convolution.default(relu_487, arg1206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1206_1 = None
        add_1639 = torch.ops.aten.add.Tensor(arg1208_1, 1e-05);  arg1208_1 = None
        sqrt_566 = torch.ops.aten.sqrt.default(add_1639);  add_1639 = None
        reciprocal_566 = torch.ops.aten.reciprocal.default(sqrt_566);  sqrt_566 = None
        mul_1898 = torch.ops.aten.mul.Tensor(reciprocal_566, 1);  reciprocal_566 = None
        unsqueeze_4578 = torch.ops.aten.unsqueeze.default(arg1207_1, -1);  arg1207_1 = None
        unsqueeze_4579 = torch.ops.aten.unsqueeze.default(unsqueeze_4578, -1);  unsqueeze_4578 = None
        unsqueeze_4580 = torch.ops.aten.unsqueeze.default(mul_1898, -1);  mul_1898 = None
        unsqueeze_4581 = torch.ops.aten.unsqueeze.default(unsqueeze_4580, -1);  unsqueeze_4580 = None
        sub_566 = torch.ops.aten.sub.Tensor(convolution_566, unsqueeze_4579);  convolution_566 = unsqueeze_4579 = None
        mul_1899 = torch.ops.aten.mul.Tensor(sub_566, unsqueeze_4581);  sub_566 = unsqueeze_4581 = None
        unsqueeze_4582 = torch.ops.aten.unsqueeze.default(arg1209_1, -1);  arg1209_1 = None
        unsqueeze_4583 = torch.ops.aten.unsqueeze.default(unsqueeze_4582, -1);  unsqueeze_4582 = None
        mul_1900 = torch.ops.aten.mul.Tensor(mul_1899, unsqueeze_4583);  mul_1899 = unsqueeze_4583 = None
        unsqueeze_4584 = torch.ops.aten.unsqueeze.default(arg1210_1, -1);  arg1210_1 = None
        unsqueeze_4585 = torch.ops.aten.unsqueeze.default(unsqueeze_4584, -1);  unsqueeze_4584 = None
        add_1640 = torch.ops.aten.add.Tensor(mul_1900, unsqueeze_4585);  mul_1900 = unsqueeze_4585 = None
        iota_100 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1901 = torch.ops.aten.mul.Tensor(iota_100, 1);  iota_100 = None
        add_1641 = torch.ops.aten.add.Tensor(mul_1901, 0);  mul_1901 = None
        convert_element_type_1334 = torch.ops.prims.convert_element_type.default(add_1641, torch.float32);  add_1641 = None
        add_1642 = torch.ops.aten.add.Tensor(convert_element_type_1334, 0.0);  convert_element_type_1334 = None
        mul_1902 = torch.ops.aten.mul.Tensor(add_1642, 0.5);  add_1642 = None
        convert_element_type_1335 = torch.ops.prims.convert_element_type.default(mul_1902, torch.int64);  mul_1902 = None
        unsqueeze_4586 = torch.ops.aten.unsqueeze.default(convert_element_type_1335, -1);  convert_element_type_1335 = None
        iota_101 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1903 = torch.ops.aten.mul.Tensor(iota_101, 1);  iota_101 = None
        add_1643 = torch.ops.aten.add.Tensor(mul_1903, 0);  mul_1903 = None
        convert_element_type_1336 = torch.ops.prims.convert_element_type.default(add_1643, torch.float32);  add_1643 = None
        add_1644 = torch.ops.aten.add.Tensor(convert_element_type_1336, 0.0);  convert_element_type_1336 = None
        mul_1904 = torch.ops.aten.mul.Tensor(add_1644, 0.5);  add_1644 = None
        convert_element_type_1337 = torch.ops.prims.convert_element_type.default(mul_1904, torch.int64);  mul_1904 = None
        _unsafe_index_50 = torch.ops.aten._unsafe_index.Tensor(add_1640, [None, None, unsqueeze_4586, convert_element_type_1337]);  add_1640 = unsqueeze_4586 = convert_element_type_1337 = None
        add_1645 = torch.ops.aten.add.Tensor(relu_479, _unsafe_index_50);  _unsafe_index_50 = None
        convolution_567 = torch.ops.aten.convolution.default(relu_495, arg1211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1211_1 = None
        add_1646 = torch.ops.aten.add.Tensor(arg1213_1, 1e-05);  arg1213_1 = None
        sqrt_567 = torch.ops.aten.sqrt.default(add_1646);  add_1646 = None
        reciprocal_567 = torch.ops.aten.reciprocal.default(sqrt_567);  sqrt_567 = None
        mul_1905 = torch.ops.aten.mul.Tensor(reciprocal_567, 1);  reciprocal_567 = None
        unsqueeze_4587 = torch.ops.aten.unsqueeze.default(arg1212_1, -1);  arg1212_1 = None
        unsqueeze_4588 = torch.ops.aten.unsqueeze.default(unsqueeze_4587, -1);  unsqueeze_4587 = None
        unsqueeze_4589 = torch.ops.aten.unsqueeze.default(mul_1905, -1);  mul_1905 = None
        unsqueeze_4590 = torch.ops.aten.unsqueeze.default(unsqueeze_4589, -1);  unsqueeze_4589 = None
        sub_567 = torch.ops.aten.sub.Tensor(convolution_567, unsqueeze_4588);  convolution_567 = unsqueeze_4588 = None
        mul_1906 = torch.ops.aten.mul.Tensor(sub_567, unsqueeze_4590);  sub_567 = unsqueeze_4590 = None
        unsqueeze_4591 = torch.ops.aten.unsqueeze.default(arg1214_1, -1);  arg1214_1 = None
        unsqueeze_4592 = torch.ops.aten.unsqueeze.default(unsqueeze_4591, -1);  unsqueeze_4591 = None
        mul_1907 = torch.ops.aten.mul.Tensor(mul_1906, unsqueeze_4592);  mul_1906 = unsqueeze_4592 = None
        unsqueeze_4593 = torch.ops.aten.unsqueeze.default(arg1215_1, -1);  arg1215_1 = None
        unsqueeze_4594 = torch.ops.aten.unsqueeze.default(unsqueeze_4593, -1);  unsqueeze_4593 = None
        add_1647 = torch.ops.aten.add.Tensor(mul_1907, unsqueeze_4594);  mul_1907 = unsqueeze_4594 = None
        iota_102 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1908 = torch.ops.aten.mul.Tensor(iota_102, 1);  iota_102 = None
        add_1648 = torch.ops.aten.add.Tensor(mul_1908, 0);  mul_1908 = None
        convert_element_type_1340 = torch.ops.prims.convert_element_type.default(add_1648, torch.float32);  add_1648 = None
        add_1649 = torch.ops.aten.add.Tensor(convert_element_type_1340, 0.0);  convert_element_type_1340 = None
        mul_1909 = torch.ops.aten.mul.Tensor(add_1649, 0.25);  add_1649 = None
        convert_element_type_1341 = torch.ops.prims.convert_element_type.default(mul_1909, torch.int64);  mul_1909 = None
        unsqueeze_4595 = torch.ops.aten.unsqueeze.default(convert_element_type_1341, -1);  convert_element_type_1341 = None
        iota_103 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1910 = torch.ops.aten.mul.Tensor(iota_103, 1);  iota_103 = None
        add_1650 = torch.ops.aten.add.Tensor(mul_1910, 0);  mul_1910 = None
        convert_element_type_1342 = torch.ops.prims.convert_element_type.default(add_1650, torch.float32);  add_1650 = None
        add_1651 = torch.ops.aten.add.Tensor(convert_element_type_1342, 0.0);  convert_element_type_1342 = None
        mul_1911 = torch.ops.aten.mul.Tensor(add_1651, 0.25);  add_1651 = None
        convert_element_type_1343 = torch.ops.prims.convert_element_type.default(mul_1911, torch.int64);  mul_1911 = None
        _unsafe_index_51 = torch.ops.aten._unsafe_index.Tensor(add_1647, [None, None, unsqueeze_4595, convert_element_type_1343]);  add_1647 = unsqueeze_4595 = convert_element_type_1343 = None
        add_1652 = torch.ops.aten.add.Tensor(add_1645, _unsafe_index_51);  add_1645 = _unsafe_index_51 = None
        convolution_568 = torch.ops.aten.convolution.default(relu_503, arg1216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1216_1 = None
        add_1653 = torch.ops.aten.add.Tensor(arg1218_1, 1e-05);  arg1218_1 = None
        sqrt_568 = torch.ops.aten.sqrt.default(add_1653);  add_1653 = None
        reciprocal_568 = torch.ops.aten.reciprocal.default(sqrt_568);  sqrt_568 = None
        mul_1912 = torch.ops.aten.mul.Tensor(reciprocal_568, 1);  reciprocal_568 = None
        unsqueeze_4596 = torch.ops.aten.unsqueeze.default(arg1217_1, -1);  arg1217_1 = None
        unsqueeze_4597 = torch.ops.aten.unsqueeze.default(unsqueeze_4596, -1);  unsqueeze_4596 = None
        unsqueeze_4598 = torch.ops.aten.unsqueeze.default(mul_1912, -1);  mul_1912 = None
        unsqueeze_4599 = torch.ops.aten.unsqueeze.default(unsqueeze_4598, -1);  unsqueeze_4598 = None
        sub_568 = torch.ops.aten.sub.Tensor(convolution_568, unsqueeze_4597);  convolution_568 = unsqueeze_4597 = None
        mul_1913 = torch.ops.aten.mul.Tensor(sub_568, unsqueeze_4599);  sub_568 = unsqueeze_4599 = None
        unsqueeze_4600 = torch.ops.aten.unsqueeze.default(arg1219_1, -1);  arg1219_1 = None
        unsqueeze_4601 = torch.ops.aten.unsqueeze.default(unsqueeze_4600, -1);  unsqueeze_4600 = None
        mul_1914 = torch.ops.aten.mul.Tensor(mul_1913, unsqueeze_4601);  mul_1913 = unsqueeze_4601 = None
        unsqueeze_4602 = torch.ops.aten.unsqueeze.default(arg1220_1, -1);  arg1220_1 = None
        unsqueeze_4603 = torch.ops.aten.unsqueeze.default(unsqueeze_4602, -1);  unsqueeze_4602 = None
        add_1654 = torch.ops.aten.add.Tensor(mul_1914, unsqueeze_4603);  mul_1914 = unsqueeze_4603 = None
        iota_104 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1915 = torch.ops.aten.mul.Tensor(iota_104, 1);  iota_104 = None
        add_1655 = torch.ops.aten.add.Tensor(mul_1915, 0);  mul_1915 = None
        convert_element_type_1346 = torch.ops.prims.convert_element_type.default(add_1655, torch.float32);  add_1655 = None
        add_1656 = torch.ops.aten.add.Tensor(convert_element_type_1346, 0.0);  convert_element_type_1346 = None
        mul_1916 = torch.ops.aten.mul.Tensor(add_1656, 0.125);  add_1656 = None
        convert_element_type_1347 = torch.ops.prims.convert_element_type.default(mul_1916, torch.int64);  mul_1916 = None
        unsqueeze_4604 = torch.ops.aten.unsqueeze.default(convert_element_type_1347, -1);  convert_element_type_1347 = None
        iota_105 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1917 = torch.ops.aten.mul.Tensor(iota_105, 1);  iota_105 = None
        add_1657 = torch.ops.aten.add.Tensor(mul_1917, 0);  mul_1917 = None
        convert_element_type_1348 = torch.ops.prims.convert_element_type.default(add_1657, torch.float32);  add_1657 = None
        add_1658 = torch.ops.aten.add.Tensor(convert_element_type_1348, 0.0);  convert_element_type_1348 = None
        mul_1918 = torch.ops.aten.mul.Tensor(add_1658, 0.125);  add_1658 = None
        convert_element_type_1349 = torch.ops.prims.convert_element_type.default(mul_1918, torch.int64);  mul_1918 = None
        _unsafe_index_52 = torch.ops.aten._unsafe_index.Tensor(add_1654, [None, None, unsqueeze_4604, convert_element_type_1349]);  add_1654 = unsqueeze_4604 = convert_element_type_1349 = None
        add_1659 = torch.ops.aten.add.Tensor(add_1652, _unsafe_index_52);  add_1652 = _unsafe_index_52 = None
        relu_504 = torch.ops.aten.relu.default(add_1659);  add_1659 = None
        convolution_569 = torch.ops.aten.convolution.default(relu_479, arg1221_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1221_1 = None
        add_1660 = torch.ops.aten.add.Tensor(arg1223_1, 1e-05);  arg1223_1 = None
        sqrt_569 = torch.ops.aten.sqrt.default(add_1660);  add_1660 = None
        reciprocal_569 = torch.ops.aten.reciprocal.default(sqrt_569);  sqrt_569 = None
        mul_1919 = torch.ops.aten.mul.Tensor(reciprocal_569, 1);  reciprocal_569 = None
        unsqueeze_4605 = torch.ops.aten.unsqueeze.default(arg1222_1, -1);  arg1222_1 = None
        unsqueeze_4606 = torch.ops.aten.unsqueeze.default(unsqueeze_4605, -1);  unsqueeze_4605 = None
        unsqueeze_4607 = torch.ops.aten.unsqueeze.default(mul_1919, -1);  mul_1919 = None
        unsqueeze_4608 = torch.ops.aten.unsqueeze.default(unsqueeze_4607, -1);  unsqueeze_4607 = None
        sub_569 = torch.ops.aten.sub.Tensor(convolution_569, unsqueeze_4606);  convolution_569 = unsqueeze_4606 = None
        mul_1920 = torch.ops.aten.mul.Tensor(sub_569, unsqueeze_4608);  sub_569 = unsqueeze_4608 = None
        unsqueeze_4609 = torch.ops.aten.unsqueeze.default(arg1224_1, -1);  arg1224_1 = None
        unsqueeze_4610 = torch.ops.aten.unsqueeze.default(unsqueeze_4609, -1);  unsqueeze_4609 = None
        mul_1921 = torch.ops.aten.mul.Tensor(mul_1920, unsqueeze_4610);  mul_1920 = unsqueeze_4610 = None
        unsqueeze_4611 = torch.ops.aten.unsqueeze.default(arg1225_1, -1);  arg1225_1 = None
        unsqueeze_4612 = torch.ops.aten.unsqueeze.default(unsqueeze_4611, -1);  unsqueeze_4611 = None
        add_1661 = torch.ops.aten.add.Tensor(mul_1921, unsqueeze_4612);  mul_1921 = unsqueeze_4612 = None
        add_1662 = torch.ops.aten.add.Tensor(add_1661, relu_487);  add_1661 = None
        convolution_570 = torch.ops.aten.convolution.default(relu_495, arg1226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1226_1 = None
        add_1663 = torch.ops.aten.add.Tensor(arg1228_1, 1e-05);  arg1228_1 = None
        sqrt_570 = torch.ops.aten.sqrt.default(add_1663);  add_1663 = None
        reciprocal_570 = torch.ops.aten.reciprocal.default(sqrt_570);  sqrt_570 = None
        mul_1922 = torch.ops.aten.mul.Tensor(reciprocal_570, 1);  reciprocal_570 = None
        unsqueeze_4613 = torch.ops.aten.unsqueeze.default(arg1227_1, -1);  arg1227_1 = None
        unsqueeze_4614 = torch.ops.aten.unsqueeze.default(unsqueeze_4613, -1);  unsqueeze_4613 = None
        unsqueeze_4615 = torch.ops.aten.unsqueeze.default(mul_1922, -1);  mul_1922 = None
        unsqueeze_4616 = torch.ops.aten.unsqueeze.default(unsqueeze_4615, -1);  unsqueeze_4615 = None
        sub_570 = torch.ops.aten.sub.Tensor(convolution_570, unsqueeze_4614);  convolution_570 = unsqueeze_4614 = None
        mul_1923 = torch.ops.aten.mul.Tensor(sub_570, unsqueeze_4616);  sub_570 = unsqueeze_4616 = None
        unsqueeze_4617 = torch.ops.aten.unsqueeze.default(arg1229_1, -1);  arg1229_1 = None
        unsqueeze_4618 = torch.ops.aten.unsqueeze.default(unsqueeze_4617, -1);  unsqueeze_4617 = None
        mul_1924 = torch.ops.aten.mul.Tensor(mul_1923, unsqueeze_4618);  mul_1923 = unsqueeze_4618 = None
        unsqueeze_4619 = torch.ops.aten.unsqueeze.default(arg1230_1, -1);  arg1230_1 = None
        unsqueeze_4620 = torch.ops.aten.unsqueeze.default(unsqueeze_4619, -1);  unsqueeze_4619 = None
        add_1664 = torch.ops.aten.add.Tensor(mul_1924, unsqueeze_4620);  mul_1924 = unsqueeze_4620 = None
        iota_106 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1925 = torch.ops.aten.mul.Tensor(iota_106, 1);  iota_106 = None
        add_1665 = torch.ops.aten.add.Tensor(mul_1925, 0);  mul_1925 = None
        convert_element_type_1354 = torch.ops.prims.convert_element_type.default(add_1665, torch.float32);  add_1665 = None
        add_1666 = torch.ops.aten.add.Tensor(convert_element_type_1354, 0.0);  convert_element_type_1354 = None
        mul_1926 = torch.ops.aten.mul.Tensor(add_1666, 0.5);  add_1666 = None
        convert_element_type_1355 = torch.ops.prims.convert_element_type.default(mul_1926, torch.int64);  mul_1926 = None
        unsqueeze_4621 = torch.ops.aten.unsqueeze.default(convert_element_type_1355, -1);  convert_element_type_1355 = None
        iota_107 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1927 = torch.ops.aten.mul.Tensor(iota_107, 1);  iota_107 = None
        add_1667 = torch.ops.aten.add.Tensor(mul_1927, 0);  mul_1927 = None
        convert_element_type_1356 = torch.ops.prims.convert_element_type.default(add_1667, torch.float32);  add_1667 = None
        add_1668 = torch.ops.aten.add.Tensor(convert_element_type_1356, 0.0);  convert_element_type_1356 = None
        mul_1928 = torch.ops.aten.mul.Tensor(add_1668, 0.5);  add_1668 = None
        convert_element_type_1357 = torch.ops.prims.convert_element_type.default(mul_1928, torch.int64);  mul_1928 = None
        _unsafe_index_53 = torch.ops.aten._unsafe_index.Tensor(add_1664, [None, None, unsqueeze_4621, convert_element_type_1357]);  add_1664 = unsqueeze_4621 = convert_element_type_1357 = None
        add_1669 = torch.ops.aten.add.Tensor(add_1662, _unsafe_index_53);  add_1662 = _unsafe_index_53 = None
        convolution_571 = torch.ops.aten.convolution.default(relu_503, arg1231_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1231_1 = None
        add_1670 = torch.ops.aten.add.Tensor(arg1233_1, 1e-05);  arg1233_1 = None
        sqrt_571 = torch.ops.aten.sqrt.default(add_1670);  add_1670 = None
        reciprocal_571 = torch.ops.aten.reciprocal.default(sqrt_571);  sqrt_571 = None
        mul_1929 = torch.ops.aten.mul.Tensor(reciprocal_571, 1);  reciprocal_571 = None
        unsqueeze_4622 = torch.ops.aten.unsqueeze.default(arg1232_1, -1);  arg1232_1 = None
        unsqueeze_4623 = torch.ops.aten.unsqueeze.default(unsqueeze_4622, -1);  unsqueeze_4622 = None
        unsqueeze_4624 = torch.ops.aten.unsqueeze.default(mul_1929, -1);  mul_1929 = None
        unsqueeze_4625 = torch.ops.aten.unsqueeze.default(unsqueeze_4624, -1);  unsqueeze_4624 = None
        sub_571 = torch.ops.aten.sub.Tensor(convolution_571, unsqueeze_4623);  convolution_571 = unsqueeze_4623 = None
        mul_1930 = torch.ops.aten.mul.Tensor(sub_571, unsqueeze_4625);  sub_571 = unsqueeze_4625 = None
        unsqueeze_4626 = torch.ops.aten.unsqueeze.default(arg1234_1, -1);  arg1234_1 = None
        unsqueeze_4627 = torch.ops.aten.unsqueeze.default(unsqueeze_4626, -1);  unsqueeze_4626 = None
        mul_1931 = torch.ops.aten.mul.Tensor(mul_1930, unsqueeze_4627);  mul_1930 = unsqueeze_4627 = None
        unsqueeze_4628 = torch.ops.aten.unsqueeze.default(arg1235_1, -1);  arg1235_1 = None
        unsqueeze_4629 = torch.ops.aten.unsqueeze.default(unsqueeze_4628, -1);  unsqueeze_4628 = None
        add_1671 = torch.ops.aten.add.Tensor(mul_1931, unsqueeze_4629);  mul_1931 = unsqueeze_4629 = None
        iota_108 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1932 = torch.ops.aten.mul.Tensor(iota_108, 1);  iota_108 = None
        add_1672 = torch.ops.aten.add.Tensor(mul_1932, 0);  mul_1932 = None
        convert_element_type_1360 = torch.ops.prims.convert_element_type.default(add_1672, torch.float32);  add_1672 = None
        add_1673 = torch.ops.aten.add.Tensor(convert_element_type_1360, 0.0);  convert_element_type_1360 = None
        mul_1933 = torch.ops.aten.mul.Tensor(add_1673, 0.25);  add_1673 = None
        convert_element_type_1361 = torch.ops.prims.convert_element_type.default(mul_1933, torch.int64);  mul_1933 = None
        unsqueeze_4630 = torch.ops.aten.unsqueeze.default(convert_element_type_1361, -1);  convert_element_type_1361 = None
        iota_109 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1934 = torch.ops.aten.mul.Tensor(iota_109, 1);  iota_109 = None
        add_1674 = torch.ops.aten.add.Tensor(mul_1934, 0);  mul_1934 = None
        convert_element_type_1362 = torch.ops.prims.convert_element_type.default(add_1674, torch.float32);  add_1674 = None
        add_1675 = torch.ops.aten.add.Tensor(convert_element_type_1362, 0.0);  convert_element_type_1362 = None
        mul_1935 = torch.ops.aten.mul.Tensor(add_1675, 0.25);  add_1675 = None
        convert_element_type_1363 = torch.ops.prims.convert_element_type.default(mul_1935, torch.int64);  mul_1935 = None
        _unsafe_index_54 = torch.ops.aten._unsafe_index.Tensor(add_1671, [None, None, unsqueeze_4630, convert_element_type_1363]);  add_1671 = unsqueeze_4630 = convert_element_type_1363 = None
        add_1676 = torch.ops.aten.add.Tensor(add_1669, _unsafe_index_54);  add_1669 = _unsafe_index_54 = None
        relu_505 = torch.ops.aten.relu.default(add_1676);  add_1676 = None
        convolution_572 = torch.ops.aten.convolution.default(relu_479, arg1236_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1236_1 = None
        add_1677 = torch.ops.aten.add.Tensor(arg1238_1, 1e-05);  arg1238_1 = None
        sqrt_572 = torch.ops.aten.sqrt.default(add_1677);  add_1677 = None
        reciprocal_572 = torch.ops.aten.reciprocal.default(sqrt_572);  sqrt_572 = None
        mul_1936 = torch.ops.aten.mul.Tensor(reciprocal_572, 1);  reciprocal_572 = None
        unsqueeze_4631 = torch.ops.aten.unsqueeze.default(arg1237_1, -1);  arg1237_1 = None
        unsqueeze_4632 = torch.ops.aten.unsqueeze.default(unsqueeze_4631, -1);  unsqueeze_4631 = None
        unsqueeze_4633 = torch.ops.aten.unsqueeze.default(mul_1936, -1);  mul_1936 = None
        unsqueeze_4634 = torch.ops.aten.unsqueeze.default(unsqueeze_4633, -1);  unsqueeze_4633 = None
        sub_572 = torch.ops.aten.sub.Tensor(convolution_572, unsqueeze_4632);  convolution_572 = unsqueeze_4632 = None
        mul_1937 = torch.ops.aten.mul.Tensor(sub_572, unsqueeze_4634);  sub_572 = unsqueeze_4634 = None
        unsqueeze_4635 = torch.ops.aten.unsqueeze.default(arg1239_1, -1);  arg1239_1 = None
        unsqueeze_4636 = torch.ops.aten.unsqueeze.default(unsqueeze_4635, -1);  unsqueeze_4635 = None
        mul_1938 = torch.ops.aten.mul.Tensor(mul_1937, unsqueeze_4636);  mul_1937 = unsqueeze_4636 = None
        unsqueeze_4637 = torch.ops.aten.unsqueeze.default(arg1240_1, -1);  arg1240_1 = None
        unsqueeze_4638 = torch.ops.aten.unsqueeze.default(unsqueeze_4637, -1);  unsqueeze_4637 = None
        add_1678 = torch.ops.aten.add.Tensor(mul_1938, unsqueeze_4638);  mul_1938 = unsqueeze_4638 = None
        relu_506 = torch.ops.aten.relu.default(add_1678);  add_1678 = None
        convolution_573 = torch.ops.aten.convolution.default(relu_506, arg1241_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_506 = arg1241_1 = None
        add_1679 = torch.ops.aten.add.Tensor(arg1243_1, 1e-05);  arg1243_1 = None
        sqrt_573 = torch.ops.aten.sqrt.default(add_1679);  add_1679 = None
        reciprocal_573 = torch.ops.aten.reciprocal.default(sqrt_573);  sqrt_573 = None
        mul_1939 = torch.ops.aten.mul.Tensor(reciprocal_573, 1);  reciprocal_573 = None
        unsqueeze_4639 = torch.ops.aten.unsqueeze.default(arg1242_1, -1);  arg1242_1 = None
        unsqueeze_4640 = torch.ops.aten.unsqueeze.default(unsqueeze_4639, -1);  unsqueeze_4639 = None
        unsqueeze_4641 = torch.ops.aten.unsqueeze.default(mul_1939, -1);  mul_1939 = None
        unsqueeze_4642 = torch.ops.aten.unsqueeze.default(unsqueeze_4641, -1);  unsqueeze_4641 = None
        sub_573 = torch.ops.aten.sub.Tensor(convolution_573, unsqueeze_4640);  convolution_573 = unsqueeze_4640 = None
        mul_1940 = torch.ops.aten.mul.Tensor(sub_573, unsqueeze_4642);  sub_573 = unsqueeze_4642 = None
        unsqueeze_4643 = torch.ops.aten.unsqueeze.default(arg1244_1, -1);  arg1244_1 = None
        unsqueeze_4644 = torch.ops.aten.unsqueeze.default(unsqueeze_4643, -1);  unsqueeze_4643 = None
        mul_1941 = torch.ops.aten.mul.Tensor(mul_1940, unsqueeze_4644);  mul_1940 = unsqueeze_4644 = None
        unsqueeze_4645 = torch.ops.aten.unsqueeze.default(arg1245_1, -1);  arg1245_1 = None
        unsqueeze_4646 = torch.ops.aten.unsqueeze.default(unsqueeze_4645, -1);  unsqueeze_4645 = None
        add_1680 = torch.ops.aten.add.Tensor(mul_1941, unsqueeze_4646);  mul_1941 = unsqueeze_4646 = None
        convolution_574 = torch.ops.aten.convolution.default(relu_487, arg1246_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1246_1 = None
        add_1681 = torch.ops.aten.add.Tensor(arg1248_1, 1e-05);  arg1248_1 = None
        sqrt_574 = torch.ops.aten.sqrt.default(add_1681);  add_1681 = None
        reciprocal_574 = torch.ops.aten.reciprocal.default(sqrt_574);  sqrt_574 = None
        mul_1942 = torch.ops.aten.mul.Tensor(reciprocal_574, 1);  reciprocal_574 = None
        unsqueeze_4647 = torch.ops.aten.unsqueeze.default(arg1247_1, -1);  arg1247_1 = None
        unsqueeze_4648 = torch.ops.aten.unsqueeze.default(unsqueeze_4647, -1);  unsqueeze_4647 = None
        unsqueeze_4649 = torch.ops.aten.unsqueeze.default(mul_1942, -1);  mul_1942 = None
        unsqueeze_4650 = torch.ops.aten.unsqueeze.default(unsqueeze_4649, -1);  unsqueeze_4649 = None
        sub_574 = torch.ops.aten.sub.Tensor(convolution_574, unsqueeze_4648);  convolution_574 = unsqueeze_4648 = None
        mul_1943 = torch.ops.aten.mul.Tensor(sub_574, unsqueeze_4650);  sub_574 = unsqueeze_4650 = None
        unsqueeze_4651 = torch.ops.aten.unsqueeze.default(arg1249_1, -1);  arg1249_1 = None
        unsqueeze_4652 = torch.ops.aten.unsqueeze.default(unsqueeze_4651, -1);  unsqueeze_4651 = None
        mul_1944 = torch.ops.aten.mul.Tensor(mul_1943, unsqueeze_4652);  mul_1943 = unsqueeze_4652 = None
        unsqueeze_4653 = torch.ops.aten.unsqueeze.default(arg1250_1, -1);  arg1250_1 = None
        unsqueeze_4654 = torch.ops.aten.unsqueeze.default(unsqueeze_4653, -1);  unsqueeze_4653 = None
        add_1682 = torch.ops.aten.add.Tensor(mul_1944, unsqueeze_4654);  mul_1944 = unsqueeze_4654 = None
        add_1683 = torch.ops.aten.add.Tensor(add_1680, add_1682);  add_1680 = add_1682 = None
        add_1684 = torch.ops.aten.add.Tensor(add_1683, relu_495);  add_1683 = None
        convolution_575 = torch.ops.aten.convolution.default(relu_503, arg1251_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1251_1 = None
        add_1685 = torch.ops.aten.add.Tensor(arg1253_1, 1e-05);  arg1253_1 = None
        sqrt_575 = torch.ops.aten.sqrt.default(add_1685);  add_1685 = None
        reciprocal_575 = torch.ops.aten.reciprocal.default(sqrt_575);  sqrt_575 = None
        mul_1945 = torch.ops.aten.mul.Tensor(reciprocal_575, 1);  reciprocal_575 = None
        unsqueeze_4655 = torch.ops.aten.unsqueeze.default(arg1252_1, -1);  arg1252_1 = None
        unsqueeze_4656 = torch.ops.aten.unsqueeze.default(unsqueeze_4655, -1);  unsqueeze_4655 = None
        unsqueeze_4657 = torch.ops.aten.unsqueeze.default(mul_1945, -1);  mul_1945 = None
        unsqueeze_4658 = torch.ops.aten.unsqueeze.default(unsqueeze_4657, -1);  unsqueeze_4657 = None
        sub_575 = torch.ops.aten.sub.Tensor(convolution_575, unsqueeze_4656);  convolution_575 = unsqueeze_4656 = None
        mul_1946 = torch.ops.aten.mul.Tensor(sub_575, unsqueeze_4658);  sub_575 = unsqueeze_4658 = None
        unsqueeze_4659 = torch.ops.aten.unsqueeze.default(arg1254_1, -1);  arg1254_1 = None
        unsqueeze_4660 = torch.ops.aten.unsqueeze.default(unsqueeze_4659, -1);  unsqueeze_4659 = None
        mul_1947 = torch.ops.aten.mul.Tensor(mul_1946, unsqueeze_4660);  mul_1946 = unsqueeze_4660 = None
        unsqueeze_4661 = torch.ops.aten.unsqueeze.default(arg1255_1, -1);  arg1255_1 = None
        unsqueeze_4662 = torch.ops.aten.unsqueeze.default(unsqueeze_4661, -1);  unsqueeze_4661 = None
        add_1686 = torch.ops.aten.add.Tensor(mul_1947, unsqueeze_4662);  mul_1947 = unsqueeze_4662 = None
        iota_110 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1948 = torch.ops.aten.mul.Tensor(iota_110, 1);  iota_110 = None
        add_1687 = torch.ops.aten.add.Tensor(mul_1948, 0);  mul_1948 = None
        convert_element_type_1372 = torch.ops.prims.convert_element_type.default(add_1687, torch.float32);  add_1687 = None
        add_1688 = torch.ops.aten.add.Tensor(convert_element_type_1372, 0.0);  convert_element_type_1372 = None
        mul_1949 = torch.ops.aten.mul.Tensor(add_1688, 0.5);  add_1688 = None
        convert_element_type_1373 = torch.ops.prims.convert_element_type.default(mul_1949, torch.int64);  mul_1949 = None
        unsqueeze_4663 = torch.ops.aten.unsqueeze.default(convert_element_type_1373, -1);  convert_element_type_1373 = None
        iota_111 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1950 = torch.ops.aten.mul.Tensor(iota_111, 1);  iota_111 = None
        add_1689 = torch.ops.aten.add.Tensor(mul_1950, 0);  mul_1950 = None
        convert_element_type_1374 = torch.ops.prims.convert_element_type.default(add_1689, torch.float32);  add_1689 = None
        add_1690 = torch.ops.aten.add.Tensor(convert_element_type_1374, 0.0);  convert_element_type_1374 = None
        mul_1951 = torch.ops.aten.mul.Tensor(add_1690, 0.5);  add_1690 = None
        convert_element_type_1375 = torch.ops.prims.convert_element_type.default(mul_1951, torch.int64);  mul_1951 = None
        _unsafe_index_55 = torch.ops.aten._unsafe_index.Tensor(add_1686, [None, None, unsqueeze_4663, convert_element_type_1375]);  add_1686 = unsqueeze_4663 = convert_element_type_1375 = None
        add_1691 = torch.ops.aten.add.Tensor(add_1684, _unsafe_index_55);  add_1684 = _unsafe_index_55 = None
        relu_507 = torch.ops.aten.relu.default(add_1691);  add_1691 = None
        convolution_576 = torch.ops.aten.convolution.default(relu_479, arg1256_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_479 = arg1256_1 = None
        add_1692 = torch.ops.aten.add.Tensor(arg1258_1, 1e-05);  arg1258_1 = None
        sqrt_576 = torch.ops.aten.sqrt.default(add_1692);  add_1692 = None
        reciprocal_576 = torch.ops.aten.reciprocal.default(sqrt_576);  sqrt_576 = None
        mul_1952 = torch.ops.aten.mul.Tensor(reciprocal_576, 1);  reciprocal_576 = None
        unsqueeze_4664 = torch.ops.aten.unsqueeze.default(arg1257_1, -1);  arg1257_1 = None
        unsqueeze_4665 = torch.ops.aten.unsqueeze.default(unsqueeze_4664, -1);  unsqueeze_4664 = None
        unsqueeze_4666 = torch.ops.aten.unsqueeze.default(mul_1952, -1);  mul_1952 = None
        unsqueeze_4667 = torch.ops.aten.unsqueeze.default(unsqueeze_4666, -1);  unsqueeze_4666 = None
        sub_576 = torch.ops.aten.sub.Tensor(convolution_576, unsqueeze_4665);  convolution_576 = unsqueeze_4665 = None
        mul_1953 = torch.ops.aten.mul.Tensor(sub_576, unsqueeze_4667);  sub_576 = unsqueeze_4667 = None
        unsqueeze_4668 = torch.ops.aten.unsqueeze.default(arg1259_1, -1);  arg1259_1 = None
        unsqueeze_4669 = torch.ops.aten.unsqueeze.default(unsqueeze_4668, -1);  unsqueeze_4668 = None
        mul_1954 = torch.ops.aten.mul.Tensor(mul_1953, unsqueeze_4669);  mul_1953 = unsqueeze_4669 = None
        unsqueeze_4670 = torch.ops.aten.unsqueeze.default(arg1260_1, -1);  arg1260_1 = None
        unsqueeze_4671 = torch.ops.aten.unsqueeze.default(unsqueeze_4670, -1);  unsqueeze_4670 = None
        add_1693 = torch.ops.aten.add.Tensor(mul_1954, unsqueeze_4671);  mul_1954 = unsqueeze_4671 = None
        relu_508 = torch.ops.aten.relu.default(add_1693);  add_1693 = None
        convolution_577 = torch.ops.aten.convolution.default(relu_508, arg1261_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_508 = arg1261_1 = None
        add_1694 = torch.ops.aten.add.Tensor(arg1263_1, 1e-05);  arg1263_1 = None
        sqrt_577 = torch.ops.aten.sqrt.default(add_1694);  add_1694 = None
        reciprocal_577 = torch.ops.aten.reciprocal.default(sqrt_577);  sqrt_577 = None
        mul_1955 = torch.ops.aten.mul.Tensor(reciprocal_577, 1);  reciprocal_577 = None
        unsqueeze_4672 = torch.ops.aten.unsqueeze.default(arg1262_1, -1);  arg1262_1 = None
        unsqueeze_4673 = torch.ops.aten.unsqueeze.default(unsqueeze_4672, -1);  unsqueeze_4672 = None
        unsqueeze_4674 = torch.ops.aten.unsqueeze.default(mul_1955, -1);  mul_1955 = None
        unsqueeze_4675 = torch.ops.aten.unsqueeze.default(unsqueeze_4674, -1);  unsqueeze_4674 = None
        sub_577 = torch.ops.aten.sub.Tensor(convolution_577, unsqueeze_4673);  convolution_577 = unsqueeze_4673 = None
        mul_1956 = torch.ops.aten.mul.Tensor(sub_577, unsqueeze_4675);  sub_577 = unsqueeze_4675 = None
        unsqueeze_4676 = torch.ops.aten.unsqueeze.default(arg1264_1, -1);  arg1264_1 = None
        unsqueeze_4677 = torch.ops.aten.unsqueeze.default(unsqueeze_4676, -1);  unsqueeze_4676 = None
        mul_1957 = torch.ops.aten.mul.Tensor(mul_1956, unsqueeze_4677);  mul_1956 = unsqueeze_4677 = None
        unsqueeze_4678 = torch.ops.aten.unsqueeze.default(arg1265_1, -1);  arg1265_1 = None
        unsqueeze_4679 = torch.ops.aten.unsqueeze.default(unsqueeze_4678, -1);  unsqueeze_4678 = None
        add_1695 = torch.ops.aten.add.Tensor(mul_1957, unsqueeze_4679);  mul_1957 = unsqueeze_4679 = None
        relu_509 = torch.ops.aten.relu.default(add_1695);  add_1695 = None
        convolution_578 = torch.ops.aten.convolution.default(relu_509, arg1266_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_509 = arg1266_1 = None
        add_1696 = torch.ops.aten.add.Tensor(arg1268_1, 1e-05);  arg1268_1 = None
        sqrt_578 = torch.ops.aten.sqrt.default(add_1696);  add_1696 = None
        reciprocal_578 = torch.ops.aten.reciprocal.default(sqrt_578);  sqrt_578 = None
        mul_1958 = torch.ops.aten.mul.Tensor(reciprocal_578, 1);  reciprocal_578 = None
        unsqueeze_4680 = torch.ops.aten.unsqueeze.default(arg1267_1, -1);  arg1267_1 = None
        unsqueeze_4681 = torch.ops.aten.unsqueeze.default(unsqueeze_4680, -1);  unsqueeze_4680 = None
        unsqueeze_4682 = torch.ops.aten.unsqueeze.default(mul_1958, -1);  mul_1958 = None
        unsqueeze_4683 = torch.ops.aten.unsqueeze.default(unsqueeze_4682, -1);  unsqueeze_4682 = None
        sub_578 = torch.ops.aten.sub.Tensor(convolution_578, unsqueeze_4681);  convolution_578 = unsqueeze_4681 = None
        mul_1959 = torch.ops.aten.mul.Tensor(sub_578, unsqueeze_4683);  sub_578 = unsqueeze_4683 = None
        unsqueeze_4684 = torch.ops.aten.unsqueeze.default(arg1269_1, -1);  arg1269_1 = None
        unsqueeze_4685 = torch.ops.aten.unsqueeze.default(unsqueeze_4684, -1);  unsqueeze_4684 = None
        mul_1960 = torch.ops.aten.mul.Tensor(mul_1959, unsqueeze_4685);  mul_1959 = unsqueeze_4685 = None
        unsqueeze_4686 = torch.ops.aten.unsqueeze.default(arg1270_1, -1);  arg1270_1 = None
        unsqueeze_4687 = torch.ops.aten.unsqueeze.default(unsqueeze_4686, -1);  unsqueeze_4686 = None
        add_1697 = torch.ops.aten.add.Tensor(mul_1960, unsqueeze_4687);  mul_1960 = unsqueeze_4687 = None
        convolution_579 = torch.ops.aten.convolution.default(relu_487, arg1271_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_487 = arg1271_1 = None
        add_1698 = torch.ops.aten.add.Tensor(arg1273_1, 1e-05);  arg1273_1 = None
        sqrt_579 = torch.ops.aten.sqrt.default(add_1698);  add_1698 = None
        reciprocal_579 = torch.ops.aten.reciprocal.default(sqrt_579);  sqrt_579 = None
        mul_1961 = torch.ops.aten.mul.Tensor(reciprocal_579, 1);  reciprocal_579 = None
        unsqueeze_4688 = torch.ops.aten.unsqueeze.default(arg1272_1, -1);  arg1272_1 = None
        unsqueeze_4689 = torch.ops.aten.unsqueeze.default(unsqueeze_4688, -1);  unsqueeze_4688 = None
        unsqueeze_4690 = torch.ops.aten.unsqueeze.default(mul_1961, -1);  mul_1961 = None
        unsqueeze_4691 = torch.ops.aten.unsqueeze.default(unsqueeze_4690, -1);  unsqueeze_4690 = None
        sub_579 = torch.ops.aten.sub.Tensor(convolution_579, unsqueeze_4689);  convolution_579 = unsqueeze_4689 = None
        mul_1962 = torch.ops.aten.mul.Tensor(sub_579, unsqueeze_4691);  sub_579 = unsqueeze_4691 = None
        unsqueeze_4692 = torch.ops.aten.unsqueeze.default(arg1274_1, -1);  arg1274_1 = None
        unsqueeze_4693 = torch.ops.aten.unsqueeze.default(unsqueeze_4692, -1);  unsqueeze_4692 = None
        mul_1963 = torch.ops.aten.mul.Tensor(mul_1962, unsqueeze_4693);  mul_1962 = unsqueeze_4693 = None
        unsqueeze_4694 = torch.ops.aten.unsqueeze.default(arg1275_1, -1);  arg1275_1 = None
        unsqueeze_4695 = torch.ops.aten.unsqueeze.default(unsqueeze_4694, -1);  unsqueeze_4694 = None
        add_1699 = torch.ops.aten.add.Tensor(mul_1963, unsqueeze_4695);  mul_1963 = unsqueeze_4695 = None
        relu_510 = torch.ops.aten.relu.default(add_1699);  add_1699 = None
        convolution_580 = torch.ops.aten.convolution.default(relu_510, arg1276_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_510 = arg1276_1 = None
        add_1700 = torch.ops.aten.add.Tensor(arg1278_1, 1e-05);  arg1278_1 = None
        sqrt_580 = torch.ops.aten.sqrt.default(add_1700);  add_1700 = None
        reciprocal_580 = torch.ops.aten.reciprocal.default(sqrt_580);  sqrt_580 = None
        mul_1964 = torch.ops.aten.mul.Tensor(reciprocal_580, 1);  reciprocal_580 = None
        unsqueeze_4696 = torch.ops.aten.unsqueeze.default(arg1277_1, -1);  arg1277_1 = None
        unsqueeze_4697 = torch.ops.aten.unsqueeze.default(unsqueeze_4696, -1);  unsqueeze_4696 = None
        unsqueeze_4698 = torch.ops.aten.unsqueeze.default(mul_1964, -1);  mul_1964 = None
        unsqueeze_4699 = torch.ops.aten.unsqueeze.default(unsqueeze_4698, -1);  unsqueeze_4698 = None
        sub_580 = torch.ops.aten.sub.Tensor(convolution_580, unsqueeze_4697);  convolution_580 = unsqueeze_4697 = None
        mul_1965 = torch.ops.aten.mul.Tensor(sub_580, unsqueeze_4699);  sub_580 = unsqueeze_4699 = None
        unsqueeze_4700 = torch.ops.aten.unsqueeze.default(arg1279_1, -1);  arg1279_1 = None
        unsqueeze_4701 = torch.ops.aten.unsqueeze.default(unsqueeze_4700, -1);  unsqueeze_4700 = None
        mul_1966 = torch.ops.aten.mul.Tensor(mul_1965, unsqueeze_4701);  mul_1965 = unsqueeze_4701 = None
        unsqueeze_4702 = torch.ops.aten.unsqueeze.default(arg1280_1, -1);  arg1280_1 = None
        unsqueeze_4703 = torch.ops.aten.unsqueeze.default(unsqueeze_4702, -1);  unsqueeze_4702 = None
        add_1701 = torch.ops.aten.add.Tensor(mul_1966, unsqueeze_4703);  mul_1966 = unsqueeze_4703 = None
        add_1702 = torch.ops.aten.add.Tensor(add_1697, add_1701);  add_1697 = add_1701 = None
        convolution_581 = torch.ops.aten.convolution.default(relu_495, arg1281_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_495 = arg1281_1 = None
        add_1703 = torch.ops.aten.add.Tensor(arg1283_1, 1e-05);  arg1283_1 = None
        sqrt_581 = torch.ops.aten.sqrt.default(add_1703);  add_1703 = None
        reciprocal_581 = torch.ops.aten.reciprocal.default(sqrt_581);  sqrt_581 = None
        mul_1967 = torch.ops.aten.mul.Tensor(reciprocal_581, 1);  reciprocal_581 = None
        unsqueeze_4704 = torch.ops.aten.unsqueeze.default(arg1282_1, -1);  arg1282_1 = None
        unsqueeze_4705 = torch.ops.aten.unsqueeze.default(unsqueeze_4704, -1);  unsqueeze_4704 = None
        unsqueeze_4706 = torch.ops.aten.unsqueeze.default(mul_1967, -1);  mul_1967 = None
        unsqueeze_4707 = torch.ops.aten.unsqueeze.default(unsqueeze_4706, -1);  unsqueeze_4706 = None
        sub_581 = torch.ops.aten.sub.Tensor(convolution_581, unsqueeze_4705);  convolution_581 = unsqueeze_4705 = None
        mul_1968 = torch.ops.aten.mul.Tensor(sub_581, unsqueeze_4707);  sub_581 = unsqueeze_4707 = None
        unsqueeze_4708 = torch.ops.aten.unsqueeze.default(arg1284_1, -1);  arg1284_1 = None
        unsqueeze_4709 = torch.ops.aten.unsqueeze.default(unsqueeze_4708, -1);  unsqueeze_4708 = None
        mul_1969 = torch.ops.aten.mul.Tensor(mul_1968, unsqueeze_4709);  mul_1968 = unsqueeze_4709 = None
        unsqueeze_4710 = torch.ops.aten.unsqueeze.default(arg1285_1, -1);  arg1285_1 = None
        unsqueeze_4711 = torch.ops.aten.unsqueeze.default(unsqueeze_4710, -1);  unsqueeze_4710 = None
        add_1704 = torch.ops.aten.add.Tensor(mul_1969, unsqueeze_4711);  mul_1969 = unsqueeze_4711 = None
        add_1705 = torch.ops.aten.add.Tensor(add_1702, add_1704);  add_1702 = add_1704 = None
        add_1706 = torch.ops.aten.add.Tensor(add_1705, relu_503);  add_1705 = relu_503 = None
        relu_511 = torch.ops.aten.relu.default(add_1706);  add_1706 = None
        convolution_582 = torch.ops.aten.convolution.default(relu_504, arg1286_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1286_1 = None
        add_1707 = torch.ops.aten.add.Tensor(arg1288_1, 1e-05);  arg1288_1 = None
        sqrt_582 = torch.ops.aten.sqrt.default(add_1707);  add_1707 = None
        reciprocal_582 = torch.ops.aten.reciprocal.default(sqrt_582);  sqrt_582 = None
        mul_1970 = torch.ops.aten.mul.Tensor(reciprocal_582, 1);  reciprocal_582 = None
        unsqueeze_4712 = torch.ops.aten.unsqueeze.default(arg1287_1, -1);  arg1287_1 = None
        unsqueeze_4713 = torch.ops.aten.unsqueeze.default(unsqueeze_4712, -1);  unsqueeze_4712 = None
        unsqueeze_4714 = torch.ops.aten.unsqueeze.default(mul_1970, -1);  mul_1970 = None
        unsqueeze_4715 = torch.ops.aten.unsqueeze.default(unsqueeze_4714, -1);  unsqueeze_4714 = None
        sub_582 = torch.ops.aten.sub.Tensor(convolution_582, unsqueeze_4713);  convolution_582 = unsqueeze_4713 = None
        mul_1971 = torch.ops.aten.mul.Tensor(sub_582, unsqueeze_4715);  sub_582 = unsqueeze_4715 = None
        unsqueeze_4716 = torch.ops.aten.unsqueeze.default(arg1289_1, -1);  arg1289_1 = None
        unsqueeze_4717 = torch.ops.aten.unsqueeze.default(unsqueeze_4716, -1);  unsqueeze_4716 = None
        mul_1972 = torch.ops.aten.mul.Tensor(mul_1971, unsqueeze_4717);  mul_1971 = unsqueeze_4717 = None
        unsqueeze_4718 = torch.ops.aten.unsqueeze.default(arg1290_1, -1);  arg1290_1 = None
        unsqueeze_4719 = torch.ops.aten.unsqueeze.default(unsqueeze_4718, -1);  unsqueeze_4718 = None
        add_1708 = torch.ops.aten.add.Tensor(mul_1972, unsqueeze_4719);  mul_1972 = unsqueeze_4719 = None
        relu_512 = torch.ops.aten.relu.default(add_1708);  add_1708 = None
        convolution_583 = torch.ops.aten.convolution.default(relu_512, arg1291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_512 = arg1291_1 = None
        add_1709 = torch.ops.aten.add.Tensor(arg1293_1, 1e-05);  arg1293_1 = None
        sqrt_583 = torch.ops.aten.sqrt.default(add_1709);  add_1709 = None
        reciprocal_583 = torch.ops.aten.reciprocal.default(sqrt_583);  sqrt_583 = None
        mul_1973 = torch.ops.aten.mul.Tensor(reciprocal_583, 1);  reciprocal_583 = None
        unsqueeze_4720 = torch.ops.aten.unsqueeze.default(arg1292_1, -1);  arg1292_1 = None
        unsqueeze_4721 = torch.ops.aten.unsqueeze.default(unsqueeze_4720, -1);  unsqueeze_4720 = None
        unsqueeze_4722 = torch.ops.aten.unsqueeze.default(mul_1973, -1);  mul_1973 = None
        unsqueeze_4723 = torch.ops.aten.unsqueeze.default(unsqueeze_4722, -1);  unsqueeze_4722 = None
        sub_583 = torch.ops.aten.sub.Tensor(convolution_583, unsqueeze_4721);  convolution_583 = unsqueeze_4721 = None
        mul_1974 = torch.ops.aten.mul.Tensor(sub_583, unsqueeze_4723);  sub_583 = unsqueeze_4723 = None
        unsqueeze_4724 = torch.ops.aten.unsqueeze.default(arg1294_1, -1);  arg1294_1 = None
        unsqueeze_4725 = torch.ops.aten.unsqueeze.default(unsqueeze_4724, -1);  unsqueeze_4724 = None
        mul_1975 = torch.ops.aten.mul.Tensor(mul_1974, unsqueeze_4725);  mul_1974 = unsqueeze_4725 = None
        unsqueeze_4726 = torch.ops.aten.unsqueeze.default(arg1295_1, -1);  arg1295_1 = None
        unsqueeze_4727 = torch.ops.aten.unsqueeze.default(unsqueeze_4726, -1);  unsqueeze_4726 = None
        add_1710 = torch.ops.aten.add.Tensor(mul_1975, unsqueeze_4727);  mul_1975 = unsqueeze_4727 = None
        add_1711 = torch.ops.aten.add.Tensor(add_1710, relu_504);  add_1710 = relu_504 = None
        relu_513 = torch.ops.aten.relu.default(add_1711);  add_1711 = None
        convolution_584 = torch.ops.aten.convolution.default(relu_513, arg1296_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1296_1 = None
        add_1712 = torch.ops.aten.add.Tensor(arg1298_1, 1e-05);  arg1298_1 = None
        sqrt_584 = torch.ops.aten.sqrt.default(add_1712);  add_1712 = None
        reciprocal_584 = torch.ops.aten.reciprocal.default(sqrt_584);  sqrt_584 = None
        mul_1976 = torch.ops.aten.mul.Tensor(reciprocal_584, 1);  reciprocal_584 = None
        unsqueeze_4728 = torch.ops.aten.unsqueeze.default(arg1297_1, -1);  arg1297_1 = None
        unsqueeze_4729 = torch.ops.aten.unsqueeze.default(unsqueeze_4728, -1);  unsqueeze_4728 = None
        unsqueeze_4730 = torch.ops.aten.unsqueeze.default(mul_1976, -1);  mul_1976 = None
        unsqueeze_4731 = torch.ops.aten.unsqueeze.default(unsqueeze_4730, -1);  unsqueeze_4730 = None
        sub_584 = torch.ops.aten.sub.Tensor(convolution_584, unsqueeze_4729);  convolution_584 = unsqueeze_4729 = None
        mul_1977 = torch.ops.aten.mul.Tensor(sub_584, unsqueeze_4731);  sub_584 = unsqueeze_4731 = None
        unsqueeze_4732 = torch.ops.aten.unsqueeze.default(arg1299_1, -1);  arg1299_1 = None
        unsqueeze_4733 = torch.ops.aten.unsqueeze.default(unsqueeze_4732, -1);  unsqueeze_4732 = None
        mul_1978 = torch.ops.aten.mul.Tensor(mul_1977, unsqueeze_4733);  mul_1977 = unsqueeze_4733 = None
        unsqueeze_4734 = torch.ops.aten.unsqueeze.default(arg1300_1, -1);  arg1300_1 = None
        unsqueeze_4735 = torch.ops.aten.unsqueeze.default(unsqueeze_4734, -1);  unsqueeze_4734 = None
        add_1713 = torch.ops.aten.add.Tensor(mul_1978, unsqueeze_4735);  mul_1978 = unsqueeze_4735 = None
        relu_514 = torch.ops.aten.relu.default(add_1713);  add_1713 = None
        convolution_585 = torch.ops.aten.convolution.default(relu_514, arg1301_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_514 = arg1301_1 = None
        add_1714 = torch.ops.aten.add.Tensor(arg1303_1, 1e-05);  arg1303_1 = None
        sqrt_585 = torch.ops.aten.sqrt.default(add_1714);  add_1714 = None
        reciprocal_585 = torch.ops.aten.reciprocal.default(sqrt_585);  sqrt_585 = None
        mul_1979 = torch.ops.aten.mul.Tensor(reciprocal_585, 1);  reciprocal_585 = None
        unsqueeze_4736 = torch.ops.aten.unsqueeze.default(arg1302_1, -1);  arg1302_1 = None
        unsqueeze_4737 = torch.ops.aten.unsqueeze.default(unsqueeze_4736, -1);  unsqueeze_4736 = None
        unsqueeze_4738 = torch.ops.aten.unsqueeze.default(mul_1979, -1);  mul_1979 = None
        unsqueeze_4739 = torch.ops.aten.unsqueeze.default(unsqueeze_4738, -1);  unsqueeze_4738 = None
        sub_585 = torch.ops.aten.sub.Tensor(convolution_585, unsqueeze_4737);  convolution_585 = unsqueeze_4737 = None
        mul_1980 = torch.ops.aten.mul.Tensor(sub_585, unsqueeze_4739);  sub_585 = unsqueeze_4739 = None
        unsqueeze_4740 = torch.ops.aten.unsqueeze.default(arg1304_1, -1);  arg1304_1 = None
        unsqueeze_4741 = torch.ops.aten.unsqueeze.default(unsqueeze_4740, -1);  unsqueeze_4740 = None
        mul_1981 = torch.ops.aten.mul.Tensor(mul_1980, unsqueeze_4741);  mul_1980 = unsqueeze_4741 = None
        unsqueeze_4742 = torch.ops.aten.unsqueeze.default(arg1305_1, -1);  arg1305_1 = None
        unsqueeze_4743 = torch.ops.aten.unsqueeze.default(unsqueeze_4742, -1);  unsqueeze_4742 = None
        add_1715 = torch.ops.aten.add.Tensor(mul_1981, unsqueeze_4743);  mul_1981 = unsqueeze_4743 = None
        add_1716 = torch.ops.aten.add.Tensor(add_1715, relu_513);  add_1715 = relu_513 = None
        relu_515 = torch.ops.aten.relu.default(add_1716);  add_1716 = None
        convolution_586 = torch.ops.aten.convolution.default(relu_515, arg1306_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1306_1 = None
        add_1717 = torch.ops.aten.add.Tensor(arg1308_1, 1e-05);  arg1308_1 = None
        sqrt_586 = torch.ops.aten.sqrt.default(add_1717);  add_1717 = None
        reciprocal_586 = torch.ops.aten.reciprocal.default(sqrt_586);  sqrt_586 = None
        mul_1982 = torch.ops.aten.mul.Tensor(reciprocal_586, 1);  reciprocal_586 = None
        unsqueeze_4744 = torch.ops.aten.unsqueeze.default(arg1307_1, -1);  arg1307_1 = None
        unsqueeze_4745 = torch.ops.aten.unsqueeze.default(unsqueeze_4744, -1);  unsqueeze_4744 = None
        unsqueeze_4746 = torch.ops.aten.unsqueeze.default(mul_1982, -1);  mul_1982 = None
        unsqueeze_4747 = torch.ops.aten.unsqueeze.default(unsqueeze_4746, -1);  unsqueeze_4746 = None
        sub_586 = torch.ops.aten.sub.Tensor(convolution_586, unsqueeze_4745);  convolution_586 = unsqueeze_4745 = None
        mul_1983 = torch.ops.aten.mul.Tensor(sub_586, unsqueeze_4747);  sub_586 = unsqueeze_4747 = None
        unsqueeze_4748 = torch.ops.aten.unsqueeze.default(arg1309_1, -1);  arg1309_1 = None
        unsqueeze_4749 = torch.ops.aten.unsqueeze.default(unsqueeze_4748, -1);  unsqueeze_4748 = None
        mul_1984 = torch.ops.aten.mul.Tensor(mul_1983, unsqueeze_4749);  mul_1983 = unsqueeze_4749 = None
        unsqueeze_4750 = torch.ops.aten.unsqueeze.default(arg1310_1, -1);  arg1310_1 = None
        unsqueeze_4751 = torch.ops.aten.unsqueeze.default(unsqueeze_4750, -1);  unsqueeze_4750 = None
        add_1718 = torch.ops.aten.add.Tensor(mul_1984, unsqueeze_4751);  mul_1984 = unsqueeze_4751 = None
        relu_516 = torch.ops.aten.relu.default(add_1718);  add_1718 = None
        convolution_587 = torch.ops.aten.convolution.default(relu_516, arg1311_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_516 = arg1311_1 = None
        add_1719 = torch.ops.aten.add.Tensor(arg1313_1, 1e-05);  arg1313_1 = None
        sqrt_587 = torch.ops.aten.sqrt.default(add_1719);  add_1719 = None
        reciprocal_587 = torch.ops.aten.reciprocal.default(sqrt_587);  sqrt_587 = None
        mul_1985 = torch.ops.aten.mul.Tensor(reciprocal_587, 1);  reciprocal_587 = None
        unsqueeze_4752 = torch.ops.aten.unsqueeze.default(arg1312_1, -1);  arg1312_1 = None
        unsqueeze_4753 = torch.ops.aten.unsqueeze.default(unsqueeze_4752, -1);  unsqueeze_4752 = None
        unsqueeze_4754 = torch.ops.aten.unsqueeze.default(mul_1985, -1);  mul_1985 = None
        unsqueeze_4755 = torch.ops.aten.unsqueeze.default(unsqueeze_4754, -1);  unsqueeze_4754 = None
        sub_587 = torch.ops.aten.sub.Tensor(convolution_587, unsqueeze_4753);  convolution_587 = unsqueeze_4753 = None
        mul_1986 = torch.ops.aten.mul.Tensor(sub_587, unsqueeze_4755);  sub_587 = unsqueeze_4755 = None
        unsqueeze_4756 = torch.ops.aten.unsqueeze.default(arg1314_1, -1);  arg1314_1 = None
        unsqueeze_4757 = torch.ops.aten.unsqueeze.default(unsqueeze_4756, -1);  unsqueeze_4756 = None
        mul_1987 = torch.ops.aten.mul.Tensor(mul_1986, unsqueeze_4757);  mul_1986 = unsqueeze_4757 = None
        unsqueeze_4758 = torch.ops.aten.unsqueeze.default(arg1315_1, -1);  arg1315_1 = None
        unsqueeze_4759 = torch.ops.aten.unsqueeze.default(unsqueeze_4758, -1);  unsqueeze_4758 = None
        add_1720 = torch.ops.aten.add.Tensor(mul_1987, unsqueeze_4759);  mul_1987 = unsqueeze_4759 = None
        add_1721 = torch.ops.aten.add.Tensor(add_1720, relu_515);  add_1720 = relu_515 = None
        relu_517 = torch.ops.aten.relu.default(add_1721);  add_1721 = None
        convolution_588 = torch.ops.aten.convolution.default(relu_517, arg1316_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1316_1 = None
        add_1722 = torch.ops.aten.add.Tensor(arg1318_1, 1e-05);  arg1318_1 = None
        sqrt_588 = torch.ops.aten.sqrt.default(add_1722);  add_1722 = None
        reciprocal_588 = torch.ops.aten.reciprocal.default(sqrt_588);  sqrt_588 = None
        mul_1988 = torch.ops.aten.mul.Tensor(reciprocal_588, 1);  reciprocal_588 = None
        unsqueeze_4760 = torch.ops.aten.unsqueeze.default(arg1317_1, -1);  arg1317_1 = None
        unsqueeze_4761 = torch.ops.aten.unsqueeze.default(unsqueeze_4760, -1);  unsqueeze_4760 = None
        unsqueeze_4762 = torch.ops.aten.unsqueeze.default(mul_1988, -1);  mul_1988 = None
        unsqueeze_4763 = torch.ops.aten.unsqueeze.default(unsqueeze_4762, -1);  unsqueeze_4762 = None
        sub_588 = torch.ops.aten.sub.Tensor(convolution_588, unsqueeze_4761);  convolution_588 = unsqueeze_4761 = None
        mul_1989 = torch.ops.aten.mul.Tensor(sub_588, unsqueeze_4763);  sub_588 = unsqueeze_4763 = None
        unsqueeze_4764 = torch.ops.aten.unsqueeze.default(arg1319_1, -1);  arg1319_1 = None
        unsqueeze_4765 = torch.ops.aten.unsqueeze.default(unsqueeze_4764, -1);  unsqueeze_4764 = None
        mul_1990 = torch.ops.aten.mul.Tensor(mul_1989, unsqueeze_4765);  mul_1989 = unsqueeze_4765 = None
        unsqueeze_4766 = torch.ops.aten.unsqueeze.default(arg1320_1, -1);  arg1320_1 = None
        unsqueeze_4767 = torch.ops.aten.unsqueeze.default(unsqueeze_4766, -1);  unsqueeze_4766 = None
        add_1723 = torch.ops.aten.add.Tensor(mul_1990, unsqueeze_4767);  mul_1990 = unsqueeze_4767 = None
        relu_518 = torch.ops.aten.relu.default(add_1723);  add_1723 = None
        convolution_589 = torch.ops.aten.convolution.default(relu_518, arg1321_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_518 = arg1321_1 = None
        add_1724 = torch.ops.aten.add.Tensor(arg1323_1, 1e-05);  arg1323_1 = None
        sqrt_589 = torch.ops.aten.sqrt.default(add_1724);  add_1724 = None
        reciprocal_589 = torch.ops.aten.reciprocal.default(sqrt_589);  sqrt_589 = None
        mul_1991 = torch.ops.aten.mul.Tensor(reciprocal_589, 1);  reciprocal_589 = None
        unsqueeze_4768 = torch.ops.aten.unsqueeze.default(arg1322_1, -1);  arg1322_1 = None
        unsqueeze_4769 = torch.ops.aten.unsqueeze.default(unsqueeze_4768, -1);  unsqueeze_4768 = None
        unsqueeze_4770 = torch.ops.aten.unsqueeze.default(mul_1991, -1);  mul_1991 = None
        unsqueeze_4771 = torch.ops.aten.unsqueeze.default(unsqueeze_4770, -1);  unsqueeze_4770 = None
        sub_589 = torch.ops.aten.sub.Tensor(convolution_589, unsqueeze_4769);  convolution_589 = unsqueeze_4769 = None
        mul_1992 = torch.ops.aten.mul.Tensor(sub_589, unsqueeze_4771);  sub_589 = unsqueeze_4771 = None
        unsqueeze_4772 = torch.ops.aten.unsqueeze.default(arg1324_1, -1);  arg1324_1 = None
        unsqueeze_4773 = torch.ops.aten.unsqueeze.default(unsqueeze_4772, -1);  unsqueeze_4772 = None
        mul_1993 = torch.ops.aten.mul.Tensor(mul_1992, unsqueeze_4773);  mul_1992 = unsqueeze_4773 = None
        unsqueeze_4774 = torch.ops.aten.unsqueeze.default(arg1325_1, -1);  arg1325_1 = None
        unsqueeze_4775 = torch.ops.aten.unsqueeze.default(unsqueeze_4774, -1);  unsqueeze_4774 = None
        add_1725 = torch.ops.aten.add.Tensor(mul_1993, unsqueeze_4775);  mul_1993 = unsqueeze_4775 = None
        add_1726 = torch.ops.aten.add.Tensor(add_1725, relu_517);  add_1725 = relu_517 = None
        relu_519 = torch.ops.aten.relu.default(add_1726);  add_1726 = None
        convolution_590 = torch.ops.aten.convolution.default(relu_505, arg1326_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1326_1 = None
        add_1727 = torch.ops.aten.add.Tensor(arg1328_1, 1e-05);  arg1328_1 = None
        sqrt_590 = torch.ops.aten.sqrt.default(add_1727);  add_1727 = None
        reciprocal_590 = torch.ops.aten.reciprocal.default(sqrt_590);  sqrt_590 = None
        mul_1994 = torch.ops.aten.mul.Tensor(reciprocal_590, 1);  reciprocal_590 = None
        unsqueeze_4776 = torch.ops.aten.unsqueeze.default(arg1327_1, -1);  arg1327_1 = None
        unsqueeze_4777 = torch.ops.aten.unsqueeze.default(unsqueeze_4776, -1);  unsqueeze_4776 = None
        unsqueeze_4778 = torch.ops.aten.unsqueeze.default(mul_1994, -1);  mul_1994 = None
        unsqueeze_4779 = torch.ops.aten.unsqueeze.default(unsqueeze_4778, -1);  unsqueeze_4778 = None
        sub_590 = torch.ops.aten.sub.Tensor(convolution_590, unsqueeze_4777);  convolution_590 = unsqueeze_4777 = None
        mul_1995 = torch.ops.aten.mul.Tensor(sub_590, unsqueeze_4779);  sub_590 = unsqueeze_4779 = None
        unsqueeze_4780 = torch.ops.aten.unsqueeze.default(arg1329_1, -1);  arg1329_1 = None
        unsqueeze_4781 = torch.ops.aten.unsqueeze.default(unsqueeze_4780, -1);  unsqueeze_4780 = None
        mul_1996 = torch.ops.aten.mul.Tensor(mul_1995, unsqueeze_4781);  mul_1995 = unsqueeze_4781 = None
        unsqueeze_4782 = torch.ops.aten.unsqueeze.default(arg1330_1, -1);  arg1330_1 = None
        unsqueeze_4783 = torch.ops.aten.unsqueeze.default(unsqueeze_4782, -1);  unsqueeze_4782 = None
        add_1728 = torch.ops.aten.add.Tensor(mul_1996, unsqueeze_4783);  mul_1996 = unsqueeze_4783 = None
        relu_520 = torch.ops.aten.relu.default(add_1728);  add_1728 = None
        convolution_591 = torch.ops.aten.convolution.default(relu_520, arg1331_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_520 = arg1331_1 = None
        add_1729 = torch.ops.aten.add.Tensor(arg1333_1, 1e-05);  arg1333_1 = None
        sqrt_591 = torch.ops.aten.sqrt.default(add_1729);  add_1729 = None
        reciprocal_591 = torch.ops.aten.reciprocal.default(sqrt_591);  sqrt_591 = None
        mul_1997 = torch.ops.aten.mul.Tensor(reciprocal_591, 1);  reciprocal_591 = None
        unsqueeze_4784 = torch.ops.aten.unsqueeze.default(arg1332_1, -1);  arg1332_1 = None
        unsqueeze_4785 = torch.ops.aten.unsqueeze.default(unsqueeze_4784, -1);  unsqueeze_4784 = None
        unsqueeze_4786 = torch.ops.aten.unsqueeze.default(mul_1997, -1);  mul_1997 = None
        unsqueeze_4787 = torch.ops.aten.unsqueeze.default(unsqueeze_4786, -1);  unsqueeze_4786 = None
        sub_591 = torch.ops.aten.sub.Tensor(convolution_591, unsqueeze_4785);  convolution_591 = unsqueeze_4785 = None
        mul_1998 = torch.ops.aten.mul.Tensor(sub_591, unsqueeze_4787);  sub_591 = unsqueeze_4787 = None
        unsqueeze_4788 = torch.ops.aten.unsqueeze.default(arg1334_1, -1);  arg1334_1 = None
        unsqueeze_4789 = torch.ops.aten.unsqueeze.default(unsqueeze_4788, -1);  unsqueeze_4788 = None
        mul_1999 = torch.ops.aten.mul.Tensor(mul_1998, unsqueeze_4789);  mul_1998 = unsqueeze_4789 = None
        unsqueeze_4790 = torch.ops.aten.unsqueeze.default(arg1335_1, -1);  arg1335_1 = None
        unsqueeze_4791 = torch.ops.aten.unsqueeze.default(unsqueeze_4790, -1);  unsqueeze_4790 = None
        add_1730 = torch.ops.aten.add.Tensor(mul_1999, unsqueeze_4791);  mul_1999 = unsqueeze_4791 = None
        add_1731 = torch.ops.aten.add.Tensor(add_1730, relu_505);  add_1730 = relu_505 = None
        relu_521 = torch.ops.aten.relu.default(add_1731);  add_1731 = None
        convolution_592 = torch.ops.aten.convolution.default(relu_521, arg1336_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1336_1 = None
        add_1732 = torch.ops.aten.add.Tensor(arg1338_1, 1e-05);  arg1338_1 = None
        sqrt_592 = torch.ops.aten.sqrt.default(add_1732);  add_1732 = None
        reciprocal_592 = torch.ops.aten.reciprocal.default(sqrt_592);  sqrt_592 = None
        mul_2000 = torch.ops.aten.mul.Tensor(reciprocal_592, 1);  reciprocal_592 = None
        unsqueeze_4792 = torch.ops.aten.unsqueeze.default(arg1337_1, -1);  arg1337_1 = None
        unsqueeze_4793 = torch.ops.aten.unsqueeze.default(unsqueeze_4792, -1);  unsqueeze_4792 = None
        unsqueeze_4794 = torch.ops.aten.unsqueeze.default(mul_2000, -1);  mul_2000 = None
        unsqueeze_4795 = torch.ops.aten.unsqueeze.default(unsqueeze_4794, -1);  unsqueeze_4794 = None
        sub_592 = torch.ops.aten.sub.Tensor(convolution_592, unsqueeze_4793);  convolution_592 = unsqueeze_4793 = None
        mul_2001 = torch.ops.aten.mul.Tensor(sub_592, unsqueeze_4795);  sub_592 = unsqueeze_4795 = None
        unsqueeze_4796 = torch.ops.aten.unsqueeze.default(arg1339_1, -1);  arg1339_1 = None
        unsqueeze_4797 = torch.ops.aten.unsqueeze.default(unsqueeze_4796, -1);  unsqueeze_4796 = None
        mul_2002 = torch.ops.aten.mul.Tensor(mul_2001, unsqueeze_4797);  mul_2001 = unsqueeze_4797 = None
        unsqueeze_4798 = torch.ops.aten.unsqueeze.default(arg1340_1, -1);  arg1340_1 = None
        unsqueeze_4799 = torch.ops.aten.unsqueeze.default(unsqueeze_4798, -1);  unsqueeze_4798 = None
        add_1733 = torch.ops.aten.add.Tensor(mul_2002, unsqueeze_4799);  mul_2002 = unsqueeze_4799 = None
        relu_522 = torch.ops.aten.relu.default(add_1733);  add_1733 = None
        convolution_593 = torch.ops.aten.convolution.default(relu_522, arg1341_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_522 = arg1341_1 = None
        add_1734 = torch.ops.aten.add.Tensor(arg1343_1, 1e-05);  arg1343_1 = None
        sqrt_593 = torch.ops.aten.sqrt.default(add_1734);  add_1734 = None
        reciprocal_593 = torch.ops.aten.reciprocal.default(sqrt_593);  sqrt_593 = None
        mul_2003 = torch.ops.aten.mul.Tensor(reciprocal_593, 1);  reciprocal_593 = None
        unsqueeze_4800 = torch.ops.aten.unsqueeze.default(arg1342_1, -1);  arg1342_1 = None
        unsqueeze_4801 = torch.ops.aten.unsqueeze.default(unsqueeze_4800, -1);  unsqueeze_4800 = None
        unsqueeze_4802 = torch.ops.aten.unsqueeze.default(mul_2003, -1);  mul_2003 = None
        unsqueeze_4803 = torch.ops.aten.unsqueeze.default(unsqueeze_4802, -1);  unsqueeze_4802 = None
        sub_593 = torch.ops.aten.sub.Tensor(convolution_593, unsqueeze_4801);  convolution_593 = unsqueeze_4801 = None
        mul_2004 = torch.ops.aten.mul.Tensor(sub_593, unsqueeze_4803);  sub_593 = unsqueeze_4803 = None
        unsqueeze_4804 = torch.ops.aten.unsqueeze.default(arg1344_1, -1);  arg1344_1 = None
        unsqueeze_4805 = torch.ops.aten.unsqueeze.default(unsqueeze_4804, -1);  unsqueeze_4804 = None
        mul_2005 = torch.ops.aten.mul.Tensor(mul_2004, unsqueeze_4805);  mul_2004 = unsqueeze_4805 = None
        unsqueeze_4806 = torch.ops.aten.unsqueeze.default(arg1345_1, -1);  arg1345_1 = None
        unsqueeze_4807 = torch.ops.aten.unsqueeze.default(unsqueeze_4806, -1);  unsqueeze_4806 = None
        add_1735 = torch.ops.aten.add.Tensor(mul_2005, unsqueeze_4807);  mul_2005 = unsqueeze_4807 = None
        add_1736 = torch.ops.aten.add.Tensor(add_1735, relu_521);  add_1735 = relu_521 = None
        relu_523 = torch.ops.aten.relu.default(add_1736);  add_1736 = None
        convolution_594 = torch.ops.aten.convolution.default(relu_523, arg1346_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1346_1 = None
        add_1737 = torch.ops.aten.add.Tensor(arg1348_1, 1e-05);  arg1348_1 = None
        sqrt_594 = torch.ops.aten.sqrt.default(add_1737);  add_1737 = None
        reciprocal_594 = torch.ops.aten.reciprocal.default(sqrt_594);  sqrt_594 = None
        mul_2006 = torch.ops.aten.mul.Tensor(reciprocal_594, 1);  reciprocal_594 = None
        unsqueeze_4808 = torch.ops.aten.unsqueeze.default(arg1347_1, -1);  arg1347_1 = None
        unsqueeze_4809 = torch.ops.aten.unsqueeze.default(unsqueeze_4808, -1);  unsqueeze_4808 = None
        unsqueeze_4810 = torch.ops.aten.unsqueeze.default(mul_2006, -1);  mul_2006 = None
        unsqueeze_4811 = torch.ops.aten.unsqueeze.default(unsqueeze_4810, -1);  unsqueeze_4810 = None
        sub_594 = torch.ops.aten.sub.Tensor(convolution_594, unsqueeze_4809);  convolution_594 = unsqueeze_4809 = None
        mul_2007 = torch.ops.aten.mul.Tensor(sub_594, unsqueeze_4811);  sub_594 = unsqueeze_4811 = None
        unsqueeze_4812 = torch.ops.aten.unsqueeze.default(arg1349_1, -1);  arg1349_1 = None
        unsqueeze_4813 = torch.ops.aten.unsqueeze.default(unsqueeze_4812, -1);  unsqueeze_4812 = None
        mul_2008 = torch.ops.aten.mul.Tensor(mul_2007, unsqueeze_4813);  mul_2007 = unsqueeze_4813 = None
        unsqueeze_4814 = torch.ops.aten.unsqueeze.default(arg1350_1, -1);  arg1350_1 = None
        unsqueeze_4815 = torch.ops.aten.unsqueeze.default(unsqueeze_4814, -1);  unsqueeze_4814 = None
        add_1738 = torch.ops.aten.add.Tensor(mul_2008, unsqueeze_4815);  mul_2008 = unsqueeze_4815 = None
        relu_524 = torch.ops.aten.relu.default(add_1738);  add_1738 = None
        convolution_595 = torch.ops.aten.convolution.default(relu_524, arg1351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_524 = arg1351_1 = None
        add_1739 = torch.ops.aten.add.Tensor(arg1353_1, 1e-05);  arg1353_1 = None
        sqrt_595 = torch.ops.aten.sqrt.default(add_1739);  add_1739 = None
        reciprocal_595 = torch.ops.aten.reciprocal.default(sqrt_595);  sqrt_595 = None
        mul_2009 = torch.ops.aten.mul.Tensor(reciprocal_595, 1);  reciprocal_595 = None
        unsqueeze_4816 = torch.ops.aten.unsqueeze.default(arg1352_1, -1);  arg1352_1 = None
        unsqueeze_4817 = torch.ops.aten.unsqueeze.default(unsqueeze_4816, -1);  unsqueeze_4816 = None
        unsqueeze_4818 = torch.ops.aten.unsqueeze.default(mul_2009, -1);  mul_2009 = None
        unsqueeze_4819 = torch.ops.aten.unsqueeze.default(unsqueeze_4818, -1);  unsqueeze_4818 = None
        sub_595 = torch.ops.aten.sub.Tensor(convolution_595, unsqueeze_4817);  convolution_595 = unsqueeze_4817 = None
        mul_2010 = torch.ops.aten.mul.Tensor(sub_595, unsqueeze_4819);  sub_595 = unsqueeze_4819 = None
        unsqueeze_4820 = torch.ops.aten.unsqueeze.default(arg1354_1, -1);  arg1354_1 = None
        unsqueeze_4821 = torch.ops.aten.unsqueeze.default(unsqueeze_4820, -1);  unsqueeze_4820 = None
        mul_2011 = torch.ops.aten.mul.Tensor(mul_2010, unsqueeze_4821);  mul_2010 = unsqueeze_4821 = None
        unsqueeze_4822 = torch.ops.aten.unsqueeze.default(arg1355_1, -1);  arg1355_1 = None
        unsqueeze_4823 = torch.ops.aten.unsqueeze.default(unsqueeze_4822, -1);  unsqueeze_4822 = None
        add_1740 = torch.ops.aten.add.Tensor(mul_2011, unsqueeze_4823);  mul_2011 = unsqueeze_4823 = None
        add_1741 = torch.ops.aten.add.Tensor(add_1740, relu_523);  add_1740 = relu_523 = None
        relu_525 = torch.ops.aten.relu.default(add_1741);  add_1741 = None
        convolution_596 = torch.ops.aten.convolution.default(relu_525, arg1356_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1356_1 = None
        add_1742 = torch.ops.aten.add.Tensor(arg1358_1, 1e-05);  arg1358_1 = None
        sqrt_596 = torch.ops.aten.sqrt.default(add_1742);  add_1742 = None
        reciprocal_596 = torch.ops.aten.reciprocal.default(sqrt_596);  sqrt_596 = None
        mul_2012 = torch.ops.aten.mul.Tensor(reciprocal_596, 1);  reciprocal_596 = None
        unsqueeze_4824 = torch.ops.aten.unsqueeze.default(arg1357_1, -1);  arg1357_1 = None
        unsqueeze_4825 = torch.ops.aten.unsqueeze.default(unsqueeze_4824, -1);  unsqueeze_4824 = None
        unsqueeze_4826 = torch.ops.aten.unsqueeze.default(mul_2012, -1);  mul_2012 = None
        unsqueeze_4827 = torch.ops.aten.unsqueeze.default(unsqueeze_4826, -1);  unsqueeze_4826 = None
        sub_596 = torch.ops.aten.sub.Tensor(convolution_596, unsqueeze_4825);  convolution_596 = unsqueeze_4825 = None
        mul_2013 = torch.ops.aten.mul.Tensor(sub_596, unsqueeze_4827);  sub_596 = unsqueeze_4827 = None
        unsqueeze_4828 = torch.ops.aten.unsqueeze.default(arg1359_1, -1);  arg1359_1 = None
        unsqueeze_4829 = torch.ops.aten.unsqueeze.default(unsqueeze_4828, -1);  unsqueeze_4828 = None
        mul_2014 = torch.ops.aten.mul.Tensor(mul_2013, unsqueeze_4829);  mul_2013 = unsqueeze_4829 = None
        unsqueeze_4830 = torch.ops.aten.unsqueeze.default(arg1360_1, -1);  arg1360_1 = None
        unsqueeze_4831 = torch.ops.aten.unsqueeze.default(unsqueeze_4830, -1);  unsqueeze_4830 = None
        add_1743 = torch.ops.aten.add.Tensor(mul_2014, unsqueeze_4831);  mul_2014 = unsqueeze_4831 = None
        relu_526 = torch.ops.aten.relu.default(add_1743);  add_1743 = None
        convolution_597 = torch.ops.aten.convolution.default(relu_526, arg1361_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_526 = arg1361_1 = None
        add_1744 = torch.ops.aten.add.Tensor(arg1363_1, 1e-05);  arg1363_1 = None
        sqrt_597 = torch.ops.aten.sqrt.default(add_1744);  add_1744 = None
        reciprocal_597 = torch.ops.aten.reciprocal.default(sqrt_597);  sqrt_597 = None
        mul_2015 = torch.ops.aten.mul.Tensor(reciprocal_597, 1);  reciprocal_597 = None
        unsqueeze_4832 = torch.ops.aten.unsqueeze.default(arg1362_1, -1);  arg1362_1 = None
        unsqueeze_4833 = torch.ops.aten.unsqueeze.default(unsqueeze_4832, -1);  unsqueeze_4832 = None
        unsqueeze_4834 = torch.ops.aten.unsqueeze.default(mul_2015, -1);  mul_2015 = None
        unsqueeze_4835 = torch.ops.aten.unsqueeze.default(unsqueeze_4834, -1);  unsqueeze_4834 = None
        sub_597 = torch.ops.aten.sub.Tensor(convolution_597, unsqueeze_4833);  convolution_597 = unsqueeze_4833 = None
        mul_2016 = torch.ops.aten.mul.Tensor(sub_597, unsqueeze_4835);  sub_597 = unsqueeze_4835 = None
        unsqueeze_4836 = torch.ops.aten.unsqueeze.default(arg1364_1, -1);  arg1364_1 = None
        unsqueeze_4837 = torch.ops.aten.unsqueeze.default(unsqueeze_4836, -1);  unsqueeze_4836 = None
        mul_2017 = torch.ops.aten.mul.Tensor(mul_2016, unsqueeze_4837);  mul_2016 = unsqueeze_4837 = None
        unsqueeze_4838 = torch.ops.aten.unsqueeze.default(arg1365_1, -1);  arg1365_1 = None
        unsqueeze_4839 = torch.ops.aten.unsqueeze.default(unsqueeze_4838, -1);  unsqueeze_4838 = None
        add_1745 = torch.ops.aten.add.Tensor(mul_2017, unsqueeze_4839);  mul_2017 = unsqueeze_4839 = None
        add_1746 = torch.ops.aten.add.Tensor(add_1745, relu_525);  add_1745 = relu_525 = None
        relu_527 = torch.ops.aten.relu.default(add_1746);  add_1746 = None
        convolution_598 = torch.ops.aten.convolution.default(relu_507, arg1366_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1366_1 = None
        add_1747 = torch.ops.aten.add.Tensor(arg1368_1, 1e-05);  arg1368_1 = None
        sqrt_598 = torch.ops.aten.sqrt.default(add_1747);  add_1747 = None
        reciprocal_598 = torch.ops.aten.reciprocal.default(sqrt_598);  sqrt_598 = None
        mul_2018 = torch.ops.aten.mul.Tensor(reciprocal_598, 1);  reciprocal_598 = None
        unsqueeze_4840 = torch.ops.aten.unsqueeze.default(arg1367_1, -1);  arg1367_1 = None
        unsqueeze_4841 = torch.ops.aten.unsqueeze.default(unsqueeze_4840, -1);  unsqueeze_4840 = None
        unsqueeze_4842 = torch.ops.aten.unsqueeze.default(mul_2018, -1);  mul_2018 = None
        unsqueeze_4843 = torch.ops.aten.unsqueeze.default(unsqueeze_4842, -1);  unsqueeze_4842 = None
        sub_598 = torch.ops.aten.sub.Tensor(convolution_598, unsqueeze_4841);  convolution_598 = unsqueeze_4841 = None
        mul_2019 = torch.ops.aten.mul.Tensor(sub_598, unsqueeze_4843);  sub_598 = unsqueeze_4843 = None
        unsqueeze_4844 = torch.ops.aten.unsqueeze.default(arg1369_1, -1);  arg1369_1 = None
        unsqueeze_4845 = torch.ops.aten.unsqueeze.default(unsqueeze_4844, -1);  unsqueeze_4844 = None
        mul_2020 = torch.ops.aten.mul.Tensor(mul_2019, unsqueeze_4845);  mul_2019 = unsqueeze_4845 = None
        unsqueeze_4846 = torch.ops.aten.unsqueeze.default(arg1370_1, -1);  arg1370_1 = None
        unsqueeze_4847 = torch.ops.aten.unsqueeze.default(unsqueeze_4846, -1);  unsqueeze_4846 = None
        add_1748 = torch.ops.aten.add.Tensor(mul_2020, unsqueeze_4847);  mul_2020 = unsqueeze_4847 = None
        relu_528 = torch.ops.aten.relu.default(add_1748);  add_1748 = None
        convolution_599 = torch.ops.aten.convolution.default(relu_528, arg1371_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_528 = arg1371_1 = None
        add_1749 = torch.ops.aten.add.Tensor(arg1373_1, 1e-05);  arg1373_1 = None
        sqrt_599 = torch.ops.aten.sqrt.default(add_1749);  add_1749 = None
        reciprocal_599 = torch.ops.aten.reciprocal.default(sqrt_599);  sqrt_599 = None
        mul_2021 = torch.ops.aten.mul.Tensor(reciprocal_599, 1);  reciprocal_599 = None
        unsqueeze_4848 = torch.ops.aten.unsqueeze.default(arg1372_1, -1);  arg1372_1 = None
        unsqueeze_4849 = torch.ops.aten.unsqueeze.default(unsqueeze_4848, -1);  unsqueeze_4848 = None
        unsqueeze_4850 = torch.ops.aten.unsqueeze.default(mul_2021, -1);  mul_2021 = None
        unsqueeze_4851 = torch.ops.aten.unsqueeze.default(unsqueeze_4850, -1);  unsqueeze_4850 = None
        sub_599 = torch.ops.aten.sub.Tensor(convolution_599, unsqueeze_4849);  convolution_599 = unsqueeze_4849 = None
        mul_2022 = torch.ops.aten.mul.Tensor(sub_599, unsqueeze_4851);  sub_599 = unsqueeze_4851 = None
        unsqueeze_4852 = torch.ops.aten.unsqueeze.default(arg1374_1, -1);  arg1374_1 = None
        unsqueeze_4853 = torch.ops.aten.unsqueeze.default(unsqueeze_4852, -1);  unsqueeze_4852 = None
        mul_2023 = torch.ops.aten.mul.Tensor(mul_2022, unsqueeze_4853);  mul_2022 = unsqueeze_4853 = None
        unsqueeze_4854 = torch.ops.aten.unsqueeze.default(arg1375_1, -1);  arg1375_1 = None
        unsqueeze_4855 = torch.ops.aten.unsqueeze.default(unsqueeze_4854, -1);  unsqueeze_4854 = None
        add_1750 = torch.ops.aten.add.Tensor(mul_2023, unsqueeze_4855);  mul_2023 = unsqueeze_4855 = None
        add_1751 = torch.ops.aten.add.Tensor(add_1750, relu_507);  add_1750 = relu_507 = None
        relu_529 = torch.ops.aten.relu.default(add_1751);  add_1751 = None
        convolution_600 = torch.ops.aten.convolution.default(relu_529, arg1376_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1376_1 = None
        add_1752 = torch.ops.aten.add.Tensor(arg1378_1, 1e-05);  arg1378_1 = None
        sqrt_600 = torch.ops.aten.sqrt.default(add_1752);  add_1752 = None
        reciprocal_600 = torch.ops.aten.reciprocal.default(sqrt_600);  sqrt_600 = None
        mul_2024 = torch.ops.aten.mul.Tensor(reciprocal_600, 1);  reciprocal_600 = None
        unsqueeze_4856 = torch.ops.aten.unsqueeze.default(arg1377_1, -1);  arg1377_1 = None
        unsqueeze_4857 = torch.ops.aten.unsqueeze.default(unsqueeze_4856, -1);  unsqueeze_4856 = None
        unsqueeze_4858 = torch.ops.aten.unsqueeze.default(mul_2024, -1);  mul_2024 = None
        unsqueeze_4859 = torch.ops.aten.unsqueeze.default(unsqueeze_4858, -1);  unsqueeze_4858 = None
        sub_600 = torch.ops.aten.sub.Tensor(convolution_600, unsqueeze_4857);  convolution_600 = unsqueeze_4857 = None
        mul_2025 = torch.ops.aten.mul.Tensor(sub_600, unsqueeze_4859);  sub_600 = unsqueeze_4859 = None
        unsqueeze_4860 = torch.ops.aten.unsqueeze.default(arg1379_1, -1);  arg1379_1 = None
        unsqueeze_4861 = torch.ops.aten.unsqueeze.default(unsqueeze_4860, -1);  unsqueeze_4860 = None
        mul_2026 = torch.ops.aten.mul.Tensor(mul_2025, unsqueeze_4861);  mul_2025 = unsqueeze_4861 = None
        unsqueeze_4862 = torch.ops.aten.unsqueeze.default(arg1380_1, -1);  arg1380_1 = None
        unsqueeze_4863 = torch.ops.aten.unsqueeze.default(unsqueeze_4862, -1);  unsqueeze_4862 = None
        add_1753 = torch.ops.aten.add.Tensor(mul_2026, unsqueeze_4863);  mul_2026 = unsqueeze_4863 = None
        relu_530 = torch.ops.aten.relu.default(add_1753);  add_1753 = None
        convolution_601 = torch.ops.aten.convolution.default(relu_530, arg1381_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_530 = arg1381_1 = None
        add_1754 = torch.ops.aten.add.Tensor(arg1383_1, 1e-05);  arg1383_1 = None
        sqrt_601 = torch.ops.aten.sqrt.default(add_1754);  add_1754 = None
        reciprocal_601 = torch.ops.aten.reciprocal.default(sqrt_601);  sqrt_601 = None
        mul_2027 = torch.ops.aten.mul.Tensor(reciprocal_601, 1);  reciprocal_601 = None
        unsqueeze_4864 = torch.ops.aten.unsqueeze.default(arg1382_1, -1);  arg1382_1 = None
        unsqueeze_4865 = torch.ops.aten.unsqueeze.default(unsqueeze_4864, -1);  unsqueeze_4864 = None
        unsqueeze_4866 = torch.ops.aten.unsqueeze.default(mul_2027, -1);  mul_2027 = None
        unsqueeze_4867 = torch.ops.aten.unsqueeze.default(unsqueeze_4866, -1);  unsqueeze_4866 = None
        sub_601 = torch.ops.aten.sub.Tensor(convolution_601, unsqueeze_4865);  convolution_601 = unsqueeze_4865 = None
        mul_2028 = torch.ops.aten.mul.Tensor(sub_601, unsqueeze_4867);  sub_601 = unsqueeze_4867 = None
        unsqueeze_4868 = torch.ops.aten.unsqueeze.default(arg1384_1, -1);  arg1384_1 = None
        unsqueeze_4869 = torch.ops.aten.unsqueeze.default(unsqueeze_4868, -1);  unsqueeze_4868 = None
        mul_2029 = torch.ops.aten.mul.Tensor(mul_2028, unsqueeze_4869);  mul_2028 = unsqueeze_4869 = None
        unsqueeze_4870 = torch.ops.aten.unsqueeze.default(arg1385_1, -1);  arg1385_1 = None
        unsqueeze_4871 = torch.ops.aten.unsqueeze.default(unsqueeze_4870, -1);  unsqueeze_4870 = None
        add_1755 = torch.ops.aten.add.Tensor(mul_2029, unsqueeze_4871);  mul_2029 = unsqueeze_4871 = None
        add_1756 = torch.ops.aten.add.Tensor(add_1755, relu_529);  add_1755 = relu_529 = None
        relu_531 = torch.ops.aten.relu.default(add_1756);  add_1756 = None
        convolution_602 = torch.ops.aten.convolution.default(relu_531, arg1386_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1386_1 = None
        add_1757 = torch.ops.aten.add.Tensor(arg1388_1, 1e-05);  arg1388_1 = None
        sqrt_602 = torch.ops.aten.sqrt.default(add_1757);  add_1757 = None
        reciprocal_602 = torch.ops.aten.reciprocal.default(sqrt_602);  sqrt_602 = None
        mul_2030 = torch.ops.aten.mul.Tensor(reciprocal_602, 1);  reciprocal_602 = None
        unsqueeze_4872 = torch.ops.aten.unsqueeze.default(arg1387_1, -1);  arg1387_1 = None
        unsqueeze_4873 = torch.ops.aten.unsqueeze.default(unsqueeze_4872, -1);  unsqueeze_4872 = None
        unsqueeze_4874 = torch.ops.aten.unsqueeze.default(mul_2030, -1);  mul_2030 = None
        unsqueeze_4875 = torch.ops.aten.unsqueeze.default(unsqueeze_4874, -1);  unsqueeze_4874 = None
        sub_602 = torch.ops.aten.sub.Tensor(convolution_602, unsqueeze_4873);  convolution_602 = unsqueeze_4873 = None
        mul_2031 = torch.ops.aten.mul.Tensor(sub_602, unsqueeze_4875);  sub_602 = unsqueeze_4875 = None
        unsqueeze_4876 = torch.ops.aten.unsqueeze.default(arg1389_1, -1);  arg1389_1 = None
        unsqueeze_4877 = torch.ops.aten.unsqueeze.default(unsqueeze_4876, -1);  unsqueeze_4876 = None
        mul_2032 = torch.ops.aten.mul.Tensor(mul_2031, unsqueeze_4877);  mul_2031 = unsqueeze_4877 = None
        unsqueeze_4878 = torch.ops.aten.unsqueeze.default(arg1390_1, -1);  arg1390_1 = None
        unsqueeze_4879 = torch.ops.aten.unsqueeze.default(unsqueeze_4878, -1);  unsqueeze_4878 = None
        add_1758 = torch.ops.aten.add.Tensor(mul_2032, unsqueeze_4879);  mul_2032 = unsqueeze_4879 = None
        relu_532 = torch.ops.aten.relu.default(add_1758);  add_1758 = None
        convolution_603 = torch.ops.aten.convolution.default(relu_532, arg1391_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_532 = arg1391_1 = None
        add_1759 = torch.ops.aten.add.Tensor(arg1393_1, 1e-05);  arg1393_1 = None
        sqrt_603 = torch.ops.aten.sqrt.default(add_1759);  add_1759 = None
        reciprocal_603 = torch.ops.aten.reciprocal.default(sqrt_603);  sqrt_603 = None
        mul_2033 = torch.ops.aten.mul.Tensor(reciprocal_603, 1);  reciprocal_603 = None
        unsqueeze_4880 = torch.ops.aten.unsqueeze.default(arg1392_1, -1);  arg1392_1 = None
        unsqueeze_4881 = torch.ops.aten.unsqueeze.default(unsqueeze_4880, -1);  unsqueeze_4880 = None
        unsqueeze_4882 = torch.ops.aten.unsqueeze.default(mul_2033, -1);  mul_2033 = None
        unsqueeze_4883 = torch.ops.aten.unsqueeze.default(unsqueeze_4882, -1);  unsqueeze_4882 = None
        sub_603 = torch.ops.aten.sub.Tensor(convolution_603, unsqueeze_4881);  convolution_603 = unsqueeze_4881 = None
        mul_2034 = torch.ops.aten.mul.Tensor(sub_603, unsqueeze_4883);  sub_603 = unsqueeze_4883 = None
        unsqueeze_4884 = torch.ops.aten.unsqueeze.default(arg1394_1, -1);  arg1394_1 = None
        unsqueeze_4885 = torch.ops.aten.unsqueeze.default(unsqueeze_4884, -1);  unsqueeze_4884 = None
        mul_2035 = torch.ops.aten.mul.Tensor(mul_2034, unsqueeze_4885);  mul_2034 = unsqueeze_4885 = None
        unsqueeze_4886 = torch.ops.aten.unsqueeze.default(arg1395_1, -1);  arg1395_1 = None
        unsqueeze_4887 = torch.ops.aten.unsqueeze.default(unsqueeze_4886, -1);  unsqueeze_4886 = None
        add_1760 = torch.ops.aten.add.Tensor(mul_2035, unsqueeze_4887);  mul_2035 = unsqueeze_4887 = None
        add_1761 = torch.ops.aten.add.Tensor(add_1760, relu_531);  add_1760 = relu_531 = None
        relu_533 = torch.ops.aten.relu.default(add_1761);  add_1761 = None
        convolution_604 = torch.ops.aten.convolution.default(relu_533, arg1396_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1396_1 = None
        add_1762 = torch.ops.aten.add.Tensor(arg1398_1, 1e-05);  arg1398_1 = None
        sqrt_604 = torch.ops.aten.sqrt.default(add_1762);  add_1762 = None
        reciprocal_604 = torch.ops.aten.reciprocal.default(sqrt_604);  sqrt_604 = None
        mul_2036 = torch.ops.aten.mul.Tensor(reciprocal_604, 1);  reciprocal_604 = None
        unsqueeze_4888 = torch.ops.aten.unsqueeze.default(arg1397_1, -1);  arg1397_1 = None
        unsqueeze_4889 = torch.ops.aten.unsqueeze.default(unsqueeze_4888, -1);  unsqueeze_4888 = None
        unsqueeze_4890 = torch.ops.aten.unsqueeze.default(mul_2036, -1);  mul_2036 = None
        unsqueeze_4891 = torch.ops.aten.unsqueeze.default(unsqueeze_4890, -1);  unsqueeze_4890 = None
        sub_604 = torch.ops.aten.sub.Tensor(convolution_604, unsqueeze_4889);  convolution_604 = unsqueeze_4889 = None
        mul_2037 = torch.ops.aten.mul.Tensor(sub_604, unsqueeze_4891);  sub_604 = unsqueeze_4891 = None
        unsqueeze_4892 = torch.ops.aten.unsqueeze.default(arg1399_1, -1);  arg1399_1 = None
        unsqueeze_4893 = torch.ops.aten.unsqueeze.default(unsqueeze_4892, -1);  unsqueeze_4892 = None
        mul_2038 = torch.ops.aten.mul.Tensor(mul_2037, unsqueeze_4893);  mul_2037 = unsqueeze_4893 = None
        unsqueeze_4894 = torch.ops.aten.unsqueeze.default(arg1400_1, -1);  arg1400_1 = None
        unsqueeze_4895 = torch.ops.aten.unsqueeze.default(unsqueeze_4894, -1);  unsqueeze_4894 = None
        add_1763 = torch.ops.aten.add.Tensor(mul_2038, unsqueeze_4895);  mul_2038 = unsqueeze_4895 = None
        relu_534 = torch.ops.aten.relu.default(add_1763);  add_1763 = None
        convolution_605 = torch.ops.aten.convolution.default(relu_534, arg1401_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_534 = arg1401_1 = None
        add_1764 = torch.ops.aten.add.Tensor(arg1403_1, 1e-05);  arg1403_1 = None
        sqrt_605 = torch.ops.aten.sqrt.default(add_1764);  add_1764 = None
        reciprocal_605 = torch.ops.aten.reciprocal.default(sqrt_605);  sqrt_605 = None
        mul_2039 = torch.ops.aten.mul.Tensor(reciprocal_605, 1);  reciprocal_605 = None
        unsqueeze_4896 = torch.ops.aten.unsqueeze.default(arg1402_1, -1);  arg1402_1 = None
        unsqueeze_4897 = torch.ops.aten.unsqueeze.default(unsqueeze_4896, -1);  unsqueeze_4896 = None
        unsqueeze_4898 = torch.ops.aten.unsqueeze.default(mul_2039, -1);  mul_2039 = None
        unsqueeze_4899 = torch.ops.aten.unsqueeze.default(unsqueeze_4898, -1);  unsqueeze_4898 = None
        sub_605 = torch.ops.aten.sub.Tensor(convolution_605, unsqueeze_4897);  convolution_605 = unsqueeze_4897 = None
        mul_2040 = torch.ops.aten.mul.Tensor(sub_605, unsqueeze_4899);  sub_605 = unsqueeze_4899 = None
        unsqueeze_4900 = torch.ops.aten.unsqueeze.default(arg1404_1, -1);  arg1404_1 = None
        unsqueeze_4901 = torch.ops.aten.unsqueeze.default(unsqueeze_4900, -1);  unsqueeze_4900 = None
        mul_2041 = torch.ops.aten.mul.Tensor(mul_2040, unsqueeze_4901);  mul_2040 = unsqueeze_4901 = None
        unsqueeze_4902 = torch.ops.aten.unsqueeze.default(arg1405_1, -1);  arg1405_1 = None
        unsqueeze_4903 = torch.ops.aten.unsqueeze.default(unsqueeze_4902, -1);  unsqueeze_4902 = None
        add_1765 = torch.ops.aten.add.Tensor(mul_2041, unsqueeze_4903);  mul_2041 = unsqueeze_4903 = None
        add_1766 = torch.ops.aten.add.Tensor(add_1765, relu_533);  add_1765 = relu_533 = None
        relu_535 = torch.ops.aten.relu.default(add_1766);  add_1766 = None
        convolution_606 = torch.ops.aten.convolution.default(relu_511, arg1406_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1406_1 = None
        add_1767 = torch.ops.aten.add.Tensor(arg1408_1, 1e-05);  arg1408_1 = None
        sqrt_606 = torch.ops.aten.sqrt.default(add_1767);  add_1767 = None
        reciprocal_606 = torch.ops.aten.reciprocal.default(sqrt_606);  sqrt_606 = None
        mul_2042 = torch.ops.aten.mul.Tensor(reciprocal_606, 1);  reciprocal_606 = None
        unsqueeze_4904 = torch.ops.aten.unsqueeze.default(arg1407_1, -1);  arg1407_1 = None
        unsqueeze_4905 = torch.ops.aten.unsqueeze.default(unsqueeze_4904, -1);  unsqueeze_4904 = None
        unsqueeze_4906 = torch.ops.aten.unsqueeze.default(mul_2042, -1);  mul_2042 = None
        unsqueeze_4907 = torch.ops.aten.unsqueeze.default(unsqueeze_4906, -1);  unsqueeze_4906 = None
        sub_606 = torch.ops.aten.sub.Tensor(convolution_606, unsqueeze_4905);  convolution_606 = unsqueeze_4905 = None
        mul_2043 = torch.ops.aten.mul.Tensor(sub_606, unsqueeze_4907);  sub_606 = unsqueeze_4907 = None
        unsqueeze_4908 = torch.ops.aten.unsqueeze.default(arg1409_1, -1);  arg1409_1 = None
        unsqueeze_4909 = torch.ops.aten.unsqueeze.default(unsqueeze_4908, -1);  unsqueeze_4908 = None
        mul_2044 = torch.ops.aten.mul.Tensor(mul_2043, unsqueeze_4909);  mul_2043 = unsqueeze_4909 = None
        unsqueeze_4910 = torch.ops.aten.unsqueeze.default(arg1410_1, -1);  arg1410_1 = None
        unsqueeze_4911 = torch.ops.aten.unsqueeze.default(unsqueeze_4910, -1);  unsqueeze_4910 = None
        add_1768 = torch.ops.aten.add.Tensor(mul_2044, unsqueeze_4911);  mul_2044 = unsqueeze_4911 = None
        relu_536 = torch.ops.aten.relu.default(add_1768);  add_1768 = None
        convolution_607 = torch.ops.aten.convolution.default(relu_536, arg1411_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_536 = arg1411_1 = None
        add_1769 = torch.ops.aten.add.Tensor(arg1413_1, 1e-05);  arg1413_1 = None
        sqrt_607 = torch.ops.aten.sqrt.default(add_1769);  add_1769 = None
        reciprocal_607 = torch.ops.aten.reciprocal.default(sqrt_607);  sqrt_607 = None
        mul_2045 = torch.ops.aten.mul.Tensor(reciprocal_607, 1);  reciprocal_607 = None
        unsqueeze_4912 = torch.ops.aten.unsqueeze.default(arg1412_1, -1);  arg1412_1 = None
        unsqueeze_4913 = torch.ops.aten.unsqueeze.default(unsqueeze_4912, -1);  unsqueeze_4912 = None
        unsqueeze_4914 = torch.ops.aten.unsqueeze.default(mul_2045, -1);  mul_2045 = None
        unsqueeze_4915 = torch.ops.aten.unsqueeze.default(unsqueeze_4914, -1);  unsqueeze_4914 = None
        sub_607 = torch.ops.aten.sub.Tensor(convolution_607, unsqueeze_4913);  convolution_607 = unsqueeze_4913 = None
        mul_2046 = torch.ops.aten.mul.Tensor(sub_607, unsqueeze_4915);  sub_607 = unsqueeze_4915 = None
        unsqueeze_4916 = torch.ops.aten.unsqueeze.default(arg1414_1, -1);  arg1414_1 = None
        unsqueeze_4917 = torch.ops.aten.unsqueeze.default(unsqueeze_4916, -1);  unsqueeze_4916 = None
        mul_2047 = torch.ops.aten.mul.Tensor(mul_2046, unsqueeze_4917);  mul_2046 = unsqueeze_4917 = None
        unsqueeze_4918 = torch.ops.aten.unsqueeze.default(arg1415_1, -1);  arg1415_1 = None
        unsqueeze_4919 = torch.ops.aten.unsqueeze.default(unsqueeze_4918, -1);  unsqueeze_4918 = None
        add_1770 = torch.ops.aten.add.Tensor(mul_2047, unsqueeze_4919);  mul_2047 = unsqueeze_4919 = None
        add_1771 = torch.ops.aten.add.Tensor(add_1770, relu_511);  add_1770 = relu_511 = None
        relu_537 = torch.ops.aten.relu.default(add_1771);  add_1771 = None
        convolution_608 = torch.ops.aten.convolution.default(relu_537, arg1416_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1416_1 = None
        add_1772 = torch.ops.aten.add.Tensor(arg1418_1, 1e-05);  arg1418_1 = None
        sqrt_608 = torch.ops.aten.sqrt.default(add_1772);  add_1772 = None
        reciprocal_608 = torch.ops.aten.reciprocal.default(sqrt_608);  sqrt_608 = None
        mul_2048 = torch.ops.aten.mul.Tensor(reciprocal_608, 1);  reciprocal_608 = None
        unsqueeze_4920 = torch.ops.aten.unsqueeze.default(arg1417_1, -1);  arg1417_1 = None
        unsqueeze_4921 = torch.ops.aten.unsqueeze.default(unsqueeze_4920, -1);  unsqueeze_4920 = None
        unsqueeze_4922 = torch.ops.aten.unsqueeze.default(mul_2048, -1);  mul_2048 = None
        unsqueeze_4923 = torch.ops.aten.unsqueeze.default(unsqueeze_4922, -1);  unsqueeze_4922 = None
        sub_608 = torch.ops.aten.sub.Tensor(convolution_608, unsqueeze_4921);  convolution_608 = unsqueeze_4921 = None
        mul_2049 = torch.ops.aten.mul.Tensor(sub_608, unsqueeze_4923);  sub_608 = unsqueeze_4923 = None
        unsqueeze_4924 = torch.ops.aten.unsqueeze.default(arg1419_1, -1);  arg1419_1 = None
        unsqueeze_4925 = torch.ops.aten.unsqueeze.default(unsqueeze_4924, -1);  unsqueeze_4924 = None
        mul_2050 = torch.ops.aten.mul.Tensor(mul_2049, unsqueeze_4925);  mul_2049 = unsqueeze_4925 = None
        unsqueeze_4926 = torch.ops.aten.unsqueeze.default(arg1420_1, -1);  arg1420_1 = None
        unsqueeze_4927 = torch.ops.aten.unsqueeze.default(unsqueeze_4926, -1);  unsqueeze_4926 = None
        add_1773 = torch.ops.aten.add.Tensor(mul_2050, unsqueeze_4927);  mul_2050 = unsqueeze_4927 = None
        relu_538 = torch.ops.aten.relu.default(add_1773);  add_1773 = None
        convolution_609 = torch.ops.aten.convolution.default(relu_538, arg1421_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_538 = arg1421_1 = None
        add_1774 = torch.ops.aten.add.Tensor(arg1423_1, 1e-05);  arg1423_1 = None
        sqrt_609 = torch.ops.aten.sqrt.default(add_1774);  add_1774 = None
        reciprocal_609 = torch.ops.aten.reciprocal.default(sqrt_609);  sqrt_609 = None
        mul_2051 = torch.ops.aten.mul.Tensor(reciprocal_609, 1);  reciprocal_609 = None
        unsqueeze_4928 = torch.ops.aten.unsqueeze.default(arg1422_1, -1);  arg1422_1 = None
        unsqueeze_4929 = torch.ops.aten.unsqueeze.default(unsqueeze_4928, -1);  unsqueeze_4928 = None
        unsqueeze_4930 = torch.ops.aten.unsqueeze.default(mul_2051, -1);  mul_2051 = None
        unsqueeze_4931 = torch.ops.aten.unsqueeze.default(unsqueeze_4930, -1);  unsqueeze_4930 = None
        sub_609 = torch.ops.aten.sub.Tensor(convolution_609, unsqueeze_4929);  convolution_609 = unsqueeze_4929 = None
        mul_2052 = torch.ops.aten.mul.Tensor(sub_609, unsqueeze_4931);  sub_609 = unsqueeze_4931 = None
        unsqueeze_4932 = torch.ops.aten.unsqueeze.default(arg1424_1, -1);  arg1424_1 = None
        unsqueeze_4933 = torch.ops.aten.unsqueeze.default(unsqueeze_4932, -1);  unsqueeze_4932 = None
        mul_2053 = torch.ops.aten.mul.Tensor(mul_2052, unsqueeze_4933);  mul_2052 = unsqueeze_4933 = None
        unsqueeze_4934 = torch.ops.aten.unsqueeze.default(arg1425_1, -1);  arg1425_1 = None
        unsqueeze_4935 = torch.ops.aten.unsqueeze.default(unsqueeze_4934, -1);  unsqueeze_4934 = None
        add_1775 = torch.ops.aten.add.Tensor(mul_2053, unsqueeze_4935);  mul_2053 = unsqueeze_4935 = None
        add_1776 = torch.ops.aten.add.Tensor(add_1775, relu_537);  add_1775 = relu_537 = None
        relu_539 = torch.ops.aten.relu.default(add_1776);  add_1776 = None
        convolution_610 = torch.ops.aten.convolution.default(relu_539, arg1426_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1426_1 = None
        add_1777 = torch.ops.aten.add.Tensor(arg1428_1, 1e-05);  arg1428_1 = None
        sqrt_610 = torch.ops.aten.sqrt.default(add_1777);  add_1777 = None
        reciprocal_610 = torch.ops.aten.reciprocal.default(sqrt_610);  sqrt_610 = None
        mul_2054 = torch.ops.aten.mul.Tensor(reciprocal_610, 1);  reciprocal_610 = None
        unsqueeze_4936 = torch.ops.aten.unsqueeze.default(arg1427_1, -1);  arg1427_1 = None
        unsqueeze_4937 = torch.ops.aten.unsqueeze.default(unsqueeze_4936, -1);  unsqueeze_4936 = None
        unsqueeze_4938 = torch.ops.aten.unsqueeze.default(mul_2054, -1);  mul_2054 = None
        unsqueeze_4939 = torch.ops.aten.unsqueeze.default(unsqueeze_4938, -1);  unsqueeze_4938 = None
        sub_610 = torch.ops.aten.sub.Tensor(convolution_610, unsqueeze_4937);  convolution_610 = unsqueeze_4937 = None
        mul_2055 = torch.ops.aten.mul.Tensor(sub_610, unsqueeze_4939);  sub_610 = unsqueeze_4939 = None
        unsqueeze_4940 = torch.ops.aten.unsqueeze.default(arg1429_1, -1);  arg1429_1 = None
        unsqueeze_4941 = torch.ops.aten.unsqueeze.default(unsqueeze_4940, -1);  unsqueeze_4940 = None
        mul_2056 = torch.ops.aten.mul.Tensor(mul_2055, unsqueeze_4941);  mul_2055 = unsqueeze_4941 = None
        unsqueeze_4942 = torch.ops.aten.unsqueeze.default(arg1430_1, -1);  arg1430_1 = None
        unsqueeze_4943 = torch.ops.aten.unsqueeze.default(unsqueeze_4942, -1);  unsqueeze_4942 = None
        add_1778 = torch.ops.aten.add.Tensor(mul_2056, unsqueeze_4943);  mul_2056 = unsqueeze_4943 = None
        relu_540 = torch.ops.aten.relu.default(add_1778);  add_1778 = None
        convolution_611 = torch.ops.aten.convolution.default(relu_540, arg1431_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_540 = arg1431_1 = None
        add_1779 = torch.ops.aten.add.Tensor(arg1433_1, 1e-05);  arg1433_1 = None
        sqrt_611 = torch.ops.aten.sqrt.default(add_1779);  add_1779 = None
        reciprocal_611 = torch.ops.aten.reciprocal.default(sqrt_611);  sqrt_611 = None
        mul_2057 = torch.ops.aten.mul.Tensor(reciprocal_611, 1);  reciprocal_611 = None
        unsqueeze_4944 = torch.ops.aten.unsqueeze.default(arg1432_1, -1);  arg1432_1 = None
        unsqueeze_4945 = torch.ops.aten.unsqueeze.default(unsqueeze_4944, -1);  unsqueeze_4944 = None
        unsqueeze_4946 = torch.ops.aten.unsqueeze.default(mul_2057, -1);  mul_2057 = None
        unsqueeze_4947 = torch.ops.aten.unsqueeze.default(unsqueeze_4946, -1);  unsqueeze_4946 = None
        sub_611 = torch.ops.aten.sub.Tensor(convolution_611, unsqueeze_4945);  convolution_611 = unsqueeze_4945 = None
        mul_2058 = torch.ops.aten.mul.Tensor(sub_611, unsqueeze_4947);  sub_611 = unsqueeze_4947 = None
        unsqueeze_4948 = torch.ops.aten.unsqueeze.default(arg1434_1, -1);  arg1434_1 = None
        unsqueeze_4949 = torch.ops.aten.unsqueeze.default(unsqueeze_4948, -1);  unsqueeze_4948 = None
        mul_2059 = torch.ops.aten.mul.Tensor(mul_2058, unsqueeze_4949);  mul_2058 = unsqueeze_4949 = None
        unsqueeze_4950 = torch.ops.aten.unsqueeze.default(arg1435_1, -1);  arg1435_1 = None
        unsqueeze_4951 = torch.ops.aten.unsqueeze.default(unsqueeze_4950, -1);  unsqueeze_4950 = None
        add_1780 = torch.ops.aten.add.Tensor(mul_2059, unsqueeze_4951);  mul_2059 = unsqueeze_4951 = None
        add_1781 = torch.ops.aten.add.Tensor(add_1780, relu_539);  add_1780 = relu_539 = None
        relu_541 = torch.ops.aten.relu.default(add_1781);  add_1781 = None
        convolution_612 = torch.ops.aten.convolution.default(relu_541, arg1436_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1436_1 = None
        add_1782 = torch.ops.aten.add.Tensor(arg1438_1, 1e-05);  arg1438_1 = None
        sqrt_612 = torch.ops.aten.sqrt.default(add_1782);  add_1782 = None
        reciprocal_612 = torch.ops.aten.reciprocal.default(sqrt_612);  sqrt_612 = None
        mul_2060 = torch.ops.aten.mul.Tensor(reciprocal_612, 1);  reciprocal_612 = None
        unsqueeze_4952 = torch.ops.aten.unsqueeze.default(arg1437_1, -1);  arg1437_1 = None
        unsqueeze_4953 = torch.ops.aten.unsqueeze.default(unsqueeze_4952, -1);  unsqueeze_4952 = None
        unsqueeze_4954 = torch.ops.aten.unsqueeze.default(mul_2060, -1);  mul_2060 = None
        unsqueeze_4955 = torch.ops.aten.unsqueeze.default(unsqueeze_4954, -1);  unsqueeze_4954 = None
        sub_612 = torch.ops.aten.sub.Tensor(convolution_612, unsqueeze_4953);  convolution_612 = unsqueeze_4953 = None
        mul_2061 = torch.ops.aten.mul.Tensor(sub_612, unsqueeze_4955);  sub_612 = unsqueeze_4955 = None
        unsqueeze_4956 = torch.ops.aten.unsqueeze.default(arg1439_1, -1);  arg1439_1 = None
        unsqueeze_4957 = torch.ops.aten.unsqueeze.default(unsqueeze_4956, -1);  unsqueeze_4956 = None
        mul_2062 = torch.ops.aten.mul.Tensor(mul_2061, unsqueeze_4957);  mul_2061 = unsqueeze_4957 = None
        unsqueeze_4958 = torch.ops.aten.unsqueeze.default(arg1440_1, -1);  arg1440_1 = None
        unsqueeze_4959 = torch.ops.aten.unsqueeze.default(unsqueeze_4958, -1);  unsqueeze_4958 = None
        add_1783 = torch.ops.aten.add.Tensor(mul_2062, unsqueeze_4959);  mul_2062 = unsqueeze_4959 = None
        relu_542 = torch.ops.aten.relu.default(add_1783);  add_1783 = None
        convolution_613 = torch.ops.aten.convolution.default(relu_542, arg1441_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_542 = arg1441_1 = None
        add_1784 = torch.ops.aten.add.Tensor(arg1443_1, 1e-05);  arg1443_1 = None
        sqrt_613 = torch.ops.aten.sqrt.default(add_1784);  add_1784 = None
        reciprocal_613 = torch.ops.aten.reciprocal.default(sqrt_613);  sqrt_613 = None
        mul_2063 = torch.ops.aten.mul.Tensor(reciprocal_613, 1);  reciprocal_613 = None
        unsqueeze_4960 = torch.ops.aten.unsqueeze.default(arg1442_1, -1);  arg1442_1 = None
        unsqueeze_4961 = torch.ops.aten.unsqueeze.default(unsqueeze_4960, -1);  unsqueeze_4960 = None
        unsqueeze_4962 = torch.ops.aten.unsqueeze.default(mul_2063, -1);  mul_2063 = None
        unsqueeze_4963 = torch.ops.aten.unsqueeze.default(unsqueeze_4962, -1);  unsqueeze_4962 = None
        sub_613 = torch.ops.aten.sub.Tensor(convolution_613, unsqueeze_4961);  convolution_613 = unsqueeze_4961 = None
        mul_2064 = torch.ops.aten.mul.Tensor(sub_613, unsqueeze_4963);  sub_613 = unsqueeze_4963 = None
        unsqueeze_4964 = torch.ops.aten.unsqueeze.default(arg1444_1, -1);  arg1444_1 = None
        unsqueeze_4965 = torch.ops.aten.unsqueeze.default(unsqueeze_4964, -1);  unsqueeze_4964 = None
        mul_2065 = torch.ops.aten.mul.Tensor(mul_2064, unsqueeze_4965);  mul_2064 = unsqueeze_4965 = None
        unsqueeze_4966 = torch.ops.aten.unsqueeze.default(arg1445_1, -1);  arg1445_1 = None
        unsqueeze_4967 = torch.ops.aten.unsqueeze.default(unsqueeze_4966, -1);  unsqueeze_4966 = None
        add_1785 = torch.ops.aten.add.Tensor(mul_2065, unsqueeze_4967);  mul_2065 = unsqueeze_4967 = None
        add_1786 = torch.ops.aten.add.Tensor(add_1785, relu_541);  add_1785 = relu_541 = None
        relu_543 = torch.ops.aten.relu.default(add_1786);  add_1786 = None
        convolution_614 = torch.ops.aten.convolution.default(relu_527, arg1446_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1446_1 = None
        add_1787 = torch.ops.aten.add.Tensor(arg1448_1, 1e-05);  arg1448_1 = None
        sqrt_614 = torch.ops.aten.sqrt.default(add_1787);  add_1787 = None
        reciprocal_614 = torch.ops.aten.reciprocal.default(sqrt_614);  sqrt_614 = None
        mul_2066 = torch.ops.aten.mul.Tensor(reciprocal_614, 1);  reciprocal_614 = None
        unsqueeze_4968 = torch.ops.aten.unsqueeze.default(arg1447_1, -1);  arg1447_1 = None
        unsqueeze_4969 = torch.ops.aten.unsqueeze.default(unsqueeze_4968, -1);  unsqueeze_4968 = None
        unsqueeze_4970 = torch.ops.aten.unsqueeze.default(mul_2066, -1);  mul_2066 = None
        unsqueeze_4971 = torch.ops.aten.unsqueeze.default(unsqueeze_4970, -1);  unsqueeze_4970 = None
        sub_614 = torch.ops.aten.sub.Tensor(convolution_614, unsqueeze_4969);  convolution_614 = unsqueeze_4969 = None
        mul_2067 = torch.ops.aten.mul.Tensor(sub_614, unsqueeze_4971);  sub_614 = unsqueeze_4971 = None
        unsqueeze_4972 = torch.ops.aten.unsqueeze.default(arg1449_1, -1);  arg1449_1 = None
        unsqueeze_4973 = torch.ops.aten.unsqueeze.default(unsqueeze_4972, -1);  unsqueeze_4972 = None
        mul_2068 = torch.ops.aten.mul.Tensor(mul_2067, unsqueeze_4973);  mul_2067 = unsqueeze_4973 = None
        unsqueeze_4974 = torch.ops.aten.unsqueeze.default(arg1450_1, -1);  arg1450_1 = None
        unsqueeze_4975 = torch.ops.aten.unsqueeze.default(unsqueeze_4974, -1);  unsqueeze_4974 = None
        add_1788 = torch.ops.aten.add.Tensor(mul_2068, unsqueeze_4975);  mul_2068 = unsqueeze_4975 = None
        iota_112 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2069 = torch.ops.aten.mul.Tensor(iota_112, 1);  iota_112 = None
        add_1789 = torch.ops.aten.add.Tensor(mul_2069, 0);  mul_2069 = None
        convert_element_type_1454 = torch.ops.prims.convert_element_type.default(add_1789, torch.float32);  add_1789 = None
        add_1790 = torch.ops.aten.add.Tensor(convert_element_type_1454, 0.0);  convert_element_type_1454 = None
        mul_2070 = torch.ops.aten.mul.Tensor(add_1790, 0.5);  add_1790 = None
        convert_element_type_1455 = torch.ops.prims.convert_element_type.default(mul_2070, torch.int64);  mul_2070 = None
        unsqueeze_4976 = torch.ops.aten.unsqueeze.default(convert_element_type_1455, -1);  convert_element_type_1455 = None
        iota_113 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2071 = torch.ops.aten.mul.Tensor(iota_113, 1);  iota_113 = None
        add_1791 = torch.ops.aten.add.Tensor(mul_2071, 0);  mul_2071 = None
        convert_element_type_1456 = torch.ops.prims.convert_element_type.default(add_1791, torch.float32);  add_1791 = None
        add_1792 = torch.ops.aten.add.Tensor(convert_element_type_1456, 0.0);  convert_element_type_1456 = None
        mul_2072 = torch.ops.aten.mul.Tensor(add_1792, 0.5);  add_1792 = None
        convert_element_type_1457 = torch.ops.prims.convert_element_type.default(mul_2072, torch.int64);  mul_2072 = None
        _unsafe_index_56 = torch.ops.aten._unsafe_index.Tensor(add_1788, [None, None, unsqueeze_4976, convert_element_type_1457]);  add_1788 = unsqueeze_4976 = convert_element_type_1457 = None
        add_1793 = torch.ops.aten.add.Tensor(relu_519, _unsafe_index_56);  _unsafe_index_56 = None
        convolution_615 = torch.ops.aten.convolution.default(relu_535, arg1451_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1451_1 = None
        add_1794 = torch.ops.aten.add.Tensor(arg1453_1, 1e-05);  arg1453_1 = None
        sqrt_615 = torch.ops.aten.sqrt.default(add_1794);  add_1794 = None
        reciprocal_615 = torch.ops.aten.reciprocal.default(sqrt_615);  sqrt_615 = None
        mul_2073 = torch.ops.aten.mul.Tensor(reciprocal_615, 1);  reciprocal_615 = None
        unsqueeze_4977 = torch.ops.aten.unsqueeze.default(arg1452_1, -1);  arg1452_1 = None
        unsqueeze_4978 = torch.ops.aten.unsqueeze.default(unsqueeze_4977, -1);  unsqueeze_4977 = None
        unsqueeze_4979 = torch.ops.aten.unsqueeze.default(mul_2073, -1);  mul_2073 = None
        unsqueeze_4980 = torch.ops.aten.unsqueeze.default(unsqueeze_4979, -1);  unsqueeze_4979 = None
        sub_615 = torch.ops.aten.sub.Tensor(convolution_615, unsqueeze_4978);  convolution_615 = unsqueeze_4978 = None
        mul_2074 = torch.ops.aten.mul.Tensor(sub_615, unsqueeze_4980);  sub_615 = unsqueeze_4980 = None
        unsqueeze_4981 = torch.ops.aten.unsqueeze.default(arg1454_1, -1);  arg1454_1 = None
        unsqueeze_4982 = torch.ops.aten.unsqueeze.default(unsqueeze_4981, -1);  unsqueeze_4981 = None
        mul_2075 = torch.ops.aten.mul.Tensor(mul_2074, unsqueeze_4982);  mul_2074 = unsqueeze_4982 = None
        unsqueeze_4983 = torch.ops.aten.unsqueeze.default(arg1455_1, -1);  arg1455_1 = None
        unsqueeze_4984 = torch.ops.aten.unsqueeze.default(unsqueeze_4983, -1);  unsqueeze_4983 = None
        add_1795 = torch.ops.aten.add.Tensor(mul_2075, unsqueeze_4984);  mul_2075 = unsqueeze_4984 = None
        iota_114 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2076 = torch.ops.aten.mul.Tensor(iota_114, 1);  iota_114 = None
        add_1796 = torch.ops.aten.add.Tensor(mul_2076, 0);  mul_2076 = None
        convert_element_type_1460 = torch.ops.prims.convert_element_type.default(add_1796, torch.float32);  add_1796 = None
        add_1797 = torch.ops.aten.add.Tensor(convert_element_type_1460, 0.0);  convert_element_type_1460 = None
        mul_2077 = torch.ops.aten.mul.Tensor(add_1797, 0.25);  add_1797 = None
        convert_element_type_1461 = torch.ops.prims.convert_element_type.default(mul_2077, torch.int64);  mul_2077 = None
        unsqueeze_4985 = torch.ops.aten.unsqueeze.default(convert_element_type_1461, -1);  convert_element_type_1461 = None
        iota_115 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2078 = torch.ops.aten.mul.Tensor(iota_115, 1);  iota_115 = None
        add_1798 = torch.ops.aten.add.Tensor(mul_2078, 0);  mul_2078 = None
        convert_element_type_1462 = torch.ops.prims.convert_element_type.default(add_1798, torch.float32);  add_1798 = None
        add_1799 = torch.ops.aten.add.Tensor(convert_element_type_1462, 0.0);  convert_element_type_1462 = None
        mul_2079 = torch.ops.aten.mul.Tensor(add_1799, 0.25);  add_1799 = None
        convert_element_type_1463 = torch.ops.prims.convert_element_type.default(mul_2079, torch.int64);  mul_2079 = None
        _unsafe_index_57 = torch.ops.aten._unsafe_index.Tensor(add_1795, [None, None, unsqueeze_4985, convert_element_type_1463]);  add_1795 = unsqueeze_4985 = convert_element_type_1463 = None
        add_1800 = torch.ops.aten.add.Tensor(add_1793, _unsafe_index_57);  add_1793 = _unsafe_index_57 = None
        convolution_616 = torch.ops.aten.convolution.default(relu_543, arg1456_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1456_1 = None
        add_1801 = torch.ops.aten.add.Tensor(arg1458_1, 1e-05);  arg1458_1 = None
        sqrt_616 = torch.ops.aten.sqrt.default(add_1801);  add_1801 = None
        reciprocal_616 = torch.ops.aten.reciprocal.default(sqrt_616);  sqrt_616 = None
        mul_2080 = torch.ops.aten.mul.Tensor(reciprocal_616, 1);  reciprocal_616 = None
        unsqueeze_4986 = torch.ops.aten.unsqueeze.default(arg1457_1, -1);  arg1457_1 = None
        unsqueeze_4987 = torch.ops.aten.unsqueeze.default(unsqueeze_4986, -1);  unsqueeze_4986 = None
        unsqueeze_4988 = torch.ops.aten.unsqueeze.default(mul_2080, -1);  mul_2080 = None
        unsqueeze_4989 = torch.ops.aten.unsqueeze.default(unsqueeze_4988, -1);  unsqueeze_4988 = None
        sub_616 = torch.ops.aten.sub.Tensor(convolution_616, unsqueeze_4987);  convolution_616 = unsqueeze_4987 = None
        mul_2081 = torch.ops.aten.mul.Tensor(sub_616, unsqueeze_4989);  sub_616 = unsqueeze_4989 = None
        unsqueeze_4990 = torch.ops.aten.unsqueeze.default(arg1459_1, -1);  arg1459_1 = None
        unsqueeze_4991 = torch.ops.aten.unsqueeze.default(unsqueeze_4990, -1);  unsqueeze_4990 = None
        mul_2082 = torch.ops.aten.mul.Tensor(mul_2081, unsqueeze_4991);  mul_2081 = unsqueeze_4991 = None
        unsqueeze_4992 = torch.ops.aten.unsqueeze.default(arg1460_1, -1);  arg1460_1 = None
        unsqueeze_4993 = torch.ops.aten.unsqueeze.default(unsqueeze_4992, -1);  unsqueeze_4992 = None
        add_1802 = torch.ops.aten.add.Tensor(mul_2082, unsqueeze_4993);  mul_2082 = unsqueeze_4993 = None
        iota_116 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2083 = torch.ops.aten.mul.Tensor(iota_116, 1);  iota_116 = None
        add_1803 = torch.ops.aten.add.Tensor(mul_2083, 0);  mul_2083 = None
        convert_element_type_1466 = torch.ops.prims.convert_element_type.default(add_1803, torch.float32);  add_1803 = None
        add_1804 = torch.ops.aten.add.Tensor(convert_element_type_1466, 0.0);  convert_element_type_1466 = None
        mul_2084 = torch.ops.aten.mul.Tensor(add_1804, 0.125);  add_1804 = None
        convert_element_type_1467 = torch.ops.prims.convert_element_type.default(mul_2084, torch.int64);  mul_2084 = None
        unsqueeze_4994 = torch.ops.aten.unsqueeze.default(convert_element_type_1467, -1);  convert_element_type_1467 = None
        iota_117 = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2085 = torch.ops.aten.mul.Tensor(iota_117, 1);  iota_117 = None
        add_1805 = torch.ops.aten.add.Tensor(mul_2085, 0);  mul_2085 = None
        convert_element_type_1468 = torch.ops.prims.convert_element_type.default(add_1805, torch.float32);  add_1805 = None
        add_1806 = torch.ops.aten.add.Tensor(convert_element_type_1468, 0.0);  convert_element_type_1468 = None
        mul_2086 = torch.ops.aten.mul.Tensor(add_1806, 0.125);  add_1806 = None
        convert_element_type_1469 = torch.ops.prims.convert_element_type.default(mul_2086, torch.int64);  mul_2086 = None
        _unsafe_index_58 = torch.ops.aten._unsafe_index.Tensor(add_1802, [None, None, unsqueeze_4994, convert_element_type_1469]);  add_1802 = unsqueeze_4994 = convert_element_type_1469 = None
        add_1807 = torch.ops.aten.add.Tensor(add_1800, _unsafe_index_58);  add_1800 = _unsafe_index_58 = None
        relu_544 = torch.ops.aten.relu.default(add_1807);  add_1807 = None
        convolution_617 = torch.ops.aten.convolution.default(relu_519, arg1461_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1461_1 = None
        add_1808 = torch.ops.aten.add.Tensor(arg1463_1, 1e-05);  arg1463_1 = None
        sqrt_617 = torch.ops.aten.sqrt.default(add_1808);  add_1808 = None
        reciprocal_617 = torch.ops.aten.reciprocal.default(sqrt_617);  sqrt_617 = None
        mul_2087 = torch.ops.aten.mul.Tensor(reciprocal_617, 1);  reciprocal_617 = None
        unsqueeze_4995 = torch.ops.aten.unsqueeze.default(arg1462_1, -1);  arg1462_1 = None
        unsqueeze_4996 = torch.ops.aten.unsqueeze.default(unsqueeze_4995, -1);  unsqueeze_4995 = None
        unsqueeze_4997 = torch.ops.aten.unsqueeze.default(mul_2087, -1);  mul_2087 = None
        unsqueeze_4998 = torch.ops.aten.unsqueeze.default(unsqueeze_4997, -1);  unsqueeze_4997 = None
        sub_617 = torch.ops.aten.sub.Tensor(convolution_617, unsqueeze_4996);  convolution_617 = unsqueeze_4996 = None
        mul_2088 = torch.ops.aten.mul.Tensor(sub_617, unsqueeze_4998);  sub_617 = unsqueeze_4998 = None
        unsqueeze_4999 = torch.ops.aten.unsqueeze.default(arg1464_1, -1);  arg1464_1 = None
        unsqueeze_5000 = torch.ops.aten.unsqueeze.default(unsqueeze_4999, -1);  unsqueeze_4999 = None
        mul_2089 = torch.ops.aten.mul.Tensor(mul_2088, unsqueeze_5000);  mul_2088 = unsqueeze_5000 = None
        unsqueeze_5001 = torch.ops.aten.unsqueeze.default(arg1465_1, -1);  arg1465_1 = None
        unsqueeze_5002 = torch.ops.aten.unsqueeze.default(unsqueeze_5001, -1);  unsqueeze_5001 = None
        add_1809 = torch.ops.aten.add.Tensor(mul_2089, unsqueeze_5002);  mul_2089 = unsqueeze_5002 = None
        add_1810 = torch.ops.aten.add.Tensor(add_1809, relu_527);  add_1809 = None
        convolution_618 = torch.ops.aten.convolution.default(relu_535, arg1466_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1466_1 = None
        add_1811 = torch.ops.aten.add.Tensor(arg1468_1, 1e-05);  arg1468_1 = None
        sqrt_618 = torch.ops.aten.sqrt.default(add_1811);  add_1811 = None
        reciprocal_618 = torch.ops.aten.reciprocal.default(sqrt_618);  sqrt_618 = None
        mul_2090 = torch.ops.aten.mul.Tensor(reciprocal_618, 1);  reciprocal_618 = None
        unsqueeze_5003 = torch.ops.aten.unsqueeze.default(arg1467_1, -1);  arg1467_1 = None
        unsqueeze_5004 = torch.ops.aten.unsqueeze.default(unsqueeze_5003, -1);  unsqueeze_5003 = None
        unsqueeze_5005 = torch.ops.aten.unsqueeze.default(mul_2090, -1);  mul_2090 = None
        unsqueeze_5006 = torch.ops.aten.unsqueeze.default(unsqueeze_5005, -1);  unsqueeze_5005 = None
        sub_618 = torch.ops.aten.sub.Tensor(convolution_618, unsqueeze_5004);  convolution_618 = unsqueeze_5004 = None
        mul_2091 = torch.ops.aten.mul.Tensor(sub_618, unsqueeze_5006);  sub_618 = unsqueeze_5006 = None
        unsqueeze_5007 = torch.ops.aten.unsqueeze.default(arg1469_1, -1);  arg1469_1 = None
        unsqueeze_5008 = torch.ops.aten.unsqueeze.default(unsqueeze_5007, -1);  unsqueeze_5007 = None
        mul_2092 = torch.ops.aten.mul.Tensor(mul_2091, unsqueeze_5008);  mul_2091 = unsqueeze_5008 = None
        unsqueeze_5009 = torch.ops.aten.unsqueeze.default(arg1470_1, -1);  arg1470_1 = None
        unsqueeze_5010 = torch.ops.aten.unsqueeze.default(unsqueeze_5009, -1);  unsqueeze_5009 = None
        add_1812 = torch.ops.aten.add.Tensor(mul_2092, unsqueeze_5010);  mul_2092 = unsqueeze_5010 = None
        iota_118 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2093 = torch.ops.aten.mul.Tensor(iota_118, 1);  iota_118 = None
        add_1813 = torch.ops.aten.add.Tensor(mul_2093, 0);  mul_2093 = None
        convert_element_type_1474 = torch.ops.prims.convert_element_type.default(add_1813, torch.float32);  add_1813 = None
        add_1814 = torch.ops.aten.add.Tensor(convert_element_type_1474, 0.0);  convert_element_type_1474 = None
        mul_2094 = torch.ops.aten.mul.Tensor(add_1814, 0.5);  add_1814 = None
        convert_element_type_1475 = torch.ops.prims.convert_element_type.default(mul_2094, torch.int64);  mul_2094 = None
        unsqueeze_5011 = torch.ops.aten.unsqueeze.default(convert_element_type_1475, -1);  convert_element_type_1475 = None
        iota_119 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2095 = torch.ops.aten.mul.Tensor(iota_119, 1);  iota_119 = None
        add_1815 = torch.ops.aten.add.Tensor(mul_2095, 0);  mul_2095 = None
        convert_element_type_1476 = torch.ops.prims.convert_element_type.default(add_1815, torch.float32);  add_1815 = None
        add_1816 = torch.ops.aten.add.Tensor(convert_element_type_1476, 0.0);  convert_element_type_1476 = None
        mul_2096 = torch.ops.aten.mul.Tensor(add_1816, 0.5);  add_1816 = None
        convert_element_type_1477 = torch.ops.prims.convert_element_type.default(mul_2096, torch.int64);  mul_2096 = None
        _unsafe_index_59 = torch.ops.aten._unsafe_index.Tensor(add_1812, [None, None, unsqueeze_5011, convert_element_type_1477]);  add_1812 = unsqueeze_5011 = convert_element_type_1477 = None
        add_1817 = torch.ops.aten.add.Tensor(add_1810, _unsafe_index_59);  add_1810 = _unsafe_index_59 = None
        convolution_619 = torch.ops.aten.convolution.default(relu_543, arg1471_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1471_1 = None
        add_1818 = torch.ops.aten.add.Tensor(arg1473_1, 1e-05);  arg1473_1 = None
        sqrt_619 = torch.ops.aten.sqrt.default(add_1818);  add_1818 = None
        reciprocal_619 = torch.ops.aten.reciprocal.default(sqrt_619);  sqrt_619 = None
        mul_2097 = torch.ops.aten.mul.Tensor(reciprocal_619, 1);  reciprocal_619 = None
        unsqueeze_5012 = torch.ops.aten.unsqueeze.default(arg1472_1, -1);  arg1472_1 = None
        unsqueeze_5013 = torch.ops.aten.unsqueeze.default(unsqueeze_5012, -1);  unsqueeze_5012 = None
        unsqueeze_5014 = torch.ops.aten.unsqueeze.default(mul_2097, -1);  mul_2097 = None
        unsqueeze_5015 = torch.ops.aten.unsqueeze.default(unsqueeze_5014, -1);  unsqueeze_5014 = None
        sub_619 = torch.ops.aten.sub.Tensor(convolution_619, unsqueeze_5013);  convolution_619 = unsqueeze_5013 = None
        mul_2098 = torch.ops.aten.mul.Tensor(sub_619, unsqueeze_5015);  sub_619 = unsqueeze_5015 = None
        unsqueeze_5016 = torch.ops.aten.unsqueeze.default(arg1474_1, -1);  arg1474_1 = None
        unsqueeze_5017 = torch.ops.aten.unsqueeze.default(unsqueeze_5016, -1);  unsqueeze_5016 = None
        mul_2099 = torch.ops.aten.mul.Tensor(mul_2098, unsqueeze_5017);  mul_2098 = unsqueeze_5017 = None
        unsqueeze_5018 = torch.ops.aten.unsqueeze.default(arg1475_1, -1);  arg1475_1 = None
        unsqueeze_5019 = torch.ops.aten.unsqueeze.default(unsqueeze_5018, -1);  unsqueeze_5018 = None
        add_1819 = torch.ops.aten.add.Tensor(mul_2099, unsqueeze_5019);  mul_2099 = unsqueeze_5019 = None
        iota_120 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2100 = torch.ops.aten.mul.Tensor(iota_120, 1);  iota_120 = None
        add_1820 = torch.ops.aten.add.Tensor(mul_2100, 0);  mul_2100 = None
        convert_element_type_1480 = torch.ops.prims.convert_element_type.default(add_1820, torch.float32);  add_1820 = None
        add_1821 = torch.ops.aten.add.Tensor(convert_element_type_1480, 0.0);  convert_element_type_1480 = None
        mul_2101 = torch.ops.aten.mul.Tensor(add_1821, 0.25);  add_1821 = None
        convert_element_type_1481 = torch.ops.prims.convert_element_type.default(mul_2101, torch.int64);  mul_2101 = None
        unsqueeze_5020 = torch.ops.aten.unsqueeze.default(convert_element_type_1481, -1);  convert_element_type_1481 = None
        iota_121 = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2102 = torch.ops.aten.mul.Tensor(iota_121, 1);  iota_121 = None
        add_1822 = torch.ops.aten.add.Tensor(mul_2102, 0);  mul_2102 = None
        convert_element_type_1482 = torch.ops.prims.convert_element_type.default(add_1822, torch.float32);  add_1822 = None
        add_1823 = torch.ops.aten.add.Tensor(convert_element_type_1482, 0.0);  convert_element_type_1482 = None
        mul_2103 = torch.ops.aten.mul.Tensor(add_1823, 0.25);  add_1823 = None
        convert_element_type_1483 = torch.ops.prims.convert_element_type.default(mul_2103, torch.int64);  mul_2103 = None
        _unsafe_index_60 = torch.ops.aten._unsafe_index.Tensor(add_1819, [None, None, unsqueeze_5020, convert_element_type_1483]);  add_1819 = unsqueeze_5020 = convert_element_type_1483 = None
        add_1824 = torch.ops.aten.add.Tensor(add_1817, _unsafe_index_60);  add_1817 = _unsafe_index_60 = None
        relu_545 = torch.ops.aten.relu.default(add_1824);  add_1824 = None
        convolution_620 = torch.ops.aten.convolution.default(relu_519, arg1476_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1476_1 = None
        add_1825 = torch.ops.aten.add.Tensor(arg1478_1, 1e-05);  arg1478_1 = None
        sqrt_620 = torch.ops.aten.sqrt.default(add_1825);  add_1825 = None
        reciprocal_620 = torch.ops.aten.reciprocal.default(sqrt_620);  sqrt_620 = None
        mul_2104 = torch.ops.aten.mul.Tensor(reciprocal_620, 1);  reciprocal_620 = None
        unsqueeze_5021 = torch.ops.aten.unsqueeze.default(arg1477_1, -1);  arg1477_1 = None
        unsqueeze_5022 = torch.ops.aten.unsqueeze.default(unsqueeze_5021, -1);  unsqueeze_5021 = None
        unsqueeze_5023 = torch.ops.aten.unsqueeze.default(mul_2104, -1);  mul_2104 = None
        unsqueeze_5024 = torch.ops.aten.unsqueeze.default(unsqueeze_5023, -1);  unsqueeze_5023 = None
        sub_620 = torch.ops.aten.sub.Tensor(convolution_620, unsqueeze_5022);  convolution_620 = unsqueeze_5022 = None
        mul_2105 = torch.ops.aten.mul.Tensor(sub_620, unsqueeze_5024);  sub_620 = unsqueeze_5024 = None
        unsqueeze_5025 = torch.ops.aten.unsqueeze.default(arg1479_1, -1);  arg1479_1 = None
        unsqueeze_5026 = torch.ops.aten.unsqueeze.default(unsqueeze_5025, -1);  unsqueeze_5025 = None
        mul_2106 = torch.ops.aten.mul.Tensor(mul_2105, unsqueeze_5026);  mul_2105 = unsqueeze_5026 = None
        unsqueeze_5027 = torch.ops.aten.unsqueeze.default(arg1480_1, -1);  arg1480_1 = None
        unsqueeze_5028 = torch.ops.aten.unsqueeze.default(unsqueeze_5027, -1);  unsqueeze_5027 = None
        add_1826 = torch.ops.aten.add.Tensor(mul_2106, unsqueeze_5028);  mul_2106 = unsqueeze_5028 = None
        relu_546 = torch.ops.aten.relu.default(add_1826);  add_1826 = None
        convolution_621 = torch.ops.aten.convolution.default(relu_546, arg1481_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_546 = arg1481_1 = None
        add_1827 = torch.ops.aten.add.Tensor(arg1483_1, 1e-05);  arg1483_1 = None
        sqrt_621 = torch.ops.aten.sqrt.default(add_1827);  add_1827 = None
        reciprocal_621 = torch.ops.aten.reciprocal.default(sqrt_621);  sqrt_621 = None
        mul_2107 = torch.ops.aten.mul.Tensor(reciprocal_621, 1);  reciprocal_621 = None
        unsqueeze_5029 = torch.ops.aten.unsqueeze.default(arg1482_1, -1);  arg1482_1 = None
        unsqueeze_5030 = torch.ops.aten.unsqueeze.default(unsqueeze_5029, -1);  unsqueeze_5029 = None
        unsqueeze_5031 = torch.ops.aten.unsqueeze.default(mul_2107, -1);  mul_2107 = None
        unsqueeze_5032 = torch.ops.aten.unsqueeze.default(unsqueeze_5031, -1);  unsqueeze_5031 = None
        sub_621 = torch.ops.aten.sub.Tensor(convolution_621, unsqueeze_5030);  convolution_621 = unsqueeze_5030 = None
        mul_2108 = torch.ops.aten.mul.Tensor(sub_621, unsqueeze_5032);  sub_621 = unsqueeze_5032 = None
        unsqueeze_5033 = torch.ops.aten.unsqueeze.default(arg1484_1, -1);  arg1484_1 = None
        unsqueeze_5034 = torch.ops.aten.unsqueeze.default(unsqueeze_5033, -1);  unsqueeze_5033 = None
        mul_2109 = torch.ops.aten.mul.Tensor(mul_2108, unsqueeze_5034);  mul_2108 = unsqueeze_5034 = None
        unsqueeze_5035 = torch.ops.aten.unsqueeze.default(arg1485_1, -1);  arg1485_1 = None
        unsqueeze_5036 = torch.ops.aten.unsqueeze.default(unsqueeze_5035, -1);  unsqueeze_5035 = None
        add_1828 = torch.ops.aten.add.Tensor(mul_2109, unsqueeze_5036);  mul_2109 = unsqueeze_5036 = None
        convolution_622 = torch.ops.aten.convolution.default(relu_527, arg1486_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1486_1 = None
        add_1829 = torch.ops.aten.add.Tensor(arg1488_1, 1e-05);  arg1488_1 = None
        sqrt_622 = torch.ops.aten.sqrt.default(add_1829);  add_1829 = None
        reciprocal_622 = torch.ops.aten.reciprocal.default(sqrt_622);  sqrt_622 = None
        mul_2110 = torch.ops.aten.mul.Tensor(reciprocal_622, 1);  reciprocal_622 = None
        unsqueeze_5037 = torch.ops.aten.unsqueeze.default(arg1487_1, -1);  arg1487_1 = None
        unsqueeze_5038 = torch.ops.aten.unsqueeze.default(unsqueeze_5037, -1);  unsqueeze_5037 = None
        unsqueeze_5039 = torch.ops.aten.unsqueeze.default(mul_2110, -1);  mul_2110 = None
        unsqueeze_5040 = torch.ops.aten.unsqueeze.default(unsqueeze_5039, -1);  unsqueeze_5039 = None
        sub_622 = torch.ops.aten.sub.Tensor(convolution_622, unsqueeze_5038);  convolution_622 = unsqueeze_5038 = None
        mul_2111 = torch.ops.aten.mul.Tensor(sub_622, unsqueeze_5040);  sub_622 = unsqueeze_5040 = None
        unsqueeze_5041 = torch.ops.aten.unsqueeze.default(arg1489_1, -1);  arg1489_1 = None
        unsqueeze_5042 = torch.ops.aten.unsqueeze.default(unsqueeze_5041, -1);  unsqueeze_5041 = None
        mul_2112 = torch.ops.aten.mul.Tensor(mul_2111, unsqueeze_5042);  mul_2111 = unsqueeze_5042 = None
        unsqueeze_5043 = torch.ops.aten.unsqueeze.default(arg1490_1, -1);  arg1490_1 = None
        unsqueeze_5044 = torch.ops.aten.unsqueeze.default(unsqueeze_5043, -1);  unsqueeze_5043 = None
        add_1830 = torch.ops.aten.add.Tensor(mul_2112, unsqueeze_5044);  mul_2112 = unsqueeze_5044 = None
        add_1831 = torch.ops.aten.add.Tensor(add_1828, add_1830);  add_1828 = add_1830 = None
        add_1832 = torch.ops.aten.add.Tensor(add_1831, relu_535);  add_1831 = None
        convolution_623 = torch.ops.aten.convolution.default(relu_543, arg1491_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1491_1 = None
        add_1833 = torch.ops.aten.add.Tensor(arg1493_1, 1e-05);  arg1493_1 = None
        sqrt_623 = torch.ops.aten.sqrt.default(add_1833);  add_1833 = None
        reciprocal_623 = torch.ops.aten.reciprocal.default(sqrt_623);  sqrt_623 = None
        mul_2113 = torch.ops.aten.mul.Tensor(reciprocal_623, 1);  reciprocal_623 = None
        unsqueeze_5045 = torch.ops.aten.unsqueeze.default(arg1492_1, -1);  arg1492_1 = None
        unsqueeze_5046 = torch.ops.aten.unsqueeze.default(unsqueeze_5045, -1);  unsqueeze_5045 = None
        unsqueeze_5047 = torch.ops.aten.unsqueeze.default(mul_2113, -1);  mul_2113 = None
        unsqueeze_5048 = torch.ops.aten.unsqueeze.default(unsqueeze_5047, -1);  unsqueeze_5047 = None
        sub_623 = torch.ops.aten.sub.Tensor(convolution_623, unsqueeze_5046);  convolution_623 = unsqueeze_5046 = None
        mul_2114 = torch.ops.aten.mul.Tensor(sub_623, unsqueeze_5048);  sub_623 = unsqueeze_5048 = None
        unsqueeze_5049 = torch.ops.aten.unsqueeze.default(arg1494_1, -1);  arg1494_1 = None
        unsqueeze_5050 = torch.ops.aten.unsqueeze.default(unsqueeze_5049, -1);  unsqueeze_5049 = None
        mul_2115 = torch.ops.aten.mul.Tensor(mul_2114, unsqueeze_5050);  mul_2114 = unsqueeze_5050 = None
        unsqueeze_5051 = torch.ops.aten.unsqueeze.default(arg1495_1, -1);  arg1495_1 = None
        unsqueeze_5052 = torch.ops.aten.unsqueeze.default(unsqueeze_5051, -1);  unsqueeze_5051 = None
        add_1834 = torch.ops.aten.add.Tensor(mul_2115, unsqueeze_5052);  mul_2115 = unsqueeze_5052 = None
        iota_122 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2116 = torch.ops.aten.mul.Tensor(iota_122, 1);  iota_122 = None
        add_1835 = torch.ops.aten.add.Tensor(mul_2116, 0);  mul_2116 = None
        convert_element_type_1492 = torch.ops.prims.convert_element_type.default(add_1835, torch.float32);  add_1835 = None
        add_1836 = torch.ops.aten.add.Tensor(convert_element_type_1492, 0.0);  convert_element_type_1492 = None
        mul_2117 = torch.ops.aten.mul.Tensor(add_1836, 0.5);  add_1836 = None
        convert_element_type_1493 = torch.ops.prims.convert_element_type.default(mul_2117, torch.int64);  mul_2117 = None
        unsqueeze_5053 = torch.ops.aten.unsqueeze.default(convert_element_type_1493, -1);  convert_element_type_1493 = None
        iota_123 = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_2118 = torch.ops.aten.mul.Tensor(iota_123, 1);  iota_123 = None
        add_1837 = torch.ops.aten.add.Tensor(mul_2118, 0);  mul_2118 = None
        convert_element_type_1494 = torch.ops.prims.convert_element_type.default(add_1837, torch.float32);  add_1837 = None
        add_1838 = torch.ops.aten.add.Tensor(convert_element_type_1494, 0.0);  convert_element_type_1494 = None
        mul_2119 = torch.ops.aten.mul.Tensor(add_1838, 0.5);  add_1838 = None
        convert_element_type_1495 = torch.ops.prims.convert_element_type.default(mul_2119, torch.int64);  mul_2119 = None
        _unsafe_index_61 = torch.ops.aten._unsafe_index.Tensor(add_1834, [None, None, unsqueeze_5053, convert_element_type_1495]);  add_1834 = unsqueeze_5053 = convert_element_type_1495 = None
        add_1839 = torch.ops.aten.add.Tensor(add_1832, _unsafe_index_61);  add_1832 = _unsafe_index_61 = None
        relu_547 = torch.ops.aten.relu.default(add_1839);  add_1839 = None
        convolution_624 = torch.ops.aten.convolution.default(relu_519, arg1496_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_519 = arg1496_1 = None
        add_1840 = torch.ops.aten.add.Tensor(arg1498_1, 1e-05);  arg1498_1 = None
        sqrt_624 = torch.ops.aten.sqrt.default(add_1840);  add_1840 = None
        reciprocal_624 = torch.ops.aten.reciprocal.default(sqrt_624);  sqrt_624 = None
        mul_2120 = torch.ops.aten.mul.Tensor(reciprocal_624, 1);  reciprocal_624 = None
        unsqueeze_5054 = torch.ops.aten.unsqueeze.default(arg1497_1, -1);  arg1497_1 = None
        unsqueeze_5055 = torch.ops.aten.unsqueeze.default(unsqueeze_5054, -1);  unsqueeze_5054 = None
        unsqueeze_5056 = torch.ops.aten.unsqueeze.default(mul_2120, -1);  mul_2120 = None
        unsqueeze_5057 = torch.ops.aten.unsqueeze.default(unsqueeze_5056, -1);  unsqueeze_5056 = None
        sub_624 = torch.ops.aten.sub.Tensor(convolution_624, unsqueeze_5055);  convolution_624 = unsqueeze_5055 = None
        mul_2121 = torch.ops.aten.mul.Tensor(sub_624, unsqueeze_5057);  sub_624 = unsqueeze_5057 = None
        unsqueeze_5058 = torch.ops.aten.unsqueeze.default(arg1499_1, -1);  arg1499_1 = None
        unsqueeze_5059 = torch.ops.aten.unsqueeze.default(unsqueeze_5058, -1);  unsqueeze_5058 = None
        mul_2122 = torch.ops.aten.mul.Tensor(mul_2121, unsqueeze_5059);  mul_2121 = unsqueeze_5059 = None
        unsqueeze_5060 = torch.ops.aten.unsqueeze.default(arg1500_1, -1);  arg1500_1 = None
        unsqueeze_5061 = torch.ops.aten.unsqueeze.default(unsqueeze_5060, -1);  unsqueeze_5060 = None
        add_1841 = torch.ops.aten.add.Tensor(mul_2122, unsqueeze_5061);  mul_2122 = unsqueeze_5061 = None
        relu_548 = torch.ops.aten.relu.default(add_1841);  add_1841 = None
        convolution_625 = torch.ops.aten.convolution.default(relu_548, arg1501_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_548 = arg1501_1 = None
        add_1842 = torch.ops.aten.add.Tensor(arg1503_1, 1e-05);  arg1503_1 = None
        sqrt_625 = torch.ops.aten.sqrt.default(add_1842);  add_1842 = None
        reciprocal_625 = torch.ops.aten.reciprocal.default(sqrt_625);  sqrt_625 = None
        mul_2123 = torch.ops.aten.mul.Tensor(reciprocal_625, 1);  reciprocal_625 = None
        unsqueeze_5062 = torch.ops.aten.unsqueeze.default(arg1502_1, -1);  arg1502_1 = None
        unsqueeze_5063 = torch.ops.aten.unsqueeze.default(unsqueeze_5062, -1);  unsqueeze_5062 = None
        unsqueeze_5064 = torch.ops.aten.unsqueeze.default(mul_2123, -1);  mul_2123 = None
        unsqueeze_5065 = torch.ops.aten.unsqueeze.default(unsqueeze_5064, -1);  unsqueeze_5064 = None
        sub_625 = torch.ops.aten.sub.Tensor(convolution_625, unsqueeze_5063);  convolution_625 = unsqueeze_5063 = None
        mul_2124 = torch.ops.aten.mul.Tensor(sub_625, unsqueeze_5065);  sub_625 = unsqueeze_5065 = None
        unsqueeze_5066 = torch.ops.aten.unsqueeze.default(arg1504_1, -1);  arg1504_1 = None
        unsqueeze_5067 = torch.ops.aten.unsqueeze.default(unsqueeze_5066, -1);  unsqueeze_5066 = None
        mul_2125 = torch.ops.aten.mul.Tensor(mul_2124, unsqueeze_5067);  mul_2124 = unsqueeze_5067 = None
        unsqueeze_5068 = torch.ops.aten.unsqueeze.default(arg1505_1, -1);  arg1505_1 = None
        unsqueeze_5069 = torch.ops.aten.unsqueeze.default(unsqueeze_5068, -1);  unsqueeze_5068 = None
        add_1843 = torch.ops.aten.add.Tensor(mul_2125, unsqueeze_5069);  mul_2125 = unsqueeze_5069 = None
        relu_549 = torch.ops.aten.relu.default(add_1843);  add_1843 = None
        convolution_626 = torch.ops.aten.convolution.default(relu_549, arg1506_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_549 = arg1506_1 = None
        add_1844 = torch.ops.aten.add.Tensor(arg1508_1, 1e-05);  arg1508_1 = None
        sqrt_626 = torch.ops.aten.sqrt.default(add_1844);  add_1844 = None
        reciprocal_626 = torch.ops.aten.reciprocal.default(sqrt_626);  sqrt_626 = None
        mul_2126 = torch.ops.aten.mul.Tensor(reciprocal_626, 1);  reciprocal_626 = None
        unsqueeze_5070 = torch.ops.aten.unsqueeze.default(arg1507_1, -1);  arg1507_1 = None
        unsqueeze_5071 = torch.ops.aten.unsqueeze.default(unsqueeze_5070, -1);  unsqueeze_5070 = None
        unsqueeze_5072 = torch.ops.aten.unsqueeze.default(mul_2126, -1);  mul_2126 = None
        unsqueeze_5073 = torch.ops.aten.unsqueeze.default(unsqueeze_5072, -1);  unsqueeze_5072 = None
        sub_626 = torch.ops.aten.sub.Tensor(convolution_626, unsqueeze_5071);  convolution_626 = unsqueeze_5071 = None
        mul_2127 = torch.ops.aten.mul.Tensor(sub_626, unsqueeze_5073);  sub_626 = unsqueeze_5073 = None
        unsqueeze_5074 = torch.ops.aten.unsqueeze.default(arg1509_1, -1);  arg1509_1 = None
        unsqueeze_5075 = torch.ops.aten.unsqueeze.default(unsqueeze_5074, -1);  unsqueeze_5074 = None
        mul_2128 = torch.ops.aten.mul.Tensor(mul_2127, unsqueeze_5075);  mul_2127 = unsqueeze_5075 = None
        unsqueeze_5076 = torch.ops.aten.unsqueeze.default(arg1510_1, -1);  arg1510_1 = None
        unsqueeze_5077 = torch.ops.aten.unsqueeze.default(unsqueeze_5076, -1);  unsqueeze_5076 = None
        add_1845 = torch.ops.aten.add.Tensor(mul_2128, unsqueeze_5077);  mul_2128 = unsqueeze_5077 = None
        convolution_627 = torch.ops.aten.convolution.default(relu_527, arg1511_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_527 = arg1511_1 = None
        add_1846 = torch.ops.aten.add.Tensor(arg1513_1, 1e-05);  arg1513_1 = None
        sqrt_627 = torch.ops.aten.sqrt.default(add_1846);  add_1846 = None
        reciprocal_627 = torch.ops.aten.reciprocal.default(sqrt_627);  sqrt_627 = None
        mul_2129 = torch.ops.aten.mul.Tensor(reciprocal_627, 1);  reciprocal_627 = None
        unsqueeze_5078 = torch.ops.aten.unsqueeze.default(arg1512_1, -1);  arg1512_1 = None
        unsqueeze_5079 = torch.ops.aten.unsqueeze.default(unsqueeze_5078, -1);  unsqueeze_5078 = None
        unsqueeze_5080 = torch.ops.aten.unsqueeze.default(mul_2129, -1);  mul_2129 = None
        unsqueeze_5081 = torch.ops.aten.unsqueeze.default(unsqueeze_5080, -1);  unsqueeze_5080 = None
        sub_627 = torch.ops.aten.sub.Tensor(convolution_627, unsqueeze_5079);  convolution_627 = unsqueeze_5079 = None
        mul_2130 = torch.ops.aten.mul.Tensor(sub_627, unsqueeze_5081);  sub_627 = unsqueeze_5081 = None
        unsqueeze_5082 = torch.ops.aten.unsqueeze.default(arg1514_1, -1);  arg1514_1 = None
        unsqueeze_5083 = torch.ops.aten.unsqueeze.default(unsqueeze_5082, -1);  unsqueeze_5082 = None
        mul_2131 = torch.ops.aten.mul.Tensor(mul_2130, unsqueeze_5083);  mul_2130 = unsqueeze_5083 = None
        unsqueeze_5084 = torch.ops.aten.unsqueeze.default(arg1515_1, -1);  arg1515_1 = None
        unsqueeze_5085 = torch.ops.aten.unsqueeze.default(unsqueeze_5084, -1);  unsqueeze_5084 = None
        add_1847 = torch.ops.aten.add.Tensor(mul_2131, unsqueeze_5085);  mul_2131 = unsqueeze_5085 = None
        relu_550 = torch.ops.aten.relu.default(add_1847);  add_1847 = None
        convolution_628 = torch.ops.aten.convolution.default(relu_550, arg1516_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_550 = arg1516_1 = None
        add_1848 = torch.ops.aten.add.Tensor(arg1518_1, 1e-05);  arg1518_1 = None
        sqrt_628 = torch.ops.aten.sqrt.default(add_1848);  add_1848 = None
        reciprocal_628 = torch.ops.aten.reciprocal.default(sqrt_628);  sqrt_628 = None
        mul_2132 = torch.ops.aten.mul.Tensor(reciprocal_628, 1);  reciprocal_628 = None
        unsqueeze_5086 = torch.ops.aten.unsqueeze.default(arg1517_1, -1);  arg1517_1 = None
        unsqueeze_5087 = torch.ops.aten.unsqueeze.default(unsqueeze_5086, -1);  unsqueeze_5086 = None
        unsqueeze_5088 = torch.ops.aten.unsqueeze.default(mul_2132, -1);  mul_2132 = None
        unsqueeze_5089 = torch.ops.aten.unsqueeze.default(unsqueeze_5088, -1);  unsqueeze_5088 = None
        sub_628 = torch.ops.aten.sub.Tensor(convolution_628, unsqueeze_5087);  convolution_628 = unsqueeze_5087 = None
        mul_2133 = torch.ops.aten.mul.Tensor(sub_628, unsqueeze_5089);  sub_628 = unsqueeze_5089 = None
        unsqueeze_5090 = torch.ops.aten.unsqueeze.default(arg1519_1, -1);  arg1519_1 = None
        unsqueeze_5091 = torch.ops.aten.unsqueeze.default(unsqueeze_5090, -1);  unsqueeze_5090 = None
        mul_2134 = torch.ops.aten.mul.Tensor(mul_2133, unsqueeze_5091);  mul_2133 = unsqueeze_5091 = None
        unsqueeze_5092 = torch.ops.aten.unsqueeze.default(arg1520_1, -1);  arg1520_1 = None
        unsqueeze_5093 = torch.ops.aten.unsqueeze.default(unsqueeze_5092, -1);  unsqueeze_5092 = None
        add_1849 = torch.ops.aten.add.Tensor(mul_2134, unsqueeze_5093);  mul_2134 = unsqueeze_5093 = None
        add_1850 = torch.ops.aten.add.Tensor(add_1845, add_1849);  add_1845 = add_1849 = None
        convolution_629 = torch.ops.aten.convolution.default(relu_535, arg1521_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_535 = arg1521_1 = None
        add_1851 = torch.ops.aten.add.Tensor(arg1523_1, 1e-05);  arg1523_1 = None
        sqrt_629 = torch.ops.aten.sqrt.default(add_1851);  add_1851 = None
        reciprocal_629 = torch.ops.aten.reciprocal.default(sqrt_629);  sqrt_629 = None
        mul_2135 = torch.ops.aten.mul.Tensor(reciprocal_629, 1);  reciprocal_629 = None
        unsqueeze_5094 = torch.ops.aten.unsqueeze.default(arg1522_1, -1);  arg1522_1 = None
        unsqueeze_5095 = torch.ops.aten.unsqueeze.default(unsqueeze_5094, -1);  unsqueeze_5094 = None
        unsqueeze_5096 = torch.ops.aten.unsqueeze.default(mul_2135, -1);  mul_2135 = None
        unsqueeze_5097 = torch.ops.aten.unsqueeze.default(unsqueeze_5096, -1);  unsqueeze_5096 = None
        sub_629 = torch.ops.aten.sub.Tensor(convolution_629, unsqueeze_5095);  convolution_629 = unsqueeze_5095 = None
        mul_2136 = torch.ops.aten.mul.Tensor(sub_629, unsqueeze_5097);  sub_629 = unsqueeze_5097 = None
        unsqueeze_5098 = torch.ops.aten.unsqueeze.default(arg1524_1, -1);  arg1524_1 = None
        unsqueeze_5099 = torch.ops.aten.unsqueeze.default(unsqueeze_5098, -1);  unsqueeze_5098 = None
        mul_2137 = torch.ops.aten.mul.Tensor(mul_2136, unsqueeze_5099);  mul_2136 = unsqueeze_5099 = None
        unsqueeze_5100 = torch.ops.aten.unsqueeze.default(arg1525_1, -1);  arg1525_1 = None
        unsqueeze_5101 = torch.ops.aten.unsqueeze.default(unsqueeze_5100, -1);  unsqueeze_5100 = None
        add_1852 = torch.ops.aten.add.Tensor(mul_2137, unsqueeze_5101);  mul_2137 = unsqueeze_5101 = None
        add_1853 = torch.ops.aten.add.Tensor(add_1850, add_1852);  add_1850 = add_1852 = None
        add_1854 = torch.ops.aten.add.Tensor(add_1853, relu_543);  add_1853 = relu_543 = None
        relu_551 = torch.ops.aten.relu.default(add_1854);  add_1854 = None
        convolution_630 = torch.ops.aten.convolution.default(relu_544, arg1526_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1526_1 = None
        add_1855 = torch.ops.aten.add.Tensor(arg1528_1, 1e-05);  arg1528_1 = None
        sqrt_630 = torch.ops.aten.sqrt.default(add_1855);  add_1855 = None
        reciprocal_630 = torch.ops.aten.reciprocal.default(sqrt_630);  sqrt_630 = None
        mul_2138 = torch.ops.aten.mul.Tensor(reciprocal_630, 1);  reciprocal_630 = None
        unsqueeze_5102 = torch.ops.aten.unsqueeze.default(arg1527_1, -1);  arg1527_1 = None
        unsqueeze_5103 = torch.ops.aten.unsqueeze.default(unsqueeze_5102, -1);  unsqueeze_5102 = None
        unsqueeze_5104 = torch.ops.aten.unsqueeze.default(mul_2138, -1);  mul_2138 = None
        unsqueeze_5105 = torch.ops.aten.unsqueeze.default(unsqueeze_5104, -1);  unsqueeze_5104 = None
        sub_630 = torch.ops.aten.sub.Tensor(convolution_630, unsqueeze_5103);  convolution_630 = unsqueeze_5103 = None
        mul_2139 = torch.ops.aten.mul.Tensor(sub_630, unsqueeze_5105);  sub_630 = unsqueeze_5105 = None
        unsqueeze_5106 = torch.ops.aten.unsqueeze.default(arg1529_1, -1);  arg1529_1 = None
        unsqueeze_5107 = torch.ops.aten.unsqueeze.default(unsqueeze_5106, -1);  unsqueeze_5106 = None
        mul_2140 = torch.ops.aten.mul.Tensor(mul_2139, unsqueeze_5107);  mul_2139 = unsqueeze_5107 = None
        unsqueeze_5108 = torch.ops.aten.unsqueeze.default(arg1530_1, -1);  arg1530_1 = None
        unsqueeze_5109 = torch.ops.aten.unsqueeze.default(unsqueeze_5108, -1);  unsqueeze_5108 = None
        add_1856 = torch.ops.aten.add.Tensor(mul_2140, unsqueeze_5109);  mul_2140 = unsqueeze_5109 = None
        relu_552 = torch.ops.aten.relu.default(add_1856);  add_1856 = None
        convolution_631 = torch.ops.aten.convolution.default(relu_552, arg1531_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_552 = arg1531_1 = None
        add_1857 = torch.ops.aten.add.Tensor(arg1533_1, 1e-05);  arg1533_1 = None
        sqrt_631 = torch.ops.aten.sqrt.default(add_1857);  add_1857 = None
        reciprocal_631 = torch.ops.aten.reciprocal.default(sqrt_631);  sqrt_631 = None
        mul_2141 = torch.ops.aten.mul.Tensor(reciprocal_631, 1);  reciprocal_631 = None
        unsqueeze_5110 = torch.ops.aten.unsqueeze.default(arg1532_1, -1);  arg1532_1 = None
        unsqueeze_5111 = torch.ops.aten.unsqueeze.default(unsqueeze_5110, -1);  unsqueeze_5110 = None
        unsqueeze_5112 = torch.ops.aten.unsqueeze.default(mul_2141, -1);  mul_2141 = None
        unsqueeze_5113 = torch.ops.aten.unsqueeze.default(unsqueeze_5112, -1);  unsqueeze_5112 = None
        sub_631 = torch.ops.aten.sub.Tensor(convolution_631, unsqueeze_5111);  convolution_631 = unsqueeze_5111 = None
        mul_2142 = torch.ops.aten.mul.Tensor(sub_631, unsqueeze_5113);  sub_631 = unsqueeze_5113 = None
        unsqueeze_5114 = torch.ops.aten.unsqueeze.default(arg1534_1, -1);  arg1534_1 = None
        unsqueeze_5115 = torch.ops.aten.unsqueeze.default(unsqueeze_5114, -1);  unsqueeze_5114 = None
        mul_2143 = torch.ops.aten.mul.Tensor(mul_2142, unsqueeze_5115);  mul_2142 = unsqueeze_5115 = None
        unsqueeze_5116 = torch.ops.aten.unsqueeze.default(arg1535_1, -1);  arg1535_1 = None
        unsqueeze_5117 = torch.ops.aten.unsqueeze.default(unsqueeze_5116, -1);  unsqueeze_5116 = None
        add_1858 = torch.ops.aten.add.Tensor(mul_2143, unsqueeze_5117);  mul_2143 = unsqueeze_5117 = None
        relu_553 = torch.ops.aten.relu.default(add_1858);  add_1858 = None
        convolution_632 = torch.ops.aten.convolution.default(relu_553, arg1536_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_553 = arg1536_1 = None
        add_1859 = torch.ops.aten.add.Tensor(arg1538_1, 1e-05);  arg1538_1 = None
        sqrt_632 = torch.ops.aten.sqrt.default(add_1859);  add_1859 = None
        reciprocal_632 = torch.ops.aten.reciprocal.default(sqrt_632);  sqrt_632 = None
        mul_2144 = torch.ops.aten.mul.Tensor(reciprocal_632, 1);  reciprocal_632 = None
        unsqueeze_5118 = torch.ops.aten.unsqueeze.default(arg1537_1, -1);  arg1537_1 = None
        unsqueeze_5119 = torch.ops.aten.unsqueeze.default(unsqueeze_5118, -1);  unsqueeze_5118 = None
        unsqueeze_5120 = torch.ops.aten.unsqueeze.default(mul_2144, -1);  mul_2144 = None
        unsqueeze_5121 = torch.ops.aten.unsqueeze.default(unsqueeze_5120, -1);  unsqueeze_5120 = None
        sub_632 = torch.ops.aten.sub.Tensor(convolution_632, unsqueeze_5119);  convolution_632 = unsqueeze_5119 = None
        mul_2145 = torch.ops.aten.mul.Tensor(sub_632, unsqueeze_5121);  sub_632 = unsqueeze_5121 = None
        unsqueeze_5122 = torch.ops.aten.unsqueeze.default(arg1539_1, -1);  arg1539_1 = None
        unsqueeze_5123 = torch.ops.aten.unsqueeze.default(unsqueeze_5122, -1);  unsqueeze_5122 = None
        mul_2146 = torch.ops.aten.mul.Tensor(mul_2145, unsqueeze_5123);  mul_2145 = unsqueeze_5123 = None
        unsqueeze_5124 = torch.ops.aten.unsqueeze.default(arg1540_1, -1);  arg1540_1 = None
        unsqueeze_5125 = torch.ops.aten.unsqueeze.default(unsqueeze_5124, -1);  unsqueeze_5124 = None
        add_1860 = torch.ops.aten.add.Tensor(mul_2146, unsqueeze_5125);  mul_2146 = unsqueeze_5125 = None
        convolution_633 = torch.ops.aten.convolution.default(relu_544, arg1541_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_544 = arg1541_1 = None
        add_1861 = torch.ops.aten.add.Tensor(arg1543_1, 1e-05);  arg1543_1 = None
        sqrt_633 = torch.ops.aten.sqrt.default(add_1861);  add_1861 = None
        reciprocal_633 = torch.ops.aten.reciprocal.default(sqrt_633);  sqrt_633 = None
        mul_2147 = torch.ops.aten.mul.Tensor(reciprocal_633, 1);  reciprocal_633 = None
        unsqueeze_5126 = torch.ops.aten.unsqueeze.default(arg1542_1, -1);  arg1542_1 = None
        unsqueeze_5127 = torch.ops.aten.unsqueeze.default(unsqueeze_5126, -1);  unsqueeze_5126 = None
        unsqueeze_5128 = torch.ops.aten.unsqueeze.default(mul_2147, -1);  mul_2147 = None
        unsqueeze_5129 = torch.ops.aten.unsqueeze.default(unsqueeze_5128, -1);  unsqueeze_5128 = None
        sub_633 = torch.ops.aten.sub.Tensor(convolution_633, unsqueeze_5127);  convolution_633 = unsqueeze_5127 = None
        mul_2148 = torch.ops.aten.mul.Tensor(sub_633, unsqueeze_5129);  sub_633 = unsqueeze_5129 = None
        unsqueeze_5130 = torch.ops.aten.unsqueeze.default(arg1544_1, -1);  arg1544_1 = None
        unsqueeze_5131 = torch.ops.aten.unsqueeze.default(unsqueeze_5130, -1);  unsqueeze_5130 = None
        mul_2149 = torch.ops.aten.mul.Tensor(mul_2148, unsqueeze_5131);  mul_2148 = unsqueeze_5131 = None
        unsqueeze_5132 = torch.ops.aten.unsqueeze.default(arg1545_1, -1);  arg1545_1 = None
        unsqueeze_5133 = torch.ops.aten.unsqueeze.default(unsqueeze_5132, -1);  unsqueeze_5132 = None
        add_1862 = torch.ops.aten.add.Tensor(mul_2149, unsqueeze_5133);  mul_2149 = unsqueeze_5133 = None
        add_1863 = torch.ops.aten.add.Tensor(add_1860, add_1862);  add_1860 = add_1862 = None
        relu_554 = torch.ops.aten.relu.default(add_1863);  add_1863 = None
        convolution_634 = torch.ops.aten.convolution.default(relu_545, arg1546_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1546_1 = None
        add_1864 = torch.ops.aten.add.Tensor(arg1548_1, 1e-05);  arg1548_1 = None
        sqrt_634 = torch.ops.aten.sqrt.default(add_1864);  add_1864 = None
        reciprocal_634 = torch.ops.aten.reciprocal.default(sqrt_634);  sqrt_634 = None
        mul_2150 = torch.ops.aten.mul.Tensor(reciprocal_634, 1);  reciprocal_634 = None
        unsqueeze_5134 = torch.ops.aten.unsqueeze.default(arg1547_1, -1);  arg1547_1 = None
        unsqueeze_5135 = torch.ops.aten.unsqueeze.default(unsqueeze_5134, -1);  unsqueeze_5134 = None
        unsqueeze_5136 = torch.ops.aten.unsqueeze.default(mul_2150, -1);  mul_2150 = None
        unsqueeze_5137 = torch.ops.aten.unsqueeze.default(unsqueeze_5136, -1);  unsqueeze_5136 = None
        sub_634 = torch.ops.aten.sub.Tensor(convolution_634, unsqueeze_5135);  convolution_634 = unsqueeze_5135 = None
        mul_2151 = torch.ops.aten.mul.Tensor(sub_634, unsqueeze_5137);  sub_634 = unsqueeze_5137 = None
        unsqueeze_5138 = torch.ops.aten.unsqueeze.default(arg1549_1, -1);  arg1549_1 = None
        unsqueeze_5139 = torch.ops.aten.unsqueeze.default(unsqueeze_5138, -1);  unsqueeze_5138 = None
        mul_2152 = torch.ops.aten.mul.Tensor(mul_2151, unsqueeze_5139);  mul_2151 = unsqueeze_5139 = None
        unsqueeze_5140 = torch.ops.aten.unsqueeze.default(arg1550_1, -1);  arg1550_1 = None
        unsqueeze_5141 = torch.ops.aten.unsqueeze.default(unsqueeze_5140, -1);  unsqueeze_5140 = None
        add_1865 = torch.ops.aten.add.Tensor(mul_2152, unsqueeze_5141);  mul_2152 = unsqueeze_5141 = None
        relu_555 = torch.ops.aten.relu.default(add_1865);  add_1865 = None
        convolution_635 = torch.ops.aten.convolution.default(relu_555, arg1551_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_555 = arg1551_1 = None
        add_1866 = torch.ops.aten.add.Tensor(arg1553_1, 1e-05);  arg1553_1 = None
        sqrt_635 = torch.ops.aten.sqrt.default(add_1866);  add_1866 = None
        reciprocal_635 = torch.ops.aten.reciprocal.default(sqrt_635);  sqrt_635 = None
        mul_2153 = torch.ops.aten.mul.Tensor(reciprocal_635, 1);  reciprocal_635 = None
        unsqueeze_5142 = torch.ops.aten.unsqueeze.default(arg1552_1, -1);  arg1552_1 = None
        unsqueeze_5143 = torch.ops.aten.unsqueeze.default(unsqueeze_5142, -1);  unsqueeze_5142 = None
        unsqueeze_5144 = torch.ops.aten.unsqueeze.default(mul_2153, -1);  mul_2153 = None
        unsqueeze_5145 = torch.ops.aten.unsqueeze.default(unsqueeze_5144, -1);  unsqueeze_5144 = None
        sub_635 = torch.ops.aten.sub.Tensor(convolution_635, unsqueeze_5143);  convolution_635 = unsqueeze_5143 = None
        mul_2154 = torch.ops.aten.mul.Tensor(sub_635, unsqueeze_5145);  sub_635 = unsqueeze_5145 = None
        unsqueeze_5146 = torch.ops.aten.unsqueeze.default(arg1554_1, -1);  arg1554_1 = None
        unsqueeze_5147 = torch.ops.aten.unsqueeze.default(unsqueeze_5146, -1);  unsqueeze_5146 = None
        mul_2155 = torch.ops.aten.mul.Tensor(mul_2154, unsqueeze_5147);  mul_2154 = unsqueeze_5147 = None
        unsqueeze_5148 = torch.ops.aten.unsqueeze.default(arg1555_1, -1);  arg1555_1 = None
        unsqueeze_5149 = torch.ops.aten.unsqueeze.default(unsqueeze_5148, -1);  unsqueeze_5148 = None
        add_1867 = torch.ops.aten.add.Tensor(mul_2155, unsqueeze_5149);  mul_2155 = unsqueeze_5149 = None
        relu_556 = torch.ops.aten.relu.default(add_1867);  add_1867 = None
        convolution_636 = torch.ops.aten.convolution.default(relu_556, arg1556_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_556 = arg1556_1 = None
        add_1868 = torch.ops.aten.add.Tensor(arg1558_1, 1e-05);  arg1558_1 = None
        sqrt_636 = torch.ops.aten.sqrt.default(add_1868);  add_1868 = None
        reciprocal_636 = torch.ops.aten.reciprocal.default(sqrt_636);  sqrt_636 = None
        mul_2156 = torch.ops.aten.mul.Tensor(reciprocal_636, 1);  reciprocal_636 = None
        unsqueeze_5150 = torch.ops.aten.unsqueeze.default(arg1557_1, -1);  arg1557_1 = None
        unsqueeze_5151 = torch.ops.aten.unsqueeze.default(unsqueeze_5150, -1);  unsqueeze_5150 = None
        unsqueeze_5152 = torch.ops.aten.unsqueeze.default(mul_2156, -1);  mul_2156 = None
        unsqueeze_5153 = torch.ops.aten.unsqueeze.default(unsqueeze_5152, -1);  unsqueeze_5152 = None
        sub_636 = torch.ops.aten.sub.Tensor(convolution_636, unsqueeze_5151);  convolution_636 = unsqueeze_5151 = None
        mul_2157 = torch.ops.aten.mul.Tensor(sub_636, unsqueeze_5153);  sub_636 = unsqueeze_5153 = None
        unsqueeze_5154 = torch.ops.aten.unsqueeze.default(arg1559_1, -1);  arg1559_1 = None
        unsqueeze_5155 = torch.ops.aten.unsqueeze.default(unsqueeze_5154, -1);  unsqueeze_5154 = None
        mul_2158 = torch.ops.aten.mul.Tensor(mul_2157, unsqueeze_5155);  mul_2157 = unsqueeze_5155 = None
        unsqueeze_5156 = torch.ops.aten.unsqueeze.default(arg1560_1, -1);  arg1560_1 = None
        unsqueeze_5157 = torch.ops.aten.unsqueeze.default(unsqueeze_5156, -1);  unsqueeze_5156 = None
        add_1869 = torch.ops.aten.add.Tensor(mul_2158, unsqueeze_5157);  mul_2158 = unsqueeze_5157 = None
        convolution_637 = torch.ops.aten.convolution.default(relu_545, arg1561_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_545 = arg1561_1 = None
        add_1870 = torch.ops.aten.add.Tensor(arg1563_1, 1e-05);  arg1563_1 = None
        sqrt_637 = torch.ops.aten.sqrt.default(add_1870);  add_1870 = None
        reciprocal_637 = torch.ops.aten.reciprocal.default(sqrt_637);  sqrt_637 = None
        mul_2159 = torch.ops.aten.mul.Tensor(reciprocal_637, 1);  reciprocal_637 = None
        unsqueeze_5158 = torch.ops.aten.unsqueeze.default(arg1562_1, -1);  arg1562_1 = None
        unsqueeze_5159 = torch.ops.aten.unsqueeze.default(unsqueeze_5158, -1);  unsqueeze_5158 = None
        unsqueeze_5160 = torch.ops.aten.unsqueeze.default(mul_2159, -1);  mul_2159 = None
        unsqueeze_5161 = torch.ops.aten.unsqueeze.default(unsqueeze_5160, -1);  unsqueeze_5160 = None
        sub_637 = torch.ops.aten.sub.Tensor(convolution_637, unsqueeze_5159);  convolution_637 = unsqueeze_5159 = None
        mul_2160 = torch.ops.aten.mul.Tensor(sub_637, unsqueeze_5161);  sub_637 = unsqueeze_5161 = None
        unsqueeze_5162 = torch.ops.aten.unsqueeze.default(arg1564_1, -1);  arg1564_1 = None
        unsqueeze_5163 = torch.ops.aten.unsqueeze.default(unsqueeze_5162, -1);  unsqueeze_5162 = None
        mul_2161 = torch.ops.aten.mul.Tensor(mul_2160, unsqueeze_5163);  mul_2160 = unsqueeze_5163 = None
        unsqueeze_5164 = torch.ops.aten.unsqueeze.default(arg1565_1, -1);  arg1565_1 = None
        unsqueeze_5165 = torch.ops.aten.unsqueeze.default(unsqueeze_5164, -1);  unsqueeze_5164 = None
        add_1871 = torch.ops.aten.add.Tensor(mul_2161, unsqueeze_5165);  mul_2161 = unsqueeze_5165 = None
        add_1872 = torch.ops.aten.add.Tensor(add_1869, add_1871);  add_1869 = add_1871 = None
        relu_557 = torch.ops.aten.relu.default(add_1872);  add_1872 = None
        convolution_638 = torch.ops.aten.convolution.default(relu_554, arg1566_1, arg1567_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_554 = arg1566_1 = arg1567_1 = None
        add_1873 = torch.ops.aten.add.Tensor(arg1569_1, 1e-05);  arg1569_1 = None
        sqrt_638 = torch.ops.aten.sqrt.default(add_1873);  add_1873 = None
        reciprocal_638 = torch.ops.aten.reciprocal.default(sqrt_638);  sqrt_638 = None
        mul_2162 = torch.ops.aten.mul.Tensor(reciprocal_638, 1);  reciprocal_638 = None
        unsqueeze_5166 = torch.ops.aten.unsqueeze.default(arg1568_1, -1);  arg1568_1 = None
        unsqueeze_5167 = torch.ops.aten.unsqueeze.default(unsqueeze_5166, -1);  unsqueeze_5166 = None
        unsqueeze_5168 = torch.ops.aten.unsqueeze.default(mul_2162, -1);  mul_2162 = None
        unsqueeze_5169 = torch.ops.aten.unsqueeze.default(unsqueeze_5168, -1);  unsqueeze_5168 = None
        sub_638 = torch.ops.aten.sub.Tensor(convolution_638, unsqueeze_5167);  convolution_638 = unsqueeze_5167 = None
        mul_2163 = torch.ops.aten.mul.Tensor(sub_638, unsqueeze_5169);  sub_638 = unsqueeze_5169 = None
        unsqueeze_5170 = torch.ops.aten.unsqueeze.default(arg1570_1, -1);  arg1570_1 = None
        unsqueeze_5171 = torch.ops.aten.unsqueeze.default(unsqueeze_5170, -1);  unsqueeze_5170 = None
        mul_2164 = torch.ops.aten.mul.Tensor(mul_2163, unsqueeze_5171);  mul_2163 = unsqueeze_5171 = None
        unsqueeze_5172 = torch.ops.aten.unsqueeze.default(arg1571_1, -1);  arg1571_1 = None
        unsqueeze_5173 = torch.ops.aten.unsqueeze.default(unsqueeze_5172, -1);  unsqueeze_5172 = None
        add_1874 = torch.ops.aten.add.Tensor(mul_2164, unsqueeze_5173);  mul_2164 = unsqueeze_5173 = None
        relu_558 = torch.ops.aten.relu.default(add_1874);  add_1874 = None
        add_1875 = torch.ops.aten.add.Tensor(relu_557, relu_558);  relu_557 = relu_558 = None
        convolution_639 = torch.ops.aten.convolution.default(relu_547, arg1572_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1572_1 = None
        add_1876 = torch.ops.aten.add.Tensor(arg1574_1, 1e-05);  arg1574_1 = None
        sqrt_639 = torch.ops.aten.sqrt.default(add_1876);  add_1876 = None
        reciprocal_639 = torch.ops.aten.reciprocal.default(sqrt_639);  sqrt_639 = None
        mul_2165 = torch.ops.aten.mul.Tensor(reciprocal_639, 1);  reciprocal_639 = None
        unsqueeze_5174 = torch.ops.aten.unsqueeze.default(arg1573_1, -1);  arg1573_1 = None
        unsqueeze_5175 = torch.ops.aten.unsqueeze.default(unsqueeze_5174, -1);  unsqueeze_5174 = None
        unsqueeze_5176 = torch.ops.aten.unsqueeze.default(mul_2165, -1);  mul_2165 = None
        unsqueeze_5177 = torch.ops.aten.unsqueeze.default(unsqueeze_5176, -1);  unsqueeze_5176 = None
        sub_639 = torch.ops.aten.sub.Tensor(convolution_639, unsqueeze_5175);  convolution_639 = unsqueeze_5175 = None
        mul_2166 = torch.ops.aten.mul.Tensor(sub_639, unsqueeze_5177);  sub_639 = unsqueeze_5177 = None
        unsqueeze_5178 = torch.ops.aten.unsqueeze.default(arg1575_1, -1);  arg1575_1 = None
        unsqueeze_5179 = torch.ops.aten.unsqueeze.default(unsqueeze_5178, -1);  unsqueeze_5178 = None
        mul_2167 = torch.ops.aten.mul.Tensor(mul_2166, unsqueeze_5179);  mul_2166 = unsqueeze_5179 = None
        unsqueeze_5180 = torch.ops.aten.unsqueeze.default(arg1576_1, -1);  arg1576_1 = None
        unsqueeze_5181 = torch.ops.aten.unsqueeze.default(unsqueeze_5180, -1);  unsqueeze_5180 = None
        add_1877 = torch.ops.aten.add.Tensor(mul_2167, unsqueeze_5181);  mul_2167 = unsqueeze_5181 = None
        relu_559 = torch.ops.aten.relu.default(add_1877);  add_1877 = None
        convolution_640 = torch.ops.aten.convolution.default(relu_559, arg1577_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_559 = arg1577_1 = None
        add_1878 = torch.ops.aten.add.Tensor(arg1579_1, 1e-05);  arg1579_1 = None
        sqrt_640 = torch.ops.aten.sqrt.default(add_1878);  add_1878 = None
        reciprocal_640 = torch.ops.aten.reciprocal.default(sqrt_640);  sqrt_640 = None
        mul_2168 = torch.ops.aten.mul.Tensor(reciprocal_640, 1);  reciprocal_640 = None
        unsqueeze_5182 = torch.ops.aten.unsqueeze.default(arg1578_1, -1);  arg1578_1 = None
        unsqueeze_5183 = torch.ops.aten.unsqueeze.default(unsqueeze_5182, -1);  unsqueeze_5182 = None
        unsqueeze_5184 = torch.ops.aten.unsqueeze.default(mul_2168, -1);  mul_2168 = None
        unsqueeze_5185 = torch.ops.aten.unsqueeze.default(unsqueeze_5184, -1);  unsqueeze_5184 = None
        sub_640 = torch.ops.aten.sub.Tensor(convolution_640, unsqueeze_5183);  convolution_640 = unsqueeze_5183 = None
        mul_2169 = torch.ops.aten.mul.Tensor(sub_640, unsqueeze_5185);  sub_640 = unsqueeze_5185 = None
        unsqueeze_5186 = torch.ops.aten.unsqueeze.default(arg1580_1, -1);  arg1580_1 = None
        unsqueeze_5187 = torch.ops.aten.unsqueeze.default(unsqueeze_5186, -1);  unsqueeze_5186 = None
        mul_2170 = torch.ops.aten.mul.Tensor(mul_2169, unsqueeze_5187);  mul_2169 = unsqueeze_5187 = None
        unsqueeze_5188 = torch.ops.aten.unsqueeze.default(arg1581_1, -1);  arg1581_1 = None
        unsqueeze_5189 = torch.ops.aten.unsqueeze.default(unsqueeze_5188, -1);  unsqueeze_5188 = None
        add_1879 = torch.ops.aten.add.Tensor(mul_2170, unsqueeze_5189);  mul_2170 = unsqueeze_5189 = None
        relu_560 = torch.ops.aten.relu.default(add_1879);  add_1879 = None
        convolution_641 = torch.ops.aten.convolution.default(relu_560, arg1582_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_560 = arg1582_1 = None
        add_1880 = torch.ops.aten.add.Tensor(arg1584_1, 1e-05);  arg1584_1 = None
        sqrt_641 = torch.ops.aten.sqrt.default(add_1880);  add_1880 = None
        reciprocal_641 = torch.ops.aten.reciprocal.default(sqrt_641);  sqrt_641 = None
        mul_2171 = torch.ops.aten.mul.Tensor(reciprocal_641, 1);  reciprocal_641 = None
        unsqueeze_5190 = torch.ops.aten.unsqueeze.default(arg1583_1, -1);  arg1583_1 = None
        unsqueeze_5191 = torch.ops.aten.unsqueeze.default(unsqueeze_5190, -1);  unsqueeze_5190 = None
        unsqueeze_5192 = torch.ops.aten.unsqueeze.default(mul_2171, -1);  mul_2171 = None
        unsqueeze_5193 = torch.ops.aten.unsqueeze.default(unsqueeze_5192, -1);  unsqueeze_5192 = None
        sub_641 = torch.ops.aten.sub.Tensor(convolution_641, unsqueeze_5191);  convolution_641 = unsqueeze_5191 = None
        mul_2172 = torch.ops.aten.mul.Tensor(sub_641, unsqueeze_5193);  sub_641 = unsqueeze_5193 = None
        unsqueeze_5194 = torch.ops.aten.unsqueeze.default(arg1585_1, -1);  arg1585_1 = None
        unsqueeze_5195 = torch.ops.aten.unsqueeze.default(unsqueeze_5194, -1);  unsqueeze_5194 = None
        mul_2173 = torch.ops.aten.mul.Tensor(mul_2172, unsqueeze_5195);  mul_2172 = unsqueeze_5195 = None
        unsqueeze_5196 = torch.ops.aten.unsqueeze.default(arg1586_1, -1);  arg1586_1 = None
        unsqueeze_5197 = torch.ops.aten.unsqueeze.default(unsqueeze_5196, -1);  unsqueeze_5196 = None
        add_1881 = torch.ops.aten.add.Tensor(mul_2173, unsqueeze_5197);  mul_2173 = unsqueeze_5197 = None
        convolution_642 = torch.ops.aten.convolution.default(relu_547, arg1587_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_547 = arg1587_1 = None
        add_1882 = torch.ops.aten.add.Tensor(arg1589_1, 1e-05);  arg1589_1 = None
        sqrt_642 = torch.ops.aten.sqrt.default(add_1882);  add_1882 = None
        reciprocal_642 = torch.ops.aten.reciprocal.default(sqrt_642);  sqrt_642 = None
        mul_2174 = torch.ops.aten.mul.Tensor(reciprocal_642, 1);  reciprocal_642 = None
        unsqueeze_5198 = torch.ops.aten.unsqueeze.default(arg1588_1, -1);  arg1588_1 = None
        unsqueeze_5199 = torch.ops.aten.unsqueeze.default(unsqueeze_5198, -1);  unsqueeze_5198 = None
        unsqueeze_5200 = torch.ops.aten.unsqueeze.default(mul_2174, -1);  mul_2174 = None
        unsqueeze_5201 = torch.ops.aten.unsqueeze.default(unsqueeze_5200, -1);  unsqueeze_5200 = None
        sub_642 = torch.ops.aten.sub.Tensor(convolution_642, unsqueeze_5199);  convolution_642 = unsqueeze_5199 = None
        mul_2175 = torch.ops.aten.mul.Tensor(sub_642, unsqueeze_5201);  sub_642 = unsqueeze_5201 = None
        unsqueeze_5202 = torch.ops.aten.unsqueeze.default(arg1590_1, -1);  arg1590_1 = None
        unsqueeze_5203 = torch.ops.aten.unsqueeze.default(unsqueeze_5202, -1);  unsqueeze_5202 = None
        mul_2176 = torch.ops.aten.mul.Tensor(mul_2175, unsqueeze_5203);  mul_2175 = unsqueeze_5203 = None
        unsqueeze_5204 = torch.ops.aten.unsqueeze.default(arg1591_1, -1);  arg1591_1 = None
        unsqueeze_5205 = torch.ops.aten.unsqueeze.default(unsqueeze_5204, -1);  unsqueeze_5204 = None
        add_1883 = torch.ops.aten.add.Tensor(mul_2176, unsqueeze_5205);  mul_2176 = unsqueeze_5205 = None
        add_1884 = torch.ops.aten.add.Tensor(add_1881, add_1883);  add_1881 = add_1883 = None
        relu_561 = torch.ops.aten.relu.default(add_1884);  add_1884 = None
        convolution_643 = torch.ops.aten.convolution.default(add_1875, arg1592_1, arg1593_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  add_1875 = arg1592_1 = arg1593_1 = None
        add_1885 = torch.ops.aten.add.Tensor(arg1595_1, 1e-05);  arg1595_1 = None
        sqrt_643 = torch.ops.aten.sqrt.default(add_1885);  add_1885 = None
        reciprocal_643 = torch.ops.aten.reciprocal.default(sqrt_643);  sqrt_643 = None
        mul_2177 = torch.ops.aten.mul.Tensor(reciprocal_643, 1);  reciprocal_643 = None
        unsqueeze_5206 = torch.ops.aten.unsqueeze.default(arg1594_1, -1);  arg1594_1 = None
        unsqueeze_5207 = torch.ops.aten.unsqueeze.default(unsqueeze_5206, -1);  unsqueeze_5206 = None
        unsqueeze_5208 = torch.ops.aten.unsqueeze.default(mul_2177, -1);  mul_2177 = None
        unsqueeze_5209 = torch.ops.aten.unsqueeze.default(unsqueeze_5208, -1);  unsqueeze_5208 = None
        sub_643 = torch.ops.aten.sub.Tensor(convolution_643, unsqueeze_5207);  convolution_643 = unsqueeze_5207 = None
        mul_2178 = torch.ops.aten.mul.Tensor(sub_643, unsqueeze_5209);  sub_643 = unsqueeze_5209 = None
        unsqueeze_5210 = torch.ops.aten.unsqueeze.default(arg1596_1, -1);  arg1596_1 = None
        unsqueeze_5211 = torch.ops.aten.unsqueeze.default(unsqueeze_5210, -1);  unsqueeze_5210 = None
        mul_2179 = torch.ops.aten.mul.Tensor(mul_2178, unsqueeze_5211);  mul_2178 = unsqueeze_5211 = None
        unsqueeze_5212 = torch.ops.aten.unsqueeze.default(arg1597_1, -1);  arg1597_1 = None
        unsqueeze_5213 = torch.ops.aten.unsqueeze.default(unsqueeze_5212, -1);  unsqueeze_5212 = None
        add_1886 = torch.ops.aten.add.Tensor(mul_2179, unsqueeze_5213);  mul_2179 = unsqueeze_5213 = None
        relu_562 = torch.ops.aten.relu.default(add_1886);  add_1886 = None
        add_1887 = torch.ops.aten.add.Tensor(relu_561, relu_562);  relu_561 = relu_562 = None
        convolution_644 = torch.ops.aten.convolution.default(relu_551, arg1598_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1598_1 = None
        add_1888 = torch.ops.aten.add.Tensor(arg1600_1, 1e-05);  arg1600_1 = None
        sqrt_644 = torch.ops.aten.sqrt.default(add_1888);  add_1888 = None
        reciprocal_644 = torch.ops.aten.reciprocal.default(sqrt_644);  sqrt_644 = None
        mul_2180 = torch.ops.aten.mul.Tensor(reciprocal_644, 1);  reciprocal_644 = None
        unsqueeze_5214 = torch.ops.aten.unsqueeze.default(arg1599_1, -1);  arg1599_1 = None
        unsqueeze_5215 = torch.ops.aten.unsqueeze.default(unsqueeze_5214, -1);  unsqueeze_5214 = None
        unsqueeze_5216 = torch.ops.aten.unsqueeze.default(mul_2180, -1);  mul_2180 = None
        unsqueeze_5217 = torch.ops.aten.unsqueeze.default(unsqueeze_5216, -1);  unsqueeze_5216 = None
        sub_644 = torch.ops.aten.sub.Tensor(convolution_644, unsqueeze_5215);  convolution_644 = unsqueeze_5215 = None
        mul_2181 = torch.ops.aten.mul.Tensor(sub_644, unsqueeze_5217);  sub_644 = unsqueeze_5217 = None
        unsqueeze_5218 = torch.ops.aten.unsqueeze.default(arg1601_1, -1);  arg1601_1 = None
        unsqueeze_5219 = torch.ops.aten.unsqueeze.default(unsqueeze_5218, -1);  unsqueeze_5218 = None
        mul_2182 = torch.ops.aten.mul.Tensor(mul_2181, unsqueeze_5219);  mul_2181 = unsqueeze_5219 = None
        unsqueeze_5220 = torch.ops.aten.unsqueeze.default(arg1602_1, -1);  arg1602_1 = None
        unsqueeze_5221 = torch.ops.aten.unsqueeze.default(unsqueeze_5220, -1);  unsqueeze_5220 = None
        add_1889 = torch.ops.aten.add.Tensor(mul_2182, unsqueeze_5221);  mul_2182 = unsqueeze_5221 = None
        relu_563 = torch.ops.aten.relu.default(add_1889);  add_1889 = None
        convolution_645 = torch.ops.aten.convolution.default(relu_563, arg1603_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_563 = arg1603_1 = None
        add_1890 = torch.ops.aten.add.Tensor(arg1605_1, 1e-05);  arg1605_1 = None
        sqrt_645 = torch.ops.aten.sqrt.default(add_1890);  add_1890 = None
        reciprocal_645 = torch.ops.aten.reciprocal.default(sqrt_645);  sqrt_645 = None
        mul_2183 = torch.ops.aten.mul.Tensor(reciprocal_645, 1);  reciprocal_645 = None
        unsqueeze_5222 = torch.ops.aten.unsqueeze.default(arg1604_1, -1);  arg1604_1 = None
        unsqueeze_5223 = torch.ops.aten.unsqueeze.default(unsqueeze_5222, -1);  unsqueeze_5222 = None
        unsqueeze_5224 = torch.ops.aten.unsqueeze.default(mul_2183, -1);  mul_2183 = None
        unsqueeze_5225 = torch.ops.aten.unsqueeze.default(unsqueeze_5224, -1);  unsqueeze_5224 = None
        sub_645 = torch.ops.aten.sub.Tensor(convolution_645, unsqueeze_5223);  convolution_645 = unsqueeze_5223 = None
        mul_2184 = torch.ops.aten.mul.Tensor(sub_645, unsqueeze_5225);  sub_645 = unsqueeze_5225 = None
        unsqueeze_5226 = torch.ops.aten.unsqueeze.default(arg1606_1, -1);  arg1606_1 = None
        unsqueeze_5227 = torch.ops.aten.unsqueeze.default(unsqueeze_5226, -1);  unsqueeze_5226 = None
        mul_2185 = torch.ops.aten.mul.Tensor(mul_2184, unsqueeze_5227);  mul_2184 = unsqueeze_5227 = None
        unsqueeze_5228 = torch.ops.aten.unsqueeze.default(arg1607_1, -1);  arg1607_1 = None
        unsqueeze_5229 = torch.ops.aten.unsqueeze.default(unsqueeze_5228, -1);  unsqueeze_5228 = None
        add_1891 = torch.ops.aten.add.Tensor(mul_2185, unsqueeze_5229);  mul_2185 = unsqueeze_5229 = None
        relu_564 = torch.ops.aten.relu.default(add_1891);  add_1891 = None
        convolution_646 = torch.ops.aten.convolution.default(relu_564, arg1608_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_564 = arg1608_1 = None
        add_1892 = torch.ops.aten.add.Tensor(arg1610_1, 1e-05);  arg1610_1 = None
        sqrt_646 = torch.ops.aten.sqrt.default(add_1892);  add_1892 = None
        reciprocal_646 = torch.ops.aten.reciprocal.default(sqrt_646);  sqrt_646 = None
        mul_2186 = torch.ops.aten.mul.Tensor(reciprocal_646, 1);  reciprocal_646 = None
        unsqueeze_5230 = torch.ops.aten.unsqueeze.default(arg1609_1, -1);  arg1609_1 = None
        unsqueeze_5231 = torch.ops.aten.unsqueeze.default(unsqueeze_5230, -1);  unsqueeze_5230 = None
        unsqueeze_5232 = torch.ops.aten.unsqueeze.default(mul_2186, -1);  mul_2186 = None
        unsqueeze_5233 = torch.ops.aten.unsqueeze.default(unsqueeze_5232, -1);  unsqueeze_5232 = None
        sub_646 = torch.ops.aten.sub.Tensor(convolution_646, unsqueeze_5231);  convolution_646 = unsqueeze_5231 = None
        mul_2187 = torch.ops.aten.mul.Tensor(sub_646, unsqueeze_5233);  sub_646 = unsqueeze_5233 = None
        unsqueeze_5234 = torch.ops.aten.unsqueeze.default(arg1611_1, -1);  arg1611_1 = None
        unsqueeze_5235 = torch.ops.aten.unsqueeze.default(unsqueeze_5234, -1);  unsqueeze_5234 = None
        mul_2188 = torch.ops.aten.mul.Tensor(mul_2187, unsqueeze_5235);  mul_2187 = unsqueeze_5235 = None
        unsqueeze_5236 = torch.ops.aten.unsqueeze.default(arg1612_1, -1);  arg1612_1 = None
        unsqueeze_5237 = torch.ops.aten.unsqueeze.default(unsqueeze_5236, -1);  unsqueeze_5236 = None
        add_1893 = torch.ops.aten.add.Tensor(mul_2188, unsqueeze_5237);  mul_2188 = unsqueeze_5237 = None
        convolution_647 = torch.ops.aten.convolution.default(relu_551, arg1613_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_551 = arg1613_1 = None
        add_1894 = torch.ops.aten.add.Tensor(arg1615_1, 1e-05);  arg1615_1 = None
        sqrt_647 = torch.ops.aten.sqrt.default(add_1894);  add_1894 = None
        reciprocal_647 = torch.ops.aten.reciprocal.default(sqrt_647);  sqrt_647 = None
        mul_2189 = torch.ops.aten.mul.Tensor(reciprocal_647, 1);  reciprocal_647 = None
        unsqueeze_5238 = torch.ops.aten.unsqueeze.default(arg1614_1, -1);  arg1614_1 = None
        unsqueeze_5239 = torch.ops.aten.unsqueeze.default(unsqueeze_5238, -1);  unsqueeze_5238 = None
        unsqueeze_5240 = torch.ops.aten.unsqueeze.default(mul_2189, -1);  mul_2189 = None
        unsqueeze_5241 = torch.ops.aten.unsqueeze.default(unsqueeze_5240, -1);  unsqueeze_5240 = None
        sub_647 = torch.ops.aten.sub.Tensor(convolution_647, unsqueeze_5239);  convolution_647 = unsqueeze_5239 = None
        mul_2190 = torch.ops.aten.mul.Tensor(sub_647, unsqueeze_5241);  sub_647 = unsqueeze_5241 = None
        unsqueeze_5242 = torch.ops.aten.unsqueeze.default(arg1616_1, -1);  arg1616_1 = None
        unsqueeze_5243 = torch.ops.aten.unsqueeze.default(unsqueeze_5242, -1);  unsqueeze_5242 = None
        mul_2191 = torch.ops.aten.mul.Tensor(mul_2190, unsqueeze_5243);  mul_2190 = unsqueeze_5243 = None
        unsqueeze_5244 = torch.ops.aten.unsqueeze.default(arg1617_1, -1);  arg1617_1 = None
        unsqueeze_5245 = torch.ops.aten.unsqueeze.default(unsqueeze_5244, -1);  unsqueeze_5244 = None
        add_1895 = torch.ops.aten.add.Tensor(mul_2191, unsqueeze_5245);  mul_2191 = unsqueeze_5245 = None
        add_1896 = torch.ops.aten.add.Tensor(add_1893, add_1895);  add_1893 = add_1895 = None
        relu_565 = torch.ops.aten.relu.default(add_1896);  add_1896 = None
        convolution_648 = torch.ops.aten.convolution.default(add_1887, arg1618_1, arg1619_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  add_1887 = arg1618_1 = arg1619_1 = None
        add_1897 = torch.ops.aten.add.Tensor(arg1621_1, 1e-05);  arg1621_1 = None
        sqrt_648 = torch.ops.aten.sqrt.default(add_1897);  add_1897 = None
        reciprocal_648 = torch.ops.aten.reciprocal.default(sqrt_648);  sqrt_648 = None
        mul_2192 = torch.ops.aten.mul.Tensor(reciprocal_648, 1);  reciprocal_648 = None
        unsqueeze_5246 = torch.ops.aten.unsqueeze.default(arg1620_1, -1);  arg1620_1 = None
        unsqueeze_5247 = torch.ops.aten.unsqueeze.default(unsqueeze_5246, -1);  unsqueeze_5246 = None
        unsqueeze_5248 = torch.ops.aten.unsqueeze.default(mul_2192, -1);  mul_2192 = None
        unsqueeze_5249 = torch.ops.aten.unsqueeze.default(unsqueeze_5248, -1);  unsqueeze_5248 = None
        sub_648 = torch.ops.aten.sub.Tensor(convolution_648, unsqueeze_5247);  convolution_648 = unsqueeze_5247 = None
        mul_2193 = torch.ops.aten.mul.Tensor(sub_648, unsqueeze_5249);  sub_648 = unsqueeze_5249 = None
        unsqueeze_5250 = torch.ops.aten.unsqueeze.default(arg1622_1, -1);  arg1622_1 = None
        unsqueeze_5251 = torch.ops.aten.unsqueeze.default(unsqueeze_5250, -1);  unsqueeze_5250 = None
        mul_2194 = torch.ops.aten.mul.Tensor(mul_2193, unsqueeze_5251);  mul_2193 = unsqueeze_5251 = None
        unsqueeze_5252 = torch.ops.aten.unsqueeze.default(arg1623_1, -1);  arg1623_1 = None
        unsqueeze_5253 = torch.ops.aten.unsqueeze.default(unsqueeze_5252, -1);  unsqueeze_5252 = None
        add_1898 = torch.ops.aten.add.Tensor(mul_2194, unsqueeze_5253);  mul_2194 = unsqueeze_5253 = None
        relu_566 = torch.ops.aten.relu.default(add_1898);  add_1898 = None
        add_1899 = torch.ops.aten.add.Tensor(relu_565, relu_566);  relu_565 = relu_566 = None
        convolution_649 = torch.ops.aten.convolution.default(add_1899, arg1624_1, arg1625_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_1899 = arg1624_1 = arg1625_1 = None
        add_1900 = torch.ops.aten.add.Tensor(arg1627_1, 1e-05);  arg1627_1 = None
        sqrt_649 = torch.ops.aten.sqrt.default(add_1900);  add_1900 = None
        reciprocal_649 = torch.ops.aten.reciprocal.default(sqrt_649);  sqrt_649 = None
        mul_2195 = torch.ops.aten.mul.Tensor(reciprocal_649, 1);  reciprocal_649 = None
        unsqueeze_5254 = torch.ops.aten.unsqueeze.default(arg1626_1, -1);  arg1626_1 = None
        unsqueeze_5255 = torch.ops.aten.unsqueeze.default(unsqueeze_5254, -1);  unsqueeze_5254 = None
        unsqueeze_5256 = torch.ops.aten.unsqueeze.default(mul_2195, -1);  mul_2195 = None
        unsqueeze_5257 = torch.ops.aten.unsqueeze.default(unsqueeze_5256, -1);  unsqueeze_5256 = None
        sub_649 = torch.ops.aten.sub.Tensor(convolution_649, unsqueeze_5255);  convolution_649 = unsqueeze_5255 = None
        mul_2196 = torch.ops.aten.mul.Tensor(sub_649, unsqueeze_5257);  sub_649 = unsqueeze_5257 = None
        unsqueeze_5258 = torch.ops.aten.unsqueeze.default(arg1628_1, -1);  arg1628_1 = None
        unsqueeze_5259 = torch.ops.aten.unsqueeze.default(unsqueeze_5258, -1);  unsqueeze_5258 = None
        mul_2197 = torch.ops.aten.mul.Tensor(mul_2196, unsqueeze_5259);  mul_2196 = unsqueeze_5259 = None
        unsqueeze_5260 = torch.ops.aten.unsqueeze.default(arg1629_1, -1);  arg1629_1 = None
        unsqueeze_5261 = torch.ops.aten.unsqueeze.default(unsqueeze_5260, -1);  unsqueeze_5260 = None
        add_1901 = torch.ops.aten.add.Tensor(mul_2197, unsqueeze_5261);  mul_2197 = unsqueeze_5261 = None
        relu_567 = torch.ops.aten.relu.default(add_1901);  add_1901 = None
        mean_1 = torch.ops.aten.mean.dim(relu_567, [-1, -2], True);  relu_567 = None
        view_1 = torch.ops.aten.view.default(mean_1, [8, 2048]);  mean_1 = None
        permute_1 = torch.ops.aten.permute.default(arg1630_1, [1, 0]);  arg1630_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg1631_1, view_1, permute_1);  arg1631_1 = view_1 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 224, 224), is_leaf=True)  # arg1_1
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
    buf11 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 64, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf16, (64, 64, 3, 3), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf18, (64,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf19, (64,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf20, (64,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 64, 1, 1), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf26, (256, 64, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64, 256, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf32, (64,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf33, (64,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf34, (64,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf35, (64,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf36, (64, 64, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf37, (64,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf38, (64,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf39, (64,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf40, (64,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256, 64, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64, 256, 1, 1), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf50, (64,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf51, (64, 64, 3, 3), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf52, (64,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf53, (64,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf54, (64,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf55, (64,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256, 64, 1, 1), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf61, (64, 256, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf62, (64,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf63, (64,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf64, (64,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf65, (64,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf66, (64, 64, 3, 3), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf67, (64,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf68, (64,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf69, (64,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf70, (64,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf71, (256, 64, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf72, (256,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf74, (256,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf75, (256,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf76, (18, 256, 3, 3), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf77, (18,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf78, (18,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf79, (18,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf80, (18,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 331776, device=device(type='cuda', index=0))
    reader.tensor(buf81, (36, 256, 3, 3), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf82, (36,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf83, (36,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf84, (36,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (36,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf86, (18, 18, 3, 3), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf87, (18,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf88, (18,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf89, (18,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf90, (18,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf91, (18, 18, 3, 3), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf92, (18,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf93, (18,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf94, (18,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf95, (18,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf96, (18, 18, 3, 3), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf97, (18,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf98, (18,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf99, (18,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf100, (18,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf101, (18, 18, 3, 3), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf102, (18,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf103, (18,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf104, (18,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf105, (18,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf106, (18, 18, 3, 3), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf107, (18,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf108, (18,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf109, (18,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf110, (18,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf111, (18, 18, 3, 3), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf112, (18,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf113, (18,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf114, (18,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf115, (18,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf116, (18, 18, 3, 3), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf117, (18,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf118, (18,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf119, (18,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf120, (18,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf121, (18, 18, 3, 3), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf122, (18,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf123, (18,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf124, (18,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf125, (18,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf126, (36, 36, 3, 3), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf127, (36,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf128, (36,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf129, (36,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf130, (36,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf131, (36, 36, 3, 3), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf132, (36,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf133, (36,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf134, (36,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf135, (36,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf136, (36, 36, 3, 3), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf137, (36,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf138, (36,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf139, (36,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf140, (36,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf141, (36, 36, 3, 3), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf142, (36,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf143, (36,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf144, (36,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf145, (36,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf146, (36, 36, 3, 3), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf147, (36,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf148, (36,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf149, (36,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf150, (36,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf151, (36, 36, 3, 3), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf152, (36,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf153, (36,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf154, (36,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf155, (36,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf156, (36, 36, 3, 3), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf157, (36,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf158, (36,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf159, (36,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf160, (36,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf161, (36, 36, 3, 3), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf162, (36,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf163, (36,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf164, (36,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf165, (36,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf166, (18, 36, 1, 1), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf167, (18,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf168, (18,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf169, (18,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf170, (18,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 23328, device=device(type='cuda', index=0))
    reader.tensor(buf171, (36, 18, 3, 3), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf172, (36,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf173, (36,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf174, (36,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf175, (36,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf176, (72, 36, 3, 3), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf177, (72,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf178, (72,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf179, (72,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf180, (72,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf181, (18, 18, 3, 3), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf182, (18,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf183, (18,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf184, (18,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf185, (18,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf186, (18, 18, 3, 3), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf187, (18,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf188, (18,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf189, (18,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf190, (18,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf191, (18, 18, 3, 3), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf192, (18,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf193, (18,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf194, (18,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf195, (18,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf196, (18, 18, 3, 3), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf197, (18,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf198, (18,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf199, (18,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf200, (18,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf201, (18, 18, 3, 3), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf202, (18,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf203, (18,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf204, (18,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf205, (18,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf206, (18, 18, 3, 3), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf207, (18,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf208, (18,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf209, (18,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf210, (18,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf211, (18, 18, 3, 3), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf212, (18,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf213, (18,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf214, (18,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf215, (18,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf216, (18, 18, 3, 3), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf217, (18,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf218, (18,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf219, (18,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf220, (18,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf221, (36, 36, 3, 3), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf222, (36,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf223, (36,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf224, (36,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf225, (36,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf226, (36, 36, 3, 3), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf227, (36,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf228, (36,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf229, (36,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf230, (36,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf231, (36, 36, 3, 3), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf232, (36,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf233, (36,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf234, (36,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf235, (36,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf236, (36, 36, 3, 3), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf237, (36,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf238, (36,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf239, (36,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf240, (36,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf241, (36, 36, 3, 3), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf242, (36,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf243, (36,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf244, (36,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf245, (36,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf246, (36, 36, 3, 3), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf247, (36,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf248, (36,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf249, (36,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf250, (36,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf251, (36, 36, 3, 3), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf252, (36,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf253, (36,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf254, (36,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf255, (36,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf256, (36, 36, 3, 3), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf257, (36,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf258, (36,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf259, (36,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf260, (36,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf261, (72, 72, 3, 3), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf262, (72,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf263, (72,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf264, (72,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf265, (72,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf266, (72, 72, 3, 3), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf267, (72,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf268, (72,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf269, (72,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf270, (72,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf271, (72, 72, 3, 3), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf272, (72,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf273, (72,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf274, (72,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf275, (72,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf276, (72, 72, 3, 3), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf277, (72,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf278, (72,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf279, (72,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf280, (72,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf281, (72, 72, 3, 3), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf282, (72,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf283, (72,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf284, (72,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf285, (72,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf286, (72, 72, 3, 3), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf287, (72,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf288, (72,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf289, (72,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf290, (72,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf291, (72, 72, 3, 3), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf292, (72,), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf293, (72,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf294, (72,), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf295, (72,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf296, (72, 72, 3, 3), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf297, (72,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf298, (72,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf299, (72,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf300, (72,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf301, (18, 36, 1, 1), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf302, (18,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf303, (18,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf304, (18,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf305, (18,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf306, (18, 72, 1, 1), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf307, (18,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf308, (18,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf309, (18,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf310, (18,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 23328, device=device(type='cuda', index=0))
    reader.tensor(buf311, (36, 18, 3, 3), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf312, (36,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf313, (36,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf314, (36,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf315, (36,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf316, (36, 72, 1, 1), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf317, (36,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf318, (36,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf319, (36,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf320, (36,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf321, (18, 18, 3, 3), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf322, (18,), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf323, (18,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf324, (18,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf325, (18,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf326, (72, 18, 3, 3), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf327, (72,), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf328, (72,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf329, (72,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf330, (72,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf331, (72, 36, 3, 3), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf332, (72,), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf333, (72,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf334, (72,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf335, (72,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf336, (18, 18, 3, 3), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf337, (18,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf338, (18,), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf339, (18,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf340, (18,), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf341, (18, 18, 3, 3), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf342, (18,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf343, (18,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf344, (18,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf345, (18,), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf346, (18, 18, 3, 3), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf347, (18,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf348, (18,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf349, (18,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf350, (18,), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf351, (18, 18, 3, 3), is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf352, (18,), is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf353, (18,), is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf354, (18,), is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf355, (18,), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf356, (18, 18, 3, 3), is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf357, (18,), is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf358, (18,), is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf359, (18,), is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf360, (18,), is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf361, (18, 18, 3, 3), is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf362, (18,), is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf363, (18,), is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf364, (18,), is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf365, (18,), is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf366, (18, 18, 3, 3), is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf367, (18,), is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf368, (18,), is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf369, (18,), is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf370, (18,), is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf371, (18, 18, 3, 3), is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf372, (18,), is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf373, (18,), is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf374, (18,), is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf375, (18,), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf376, (36, 36, 3, 3), is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf377, (36,), is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf378, (36,), is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf379, (36,), is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf380, (36,), is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf381, (36, 36, 3, 3), is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf382, (36,), is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf383, (36,), is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf384, (36,), is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf385, (36,), is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf386, (36, 36, 3, 3), is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf387, (36,), is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf388, (36,), is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf389, (36,), is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf390, (36,), is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf391, (36, 36, 3, 3), is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf392, (36,), is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf393, (36,), is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf394, (36,), is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf395, (36,), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf396, (36, 36, 3, 3), is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf397, (36,), is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf398, (36,), is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf399, (36,), is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf400, (36,), is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf401, (36, 36, 3, 3), is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf402, (36,), is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf403, (36,), is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf404, (36,), is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf405, (36,), is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf406, (36, 36, 3, 3), is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf407, (36,), is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf408, (36,), is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf409, (36,), is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf410, (36,), is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf411, (36, 36, 3, 3), is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf412, (36,), is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf413, (36,), is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf414, (36,), is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf415, (36,), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf416, (72, 72, 3, 3), is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf417, (72,), is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf418, (72,), is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf419, (72,), is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf420, (72,), is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf421, (72, 72, 3, 3), is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf422, (72,), is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf423, (72,), is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf424, (72,), is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf425, (72,), is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf426, (72, 72, 3, 3), is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf427, (72,), is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf428, (72,), is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf429, (72,), is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf430, (72,), is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf431, (72, 72, 3, 3), is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf432, (72,), is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf433, (72,), is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf434, (72,), is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf435, (72,), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf436, (72, 72, 3, 3), is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf437, (72,), is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf438, (72,), is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf439, (72,), is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf440, (72,), is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf441, (72, 72, 3, 3), is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf442, (72,), is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf443, (72,), is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf444, (72,), is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf445, (72,), is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf446, (72, 72, 3, 3), is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf447, (72,), is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf448, (72,), is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf449, (72,), is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf450, (72,), is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf451, (72, 72, 3, 3), is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf452, (72,), is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf453, (72,), is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf454, (72,), is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf455, (72,), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf456, (18, 36, 1, 1), is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf457, (18,), is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf458, (18,), is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf459, (18,), is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf460, (18,), is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf461, (18, 72, 1, 1), is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf462, (18,), is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf463, (18,), is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf464, (18,), is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf465, (18,), is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 23328, device=device(type='cuda', index=0))
    reader.tensor(buf466, (36, 18, 3, 3), is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf467, (36,), is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf468, (36,), is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf469, (36,), is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf470, (36,), is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf471, (36, 72, 1, 1), is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf472, (36,), is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf473, (36,), is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf474, (36,), is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf475, (36,), is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf476, (18, 18, 3, 3), is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf477, (18,), is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf478, (18,), is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf479, (18,), is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf480, (18,), is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf481, (72, 18, 3, 3), is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf482, (72,), is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf483, (72,), is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf484, (72,), is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf485, (72,), is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf486, (72, 36, 3, 3), is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf487, (72,), is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf488, (72,), is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf489, (72,), is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf490, (72,), is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf491, (18, 18, 3, 3), is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf492, (18,), is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf493, (18,), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf494, (18,), is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf495, (18,), is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf496, (18, 18, 3, 3), is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf497, (18,), is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf498, (18,), is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf499, (18,), is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf500, (18,), is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf501, (18, 18, 3, 3), is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf502, (18,), is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf503, (18,), is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf504, (18,), is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf505, (18,), is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf506, (18, 18, 3, 3), is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf507, (18,), is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf508, (18,), is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf509, (18,), is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf510, (18,), is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf511, (18, 18, 3, 3), is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf512, (18,), is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf513, (18,), is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf514, (18,), is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf515, (18,), is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf516, (18, 18, 3, 3), is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf517, (18,), is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf518, (18,), is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf519, (18,), is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf520, (18,), is_leaf=True)  # arg520_1
    buf521 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf521, (18, 18, 3, 3), is_leaf=True)  # arg521_1
    buf522 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf522, (18,), is_leaf=True)  # arg522_1
    buf523 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf523, (18,), is_leaf=True)  # arg523_1
    buf524 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf524, (18,), is_leaf=True)  # arg524_1
    buf525 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf525, (18,), is_leaf=True)  # arg525_1
    buf526 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf526, (18, 18, 3, 3), is_leaf=True)  # arg526_1
    buf527 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf527, (18,), is_leaf=True)  # arg527_1
    buf528 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf528, (18,), is_leaf=True)  # arg528_1
    buf529 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf529, (18,), is_leaf=True)  # arg529_1
    buf530 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf530, (18,), is_leaf=True)  # arg530_1
    buf531 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf531, (36, 36, 3, 3), is_leaf=True)  # arg531_1
    buf532 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf532, (36,), is_leaf=True)  # arg532_1
    buf533 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf533, (36,), is_leaf=True)  # arg533_1
    buf534 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf534, (36,), is_leaf=True)  # arg534_1
    buf535 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf535, (36,), is_leaf=True)  # arg535_1
    buf536 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf536, (36, 36, 3, 3), is_leaf=True)  # arg536_1
    buf537 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf537, (36,), is_leaf=True)  # arg537_1
    buf538 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf538, (36,), is_leaf=True)  # arg538_1
    buf539 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf539, (36,), is_leaf=True)  # arg539_1
    buf540 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf540, (36,), is_leaf=True)  # arg540_1
    buf541 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf541, (36, 36, 3, 3), is_leaf=True)  # arg541_1
    buf542 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf542, (36,), is_leaf=True)  # arg542_1
    buf543 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf543, (36,), is_leaf=True)  # arg543_1
    buf544 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf544, (36,), is_leaf=True)  # arg544_1
    buf545 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf545, (36,), is_leaf=True)  # arg545_1
    buf546 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf546, (36, 36, 3, 3), is_leaf=True)  # arg546_1
    buf547 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf547, (36,), is_leaf=True)  # arg547_1
    buf548 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf548, (36,), is_leaf=True)  # arg548_1
    buf549 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf549, (36,), is_leaf=True)  # arg549_1
    buf550 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf550, (36,), is_leaf=True)  # arg550_1
    buf551 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf551, (36, 36, 3, 3), is_leaf=True)  # arg551_1
    buf552 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf552, (36,), is_leaf=True)  # arg552_1
    buf553 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf553, (36,), is_leaf=True)  # arg553_1
    buf554 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf554, (36,), is_leaf=True)  # arg554_1
    buf555 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf555, (36,), is_leaf=True)  # arg555_1
    buf556 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf556, (36, 36, 3, 3), is_leaf=True)  # arg556_1
    buf557 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf557, (36,), is_leaf=True)  # arg557_1
    buf558 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf558, (36,), is_leaf=True)  # arg558_1
    buf559 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf559, (36,), is_leaf=True)  # arg559_1
    buf560 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf560, (36,), is_leaf=True)  # arg560_1
    buf561 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf561, (36, 36, 3, 3), is_leaf=True)  # arg561_1
    buf562 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf562, (36,), is_leaf=True)  # arg562_1
    buf563 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf563, (36,), is_leaf=True)  # arg563_1
    buf564 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf564, (36,), is_leaf=True)  # arg564_1
    buf565 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf565, (36,), is_leaf=True)  # arg565_1
    buf566 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf566, (36, 36, 3, 3), is_leaf=True)  # arg566_1
    buf567 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf567, (36,), is_leaf=True)  # arg567_1
    buf568 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf568, (36,), is_leaf=True)  # arg568_1
    buf569 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf569, (36,), is_leaf=True)  # arg569_1
    buf570 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf570, (36,), is_leaf=True)  # arg570_1
    buf571 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf571, (72, 72, 3, 3), is_leaf=True)  # arg571_1
    buf572 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf572, (72,), is_leaf=True)  # arg572_1
    buf573 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf573, (72,), is_leaf=True)  # arg573_1
    buf574 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf574, (72,), is_leaf=True)  # arg574_1
    buf575 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf575, (72,), is_leaf=True)  # arg575_1
    buf576 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf576, (72, 72, 3, 3), is_leaf=True)  # arg576_1
    buf577 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf577, (72,), is_leaf=True)  # arg577_1
    buf578 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf578, (72,), is_leaf=True)  # arg578_1
    buf579 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf579, (72,), is_leaf=True)  # arg579_1
    buf580 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf580, (72,), is_leaf=True)  # arg580_1
    buf581 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf581, (72, 72, 3, 3), is_leaf=True)  # arg581_1
    buf582 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf582, (72,), is_leaf=True)  # arg582_1
    buf583 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf583, (72,), is_leaf=True)  # arg583_1
    buf584 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf584, (72,), is_leaf=True)  # arg584_1
    buf585 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf585, (72,), is_leaf=True)  # arg585_1
    buf586 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf586, (72, 72, 3, 3), is_leaf=True)  # arg586_1
    buf587 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf587, (72,), is_leaf=True)  # arg587_1
    buf588 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf588, (72,), is_leaf=True)  # arg588_1
    buf589 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf589, (72,), is_leaf=True)  # arg589_1
    buf590 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf590, (72,), is_leaf=True)  # arg590_1
    buf591 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf591, (72, 72, 3, 3), is_leaf=True)  # arg591_1
    buf592 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf592, (72,), is_leaf=True)  # arg592_1
    buf593 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf593, (72,), is_leaf=True)  # arg593_1
    buf594 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf594, (72,), is_leaf=True)  # arg594_1
    buf595 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf595, (72,), is_leaf=True)  # arg595_1
    buf596 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf596, (72, 72, 3, 3), is_leaf=True)  # arg596_1
    buf597 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf597, (72,), is_leaf=True)  # arg597_1
    buf598 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf598, (72,), is_leaf=True)  # arg598_1
    buf599 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf599, (72,), is_leaf=True)  # arg599_1
    buf600 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf600, (72,), is_leaf=True)  # arg600_1
    buf601 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf601, (72, 72, 3, 3), is_leaf=True)  # arg601_1
    buf602 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf602, (72,), is_leaf=True)  # arg602_1
    buf603 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf603, (72,), is_leaf=True)  # arg603_1
    buf604 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf604, (72,), is_leaf=True)  # arg604_1
    buf605 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf605, (72,), is_leaf=True)  # arg605_1
    buf606 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf606, (72, 72, 3, 3), is_leaf=True)  # arg606_1
    buf607 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf607, (72,), is_leaf=True)  # arg607_1
    buf608 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf608, (72,), is_leaf=True)  # arg608_1
    buf609 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf609, (72,), is_leaf=True)  # arg609_1
    buf610 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf610, (72,), is_leaf=True)  # arg610_1
    buf611 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf611, (18, 36, 1, 1), is_leaf=True)  # arg611_1
    buf612 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf612, (18,), is_leaf=True)  # arg612_1
    buf613 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf613, (18,), is_leaf=True)  # arg613_1
    buf614 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf614, (18,), is_leaf=True)  # arg614_1
    buf615 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf615, (18,), is_leaf=True)  # arg615_1
    buf616 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf616, (18, 72, 1, 1), is_leaf=True)  # arg616_1
    buf617 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf617, (18,), is_leaf=True)  # arg617_1
    buf618 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf618, (18,), is_leaf=True)  # arg618_1
    buf619 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf619, (18,), is_leaf=True)  # arg619_1
    buf620 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf620, (18,), is_leaf=True)  # arg620_1
    buf621 = reader.storage(None, 23328, device=device(type='cuda', index=0))
    reader.tensor(buf621, (36, 18, 3, 3), is_leaf=True)  # arg621_1
    buf622 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf622, (36,), is_leaf=True)  # arg622_1
    buf623 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf623, (36,), is_leaf=True)  # arg623_1
    buf624 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf624, (36,), is_leaf=True)  # arg624_1
    buf625 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf625, (36,), is_leaf=True)  # arg625_1
    buf626 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf626, (36, 72, 1, 1), is_leaf=True)  # arg626_1
    buf627 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf627, (36,), is_leaf=True)  # arg627_1
    buf628 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf628, (36,), is_leaf=True)  # arg628_1
    buf629 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf629, (36,), is_leaf=True)  # arg629_1
    buf630 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf630, (36,), is_leaf=True)  # arg630_1
    buf631 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf631, (18, 18, 3, 3), is_leaf=True)  # arg631_1
    buf632 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf632, (18,), is_leaf=True)  # arg632_1
    buf633 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf633, (18,), is_leaf=True)  # arg633_1
    buf634 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf634, (18,), is_leaf=True)  # arg634_1
    buf635 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf635, (18,), is_leaf=True)  # arg635_1
    buf636 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf636, (72, 18, 3, 3), is_leaf=True)  # arg636_1
    buf637 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf637, (72,), is_leaf=True)  # arg637_1
    buf638 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf638, (72,), is_leaf=True)  # arg638_1
    buf639 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf639, (72,), is_leaf=True)  # arg639_1
    buf640 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf640, (72,), is_leaf=True)  # arg640_1
    buf641 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf641, (72, 36, 3, 3), is_leaf=True)  # arg641_1
    buf642 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf642, (72,), is_leaf=True)  # arg642_1
    buf643 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf643, (72,), is_leaf=True)  # arg643_1
    buf644 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf644, (72,), is_leaf=True)  # arg644_1
    buf645 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf645, (72,), is_leaf=True)  # arg645_1
    buf646 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf646, (18, 18, 3, 3), is_leaf=True)  # arg646_1
    buf647 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf647, (18,), is_leaf=True)  # arg647_1
    buf648 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf648, (18,), is_leaf=True)  # arg648_1
    buf649 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf649, (18,), is_leaf=True)  # arg649_1
    buf650 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf650, (18,), is_leaf=True)  # arg650_1
    buf651 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf651, (18, 18, 3, 3), is_leaf=True)  # arg651_1
    buf652 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf652, (18,), is_leaf=True)  # arg652_1
    buf653 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf653, (18,), is_leaf=True)  # arg653_1
    buf654 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf654, (18,), is_leaf=True)  # arg654_1
    buf655 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf655, (18,), is_leaf=True)  # arg655_1
    buf656 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf656, (18, 18, 3, 3), is_leaf=True)  # arg656_1
    buf657 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf657, (18,), is_leaf=True)  # arg657_1
    buf658 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf658, (18,), is_leaf=True)  # arg658_1
    buf659 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf659, (18,), is_leaf=True)  # arg659_1
    buf660 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf660, (18,), is_leaf=True)  # arg660_1
    buf661 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf661, (18, 18, 3, 3), is_leaf=True)  # arg661_1
    buf662 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf662, (18,), is_leaf=True)  # arg662_1
    buf663 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf663, (18,), is_leaf=True)  # arg663_1
    buf664 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf664, (18,), is_leaf=True)  # arg664_1
    buf665 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf665, (18,), is_leaf=True)  # arg665_1
    buf666 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf666, (18, 18, 3, 3), is_leaf=True)  # arg666_1
    buf667 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf667, (18,), is_leaf=True)  # arg667_1
    buf668 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf668, (18,), is_leaf=True)  # arg668_1
    buf669 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf669, (18,), is_leaf=True)  # arg669_1
    buf670 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf670, (18,), is_leaf=True)  # arg670_1
    buf671 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf671, (18, 18, 3, 3), is_leaf=True)  # arg671_1
    buf672 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf672, (18,), is_leaf=True)  # arg672_1
    buf673 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf673, (18,), is_leaf=True)  # arg673_1
    buf674 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf674, (18,), is_leaf=True)  # arg674_1
    buf675 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf675, (18,), is_leaf=True)  # arg675_1
    buf676 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf676, (18, 18, 3, 3), is_leaf=True)  # arg676_1
    buf677 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf677, (18,), is_leaf=True)  # arg677_1
    buf678 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf678, (18,), is_leaf=True)  # arg678_1
    buf679 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf679, (18,), is_leaf=True)  # arg679_1
    buf680 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf680, (18,), is_leaf=True)  # arg680_1
    buf681 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf681, (18, 18, 3, 3), is_leaf=True)  # arg681_1
    buf682 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf682, (18,), is_leaf=True)  # arg682_1
    buf683 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf683, (18,), is_leaf=True)  # arg683_1
    buf684 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf684, (18,), is_leaf=True)  # arg684_1
    buf685 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf685, (18,), is_leaf=True)  # arg685_1
    buf686 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf686, (36, 36, 3, 3), is_leaf=True)  # arg686_1
    buf687 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf687, (36,), is_leaf=True)  # arg687_1
    buf688 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf688, (36,), is_leaf=True)  # arg688_1
    buf689 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf689, (36,), is_leaf=True)  # arg689_1
    buf690 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf690, (36,), is_leaf=True)  # arg690_1
    buf691 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf691, (36, 36, 3, 3), is_leaf=True)  # arg691_1
    buf692 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf692, (36,), is_leaf=True)  # arg692_1
    buf693 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf693, (36,), is_leaf=True)  # arg693_1
    buf694 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf694, (36,), is_leaf=True)  # arg694_1
    buf695 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf695, (36,), is_leaf=True)  # arg695_1
    buf696 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf696, (36, 36, 3, 3), is_leaf=True)  # arg696_1
    buf697 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf697, (36,), is_leaf=True)  # arg697_1
    buf698 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf698, (36,), is_leaf=True)  # arg698_1
    buf699 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf699, (36,), is_leaf=True)  # arg699_1
    buf700 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf700, (36,), is_leaf=True)  # arg700_1
    buf701 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf701, (36, 36, 3, 3), is_leaf=True)  # arg701_1
    buf702 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf702, (36,), is_leaf=True)  # arg702_1
    buf703 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf703, (36,), is_leaf=True)  # arg703_1
    buf704 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf704, (36,), is_leaf=True)  # arg704_1
    buf705 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf705, (36,), is_leaf=True)  # arg705_1
    buf706 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf706, (36, 36, 3, 3), is_leaf=True)  # arg706_1
    buf707 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf707, (36,), is_leaf=True)  # arg707_1
    buf708 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf708, (36,), is_leaf=True)  # arg708_1
    buf709 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf709, (36,), is_leaf=True)  # arg709_1
    buf710 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf710, (36,), is_leaf=True)  # arg710_1
    buf711 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf711, (36, 36, 3, 3), is_leaf=True)  # arg711_1
    buf712 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf712, (36,), is_leaf=True)  # arg712_1
    buf713 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf713, (36,), is_leaf=True)  # arg713_1
    buf714 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf714, (36,), is_leaf=True)  # arg714_1
    buf715 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf715, (36,), is_leaf=True)  # arg715_1
    buf716 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf716, (36, 36, 3, 3), is_leaf=True)  # arg716_1
    buf717 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf717, (36,), is_leaf=True)  # arg717_1
    buf718 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf718, (36,), is_leaf=True)  # arg718_1
    buf719 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf719, (36,), is_leaf=True)  # arg719_1
    buf720 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf720, (36,), is_leaf=True)  # arg720_1
    buf721 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf721, (36, 36, 3, 3), is_leaf=True)  # arg721_1
    buf722 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf722, (36,), is_leaf=True)  # arg722_1
    buf723 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf723, (36,), is_leaf=True)  # arg723_1
    buf724 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf724, (36,), is_leaf=True)  # arg724_1
    buf725 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf725, (36,), is_leaf=True)  # arg725_1
    buf726 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf726, (72, 72, 3, 3), is_leaf=True)  # arg726_1
    buf727 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf727, (72,), is_leaf=True)  # arg727_1
    buf728 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf728, (72,), is_leaf=True)  # arg728_1
    buf729 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf729, (72,), is_leaf=True)  # arg729_1
    buf730 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf730, (72,), is_leaf=True)  # arg730_1
    buf731 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf731, (72, 72, 3, 3), is_leaf=True)  # arg731_1
    buf732 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf732, (72,), is_leaf=True)  # arg732_1
    buf733 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf733, (72,), is_leaf=True)  # arg733_1
    buf734 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf734, (72,), is_leaf=True)  # arg734_1
    buf735 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf735, (72,), is_leaf=True)  # arg735_1
    buf736 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf736, (72, 72, 3, 3), is_leaf=True)  # arg736_1
    buf737 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf737, (72,), is_leaf=True)  # arg737_1
    buf738 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf738, (72,), is_leaf=True)  # arg738_1
    buf739 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf739, (72,), is_leaf=True)  # arg739_1
    buf740 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf740, (72,), is_leaf=True)  # arg740_1
    buf741 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf741, (72, 72, 3, 3), is_leaf=True)  # arg741_1
    buf742 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf742, (72,), is_leaf=True)  # arg742_1
    buf743 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf743, (72,), is_leaf=True)  # arg743_1
    buf744 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf744, (72,), is_leaf=True)  # arg744_1
    buf745 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf745, (72,), is_leaf=True)  # arg745_1
    buf746 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf746, (72, 72, 3, 3), is_leaf=True)  # arg746_1
    buf747 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf747, (72,), is_leaf=True)  # arg747_1
    buf748 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf748, (72,), is_leaf=True)  # arg748_1
    buf749 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf749, (72,), is_leaf=True)  # arg749_1
    buf750 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf750, (72,), is_leaf=True)  # arg750_1
    buf751 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf751, (72, 72, 3, 3), is_leaf=True)  # arg751_1
    buf752 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf752, (72,), is_leaf=True)  # arg752_1
    buf753 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf753, (72,), is_leaf=True)  # arg753_1
    buf754 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf754, (72,), is_leaf=True)  # arg754_1
    buf755 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf755, (72,), is_leaf=True)  # arg755_1
    buf756 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf756, (72, 72, 3, 3), is_leaf=True)  # arg756_1
    buf757 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf757, (72,), is_leaf=True)  # arg757_1
    buf758 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf758, (72,), is_leaf=True)  # arg758_1
    buf759 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf759, (72,), is_leaf=True)  # arg759_1
    buf760 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf760, (72,), is_leaf=True)  # arg760_1
    buf761 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf761, (72, 72, 3, 3), is_leaf=True)  # arg761_1
    buf762 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf762, (72,), is_leaf=True)  # arg762_1
    buf763 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf763, (72,), is_leaf=True)  # arg763_1
    buf764 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf764, (72,), is_leaf=True)  # arg764_1
    buf765 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf765, (72,), is_leaf=True)  # arg765_1
    buf766 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf766, (18, 36, 1, 1), is_leaf=True)  # arg766_1
    buf767 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf767, (18,), is_leaf=True)  # arg767_1
    buf768 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf768, (18,), is_leaf=True)  # arg768_1
    buf769 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf769, (18,), is_leaf=True)  # arg769_1
    buf770 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf770, (18,), is_leaf=True)  # arg770_1
    buf771 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf771, (18, 72, 1, 1), is_leaf=True)  # arg771_1
    buf772 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf772, (18,), is_leaf=True)  # arg772_1
    buf773 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf773, (18,), is_leaf=True)  # arg773_1
    buf774 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf774, (18,), is_leaf=True)  # arg774_1
    buf775 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf775, (18,), is_leaf=True)  # arg775_1
    buf776 = reader.storage(None, 23328, device=device(type='cuda', index=0))
    reader.tensor(buf776, (36, 18, 3, 3), is_leaf=True)  # arg776_1
    buf777 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf777, (36,), is_leaf=True)  # arg777_1
    buf778 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf778, (36,), is_leaf=True)  # arg778_1
    buf779 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf779, (36,), is_leaf=True)  # arg779_1
    buf780 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf780, (36,), is_leaf=True)  # arg780_1
    buf781 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf781, (36, 72, 1, 1), is_leaf=True)  # arg781_1
    buf782 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf782, (36,), is_leaf=True)  # arg782_1
    buf783 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf783, (36,), is_leaf=True)  # arg783_1
    buf784 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf784, (36,), is_leaf=True)  # arg784_1
    buf785 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf785, (36,), is_leaf=True)  # arg785_1
    buf786 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf786, (18, 18, 3, 3), is_leaf=True)  # arg786_1
    buf787 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf787, (18,), is_leaf=True)  # arg787_1
    buf788 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf788, (18,), is_leaf=True)  # arg788_1
    buf789 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf789, (18,), is_leaf=True)  # arg789_1
    buf790 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf790, (18,), is_leaf=True)  # arg790_1
    buf791 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf791, (72, 18, 3, 3), is_leaf=True)  # arg791_1
    buf792 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf792, (72,), is_leaf=True)  # arg792_1
    buf793 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf793, (72,), is_leaf=True)  # arg793_1
    buf794 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf794, (72,), is_leaf=True)  # arg794_1
    buf795 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf795, (72,), is_leaf=True)  # arg795_1
    buf796 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf796, (72, 36, 3, 3), is_leaf=True)  # arg796_1
    buf797 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf797, (72,), is_leaf=True)  # arg797_1
    buf798 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf798, (72,), is_leaf=True)  # arg798_1
    buf799 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf799, (72,), is_leaf=True)  # arg799_1
    buf800 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf800, (72,), is_leaf=True)  # arg800_1
    buf801 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf801, (144, 72, 3, 3), is_leaf=True)  # arg801_1
    buf802 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf802, (144,), is_leaf=True)  # arg802_1
    buf803 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf803, (144,), is_leaf=True)  # arg803_1
    buf804 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf804, (144,), is_leaf=True)  # arg804_1
    buf805 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf805, (144,), is_leaf=True)  # arg805_1
    buf806 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf806, (18, 18, 3, 3), is_leaf=True)  # arg806_1
    buf807 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf807, (18,), is_leaf=True)  # arg807_1
    buf808 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf808, (18,), is_leaf=True)  # arg808_1
    buf809 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf809, (18,), is_leaf=True)  # arg809_1
    buf810 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf810, (18,), is_leaf=True)  # arg810_1
    buf811 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf811, (18, 18, 3, 3), is_leaf=True)  # arg811_1
    buf812 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf812, (18,), is_leaf=True)  # arg812_1
    buf813 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf813, (18,), is_leaf=True)  # arg813_1
    buf814 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf814, (18,), is_leaf=True)  # arg814_1
    buf815 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf815, (18,), is_leaf=True)  # arg815_1
    buf816 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf816, (18, 18, 3, 3), is_leaf=True)  # arg816_1
    buf817 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf817, (18,), is_leaf=True)  # arg817_1
    buf818 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf818, (18,), is_leaf=True)  # arg818_1
    buf819 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf819, (18,), is_leaf=True)  # arg819_1
    buf820 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf820, (18,), is_leaf=True)  # arg820_1
    buf821 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf821, (18, 18, 3, 3), is_leaf=True)  # arg821_1
    buf822 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf822, (18,), is_leaf=True)  # arg822_1
    buf823 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf823, (18,), is_leaf=True)  # arg823_1
    buf824 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf824, (18,), is_leaf=True)  # arg824_1
    buf825 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf825, (18,), is_leaf=True)  # arg825_1
    buf826 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf826, (18, 18, 3, 3), is_leaf=True)  # arg826_1
    buf827 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf827, (18,), is_leaf=True)  # arg827_1
    buf828 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf828, (18,), is_leaf=True)  # arg828_1
    buf829 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf829, (18,), is_leaf=True)  # arg829_1
    buf830 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf830, (18,), is_leaf=True)  # arg830_1
    buf831 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf831, (18, 18, 3, 3), is_leaf=True)  # arg831_1
    buf832 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf832, (18,), is_leaf=True)  # arg832_1
    buf833 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf833, (18,), is_leaf=True)  # arg833_1
    buf834 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf834, (18,), is_leaf=True)  # arg834_1
    buf835 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf835, (18,), is_leaf=True)  # arg835_1
    buf836 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf836, (18, 18, 3, 3), is_leaf=True)  # arg836_1
    buf837 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf837, (18,), is_leaf=True)  # arg837_1
    buf838 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf838, (18,), is_leaf=True)  # arg838_1
    buf839 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf839, (18,), is_leaf=True)  # arg839_1
    buf840 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf840, (18,), is_leaf=True)  # arg840_1
    buf841 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf841, (18, 18, 3, 3), is_leaf=True)  # arg841_1
    buf842 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf842, (18,), is_leaf=True)  # arg842_1
    buf843 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf843, (18,), is_leaf=True)  # arg843_1
    buf844 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf844, (18,), is_leaf=True)  # arg844_1
    buf845 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf845, (18,), is_leaf=True)  # arg845_1
    buf846 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf846, (36, 36, 3, 3), is_leaf=True)  # arg846_1
    buf847 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf847, (36,), is_leaf=True)  # arg847_1
    buf848 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf848, (36,), is_leaf=True)  # arg848_1
    buf849 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf849, (36,), is_leaf=True)  # arg849_1
    buf850 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf850, (36,), is_leaf=True)  # arg850_1
    buf851 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf851, (36, 36, 3, 3), is_leaf=True)  # arg851_1
    buf852 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf852, (36,), is_leaf=True)  # arg852_1
    buf853 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf853, (36,), is_leaf=True)  # arg853_1
    buf854 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf854, (36,), is_leaf=True)  # arg854_1
    buf855 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf855, (36,), is_leaf=True)  # arg855_1
    buf856 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf856, (36, 36, 3, 3), is_leaf=True)  # arg856_1
    buf857 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf857, (36,), is_leaf=True)  # arg857_1
    buf858 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf858, (36,), is_leaf=True)  # arg858_1
    buf859 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf859, (36,), is_leaf=True)  # arg859_1
    buf860 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf860, (36,), is_leaf=True)  # arg860_1
    buf861 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf861, (36, 36, 3, 3), is_leaf=True)  # arg861_1
    buf862 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf862, (36,), is_leaf=True)  # arg862_1
    buf863 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf863, (36,), is_leaf=True)  # arg863_1
    buf864 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf864, (36,), is_leaf=True)  # arg864_1
    buf865 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf865, (36,), is_leaf=True)  # arg865_1
    buf866 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf866, (36, 36, 3, 3), is_leaf=True)  # arg866_1
    buf867 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf867, (36,), is_leaf=True)  # arg867_1
    buf868 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf868, (36,), is_leaf=True)  # arg868_1
    buf869 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf869, (36,), is_leaf=True)  # arg869_1
    buf870 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf870, (36,), is_leaf=True)  # arg870_1
    buf871 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf871, (36, 36, 3, 3), is_leaf=True)  # arg871_1
    buf872 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf872, (36,), is_leaf=True)  # arg872_1
    buf873 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf873, (36,), is_leaf=True)  # arg873_1
    buf874 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf874, (36,), is_leaf=True)  # arg874_1
    buf875 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf875, (36,), is_leaf=True)  # arg875_1
    buf876 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf876, (36, 36, 3, 3), is_leaf=True)  # arg876_1
    buf877 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf877, (36,), is_leaf=True)  # arg877_1
    buf878 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf878, (36,), is_leaf=True)  # arg878_1
    buf879 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf879, (36,), is_leaf=True)  # arg879_1
    buf880 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf880, (36,), is_leaf=True)  # arg880_1
    buf881 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf881, (36, 36, 3, 3), is_leaf=True)  # arg881_1
    buf882 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf882, (36,), is_leaf=True)  # arg882_1
    buf883 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf883, (36,), is_leaf=True)  # arg883_1
    buf884 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf884, (36,), is_leaf=True)  # arg884_1
    buf885 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf885, (36,), is_leaf=True)  # arg885_1
    buf886 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf886, (72, 72, 3, 3), is_leaf=True)  # arg886_1
    buf887 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf887, (72,), is_leaf=True)  # arg887_1
    buf888 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf888, (72,), is_leaf=True)  # arg888_1
    buf889 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf889, (72,), is_leaf=True)  # arg889_1
    buf890 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf890, (72,), is_leaf=True)  # arg890_1
    buf891 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf891, (72, 72, 3, 3), is_leaf=True)  # arg891_1
    buf892 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf892, (72,), is_leaf=True)  # arg892_1
    buf893 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf893, (72,), is_leaf=True)  # arg893_1
    buf894 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf894, (72,), is_leaf=True)  # arg894_1
    buf895 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf895, (72,), is_leaf=True)  # arg895_1
    buf896 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf896, (72, 72, 3, 3), is_leaf=True)  # arg896_1
    buf897 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf897, (72,), is_leaf=True)  # arg897_1
    buf898 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf898, (72,), is_leaf=True)  # arg898_1
    buf899 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf899, (72,), is_leaf=True)  # arg899_1
    buf900 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf900, (72,), is_leaf=True)  # arg900_1
    buf901 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf901, (72, 72, 3, 3), is_leaf=True)  # arg901_1
    buf902 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf902, (72,), is_leaf=True)  # arg902_1
    buf903 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf903, (72,), is_leaf=True)  # arg903_1
    buf904 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf904, (72,), is_leaf=True)  # arg904_1
    buf905 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf905, (72,), is_leaf=True)  # arg905_1
    buf906 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf906, (72, 72, 3, 3), is_leaf=True)  # arg906_1
    buf907 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf907, (72,), is_leaf=True)  # arg907_1
    buf908 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf908, (72,), is_leaf=True)  # arg908_1
    buf909 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf909, (72,), is_leaf=True)  # arg909_1
    buf910 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf910, (72,), is_leaf=True)  # arg910_1
    buf911 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf911, (72, 72, 3, 3), is_leaf=True)  # arg911_1
    buf912 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf912, (72,), is_leaf=True)  # arg912_1
    buf913 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf913, (72,), is_leaf=True)  # arg913_1
    buf914 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf914, (72,), is_leaf=True)  # arg914_1
    buf915 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf915, (72,), is_leaf=True)  # arg915_1
    buf916 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf916, (72, 72, 3, 3), is_leaf=True)  # arg916_1
    buf917 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf917, (72,), is_leaf=True)  # arg917_1
    buf918 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf918, (72,), is_leaf=True)  # arg918_1
    buf919 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf919, (72,), is_leaf=True)  # arg919_1
    buf920 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf920, (72,), is_leaf=True)  # arg920_1
    buf921 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf921, (72, 72, 3, 3), is_leaf=True)  # arg921_1
    buf922 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf922, (72,), is_leaf=True)  # arg922_1
    buf923 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf923, (72,), is_leaf=True)  # arg923_1
    buf924 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf924, (72,), is_leaf=True)  # arg924_1
    buf925 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf925, (72,), is_leaf=True)  # arg925_1
    buf926 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf926, (144, 144, 3, 3), is_leaf=True)  # arg926_1
    buf927 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf927, (144,), is_leaf=True)  # arg927_1
    buf928 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf928, (144,), is_leaf=True)  # arg928_1
    buf929 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf929, (144,), is_leaf=True)  # arg929_1
    buf930 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf930, (144,), is_leaf=True)  # arg930_1
    buf931 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf931, (144, 144, 3, 3), is_leaf=True)  # arg931_1
    buf932 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf932, (144,), is_leaf=True)  # arg932_1
    buf933 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf933, (144,), is_leaf=True)  # arg933_1
    buf934 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf934, (144,), is_leaf=True)  # arg934_1
    buf935 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf935, (144,), is_leaf=True)  # arg935_1
    buf936 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf936, (144, 144, 3, 3), is_leaf=True)  # arg936_1
    buf937 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf937, (144,), is_leaf=True)  # arg937_1
    buf938 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf938, (144,), is_leaf=True)  # arg938_1
    buf939 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf939, (144,), is_leaf=True)  # arg939_1
    buf940 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf940, (144,), is_leaf=True)  # arg940_1
    buf941 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf941, (144, 144, 3, 3), is_leaf=True)  # arg941_1
    buf942 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf942, (144,), is_leaf=True)  # arg942_1
    buf943 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf943, (144,), is_leaf=True)  # arg943_1
    buf944 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf944, (144,), is_leaf=True)  # arg944_1
    buf945 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf945, (144,), is_leaf=True)  # arg945_1
    buf946 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf946, (144, 144, 3, 3), is_leaf=True)  # arg946_1
    buf947 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf947, (144,), is_leaf=True)  # arg947_1
    buf948 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf948, (144,), is_leaf=True)  # arg948_1
    buf949 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf949, (144,), is_leaf=True)  # arg949_1
    buf950 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf950, (144,), is_leaf=True)  # arg950_1
    buf951 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf951, (144, 144, 3, 3), is_leaf=True)  # arg951_1
    buf952 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf952, (144,), is_leaf=True)  # arg952_1
    buf953 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf953, (144,), is_leaf=True)  # arg953_1
    buf954 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf954, (144,), is_leaf=True)  # arg954_1
    buf955 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf955, (144,), is_leaf=True)  # arg955_1
    buf956 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf956, (144, 144, 3, 3), is_leaf=True)  # arg956_1
    buf957 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf957, (144,), is_leaf=True)  # arg957_1
    buf958 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf958, (144,), is_leaf=True)  # arg958_1
    buf959 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf959, (144,), is_leaf=True)  # arg959_1
    buf960 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf960, (144,), is_leaf=True)  # arg960_1
    buf961 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf961, (144, 144, 3, 3), is_leaf=True)  # arg961_1
    buf962 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf962, (144,), is_leaf=True)  # arg962_1
    buf963 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf963, (144,), is_leaf=True)  # arg963_1
    buf964 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf964, (144,), is_leaf=True)  # arg964_1
    buf965 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf965, (144,), is_leaf=True)  # arg965_1
    buf966 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf966, (18, 36, 1, 1), is_leaf=True)  # arg966_1
    buf967 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf967, (18,), is_leaf=True)  # arg967_1
    buf968 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf968, (18,), is_leaf=True)  # arg968_1
    buf969 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf969, (18,), is_leaf=True)  # arg969_1
    buf970 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf970, (18,), is_leaf=True)  # arg970_1
    buf971 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf971, (18, 72, 1, 1), is_leaf=True)  # arg971_1
    buf972 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf972, (18,), is_leaf=True)  # arg972_1
    buf973 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf973, (18,), is_leaf=True)  # arg973_1
    buf974 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf974, (18,), is_leaf=True)  # arg974_1
    buf975 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf975, (18,), is_leaf=True)  # arg975_1
    buf976 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf976, (18, 144, 1, 1), is_leaf=True)  # arg976_1
    buf977 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf977, (18,), is_leaf=True)  # arg977_1
    buf978 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf978, (18,), is_leaf=True)  # arg978_1
    buf979 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf979, (18,), is_leaf=True)  # arg979_1
    buf980 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf980, (18,), is_leaf=True)  # arg980_1
    buf981 = reader.storage(None, 23328, device=device(type='cuda', index=0))
    reader.tensor(buf981, (36, 18, 3, 3), is_leaf=True)  # arg981_1
    buf982 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf982, (36,), is_leaf=True)  # arg982_1
    buf983 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf983, (36,), is_leaf=True)  # arg983_1
    buf984 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf984, (36,), is_leaf=True)  # arg984_1
    buf985 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf985, (36,), is_leaf=True)  # arg985_1
    buf986 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf986, (36, 72, 1, 1), is_leaf=True)  # arg986_1
    buf987 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf987, (36,), is_leaf=True)  # arg987_1
    buf988 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf988, (36,), is_leaf=True)  # arg988_1
    buf989 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf989, (36,), is_leaf=True)  # arg989_1
    buf990 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf990, (36,), is_leaf=True)  # arg990_1
    buf991 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf991, (36, 144, 1, 1), is_leaf=True)  # arg991_1
    buf992 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf992, (36,), is_leaf=True)  # arg992_1
    buf993 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf993, (36,), is_leaf=True)  # arg993_1
    buf994 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf994, (36,), is_leaf=True)  # arg994_1
    buf995 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf995, (36,), is_leaf=True)  # arg995_1
    buf996 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf996, (18, 18, 3, 3), is_leaf=True)  # arg996_1
    buf997 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf997, (18,), is_leaf=True)  # arg997_1
    buf998 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf998, (18,), is_leaf=True)  # arg998_1
    buf999 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf999, (18,), is_leaf=True)  # arg999_1
    buf1000 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1000, (18,), is_leaf=True)  # arg1000_1
    buf1001 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1001, (72, 18, 3, 3), is_leaf=True)  # arg1001_1
    buf1002 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1002, (72,), is_leaf=True)  # arg1002_1
    buf1003 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1003, (72,), is_leaf=True)  # arg1003_1
    buf1004 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1004, (72,), is_leaf=True)  # arg1004_1
    buf1005 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1005, (72,), is_leaf=True)  # arg1005_1
    buf1006 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf1006, (72, 36, 3, 3), is_leaf=True)  # arg1006_1
    buf1007 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1007, (72,), is_leaf=True)  # arg1007_1
    buf1008 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1008, (72,), is_leaf=True)  # arg1008_1
    buf1009 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1009, (72,), is_leaf=True)  # arg1009_1
    buf1010 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1010, (72,), is_leaf=True)  # arg1010_1
    buf1011 = reader.storage(None, 41472, device=device(type='cuda', index=0))
    reader.tensor(buf1011, (72, 144, 1, 1), is_leaf=True)  # arg1011_1
    buf1012 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1012, (72,), is_leaf=True)  # arg1012_1
    buf1013 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1013, (72,), is_leaf=True)  # arg1013_1
    buf1014 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1014, (72,), is_leaf=True)  # arg1014_1
    buf1015 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1015, (72,), is_leaf=True)  # arg1015_1
    buf1016 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1016, (18, 18, 3, 3), is_leaf=True)  # arg1016_1
    buf1017 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1017, (18,), is_leaf=True)  # arg1017_1
    buf1018 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1018, (18,), is_leaf=True)  # arg1018_1
    buf1019 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1019, (18,), is_leaf=True)  # arg1019_1
    buf1020 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1020, (18,), is_leaf=True)  # arg1020_1
    buf1021 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1021, (18, 18, 3, 3), is_leaf=True)  # arg1021_1
    buf1022 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1022, (18,), is_leaf=True)  # arg1022_1
    buf1023 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1023, (18,), is_leaf=True)  # arg1023_1
    buf1024 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1024, (18,), is_leaf=True)  # arg1024_1
    buf1025 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1025, (18,), is_leaf=True)  # arg1025_1
    buf1026 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf1026, (144, 18, 3, 3), is_leaf=True)  # arg1026_1
    buf1027 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1027, (144,), is_leaf=True)  # arg1027_1
    buf1028 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1028, (144,), is_leaf=True)  # arg1028_1
    buf1029 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1029, (144,), is_leaf=True)  # arg1029_1
    buf1030 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1030, (144,), is_leaf=True)  # arg1030_1
    buf1031 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1031, (36, 36, 3, 3), is_leaf=True)  # arg1031_1
    buf1032 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1032, (36,), is_leaf=True)  # arg1032_1
    buf1033 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1033, (36,), is_leaf=True)  # arg1033_1
    buf1034 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1034, (36,), is_leaf=True)  # arg1034_1
    buf1035 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1035, (36,), is_leaf=True)  # arg1035_1
    buf1036 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1036, (144, 36, 3, 3), is_leaf=True)  # arg1036_1
    buf1037 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1037, (144,), is_leaf=True)  # arg1037_1
    buf1038 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1038, (144,), is_leaf=True)  # arg1038_1
    buf1039 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1039, (144,), is_leaf=True)  # arg1039_1
    buf1040 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1040, (144,), is_leaf=True)  # arg1040_1
    buf1041 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf1041, (144, 72, 3, 3), is_leaf=True)  # arg1041_1
    buf1042 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1042, (144,), is_leaf=True)  # arg1042_1
    buf1043 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1043, (144,), is_leaf=True)  # arg1043_1
    buf1044 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1044, (144,), is_leaf=True)  # arg1044_1
    buf1045 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1045, (144,), is_leaf=True)  # arg1045_1
    buf1046 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1046, (18, 18, 3, 3), is_leaf=True)  # arg1046_1
    buf1047 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1047, (18,), is_leaf=True)  # arg1047_1
    buf1048 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1048, (18,), is_leaf=True)  # arg1048_1
    buf1049 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1049, (18,), is_leaf=True)  # arg1049_1
    buf1050 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1050, (18,), is_leaf=True)  # arg1050_1
    buf1051 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1051, (18, 18, 3, 3), is_leaf=True)  # arg1051_1
    buf1052 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1052, (18,), is_leaf=True)  # arg1052_1
    buf1053 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1053, (18,), is_leaf=True)  # arg1053_1
    buf1054 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1054, (18,), is_leaf=True)  # arg1054_1
    buf1055 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1055, (18,), is_leaf=True)  # arg1055_1
    buf1056 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1056, (18, 18, 3, 3), is_leaf=True)  # arg1056_1
    buf1057 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1057, (18,), is_leaf=True)  # arg1057_1
    buf1058 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1058, (18,), is_leaf=True)  # arg1058_1
    buf1059 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1059, (18,), is_leaf=True)  # arg1059_1
    buf1060 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1060, (18,), is_leaf=True)  # arg1060_1
    buf1061 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1061, (18, 18, 3, 3), is_leaf=True)  # arg1061_1
    buf1062 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1062, (18,), is_leaf=True)  # arg1062_1
    buf1063 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1063, (18,), is_leaf=True)  # arg1063_1
    buf1064 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1064, (18,), is_leaf=True)  # arg1064_1
    buf1065 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1065, (18,), is_leaf=True)  # arg1065_1
    buf1066 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1066, (18, 18, 3, 3), is_leaf=True)  # arg1066_1
    buf1067 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1067, (18,), is_leaf=True)  # arg1067_1
    buf1068 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1068, (18,), is_leaf=True)  # arg1068_1
    buf1069 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1069, (18,), is_leaf=True)  # arg1069_1
    buf1070 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1070, (18,), is_leaf=True)  # arg1070_1
    buf1071 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1071, (18, 18, 3, 3), is_leaf=True)  # arg1071_1
    buf1072 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1072, (18,), is_leaf=True)  # arg1072_1
    buf1073 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1073, (18,), is_leaf=True)  # arg1073_1
    buf1074 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1074, (18,), is_leaf=True)  # arg1074_1
    buf1075 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1075, (18,), is_leaf=True)  # arg1075_1
    buf1076 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1076, (18, 18, 3, 3), is_leaf=True)  # arg1076_1
    buf1077 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1077, (18,), is_leaf=True)  # arg1077_1
    buf1078 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1078, (18,), is_leaf=True)  # arg1078_1
    buf1079 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1079, (18,), is_leaf=True)  # arg1079_1
    buf1080 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1080, (18,), is_leaf=True)  # arg1080_1
    buf1081 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1081, (18, 18, 3, 3), is_leaf=True)  # arg1081_1
    buf1082 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1082, (18,), is_leaf=True)  # arg1082_1
    buf1083 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1083, (18,), is_leaf=True)  # arg1083_1
    buf1084 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1084, (18,), is_leaf=True)  # arg1084_1
    buf1085 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1085, (18,), is_leaf=True)  # arg1085_1
    buf1086 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1086, (36, 36, 3, 3), is_leaf=True)  # arg1086_1
    buf1087 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1087, (36,), is_leaf=True)  # arg1087_1
    buf1088 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1088, (36,), is_leaf=True)  # arg1088_1
    buf1089 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1089, (36,), is_leaf=True)  # arg1089_1
    buf1090 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1090, (36,), is_leaf=True)  # arg1090_1
    buf1091 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1091, (36, 36, 3, 3), is_leaf=True)  # arg1091_1
    buf1092 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1092, (36,), is_leaf=True)  # arg1092_1
    buf1093 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1093, (36,), is_leaf=True)  # arg1093_1
    buf1094 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1094, (36,), is_leaf=True)  # arg1094_1
    buf1095 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1095, (36,), is_leaf=True)  # arg1095_1
    buf1096 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1096, (36, 36, 3, 3), is_leaf=True)  # arg1096_1
    buf1097 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1097, (36,), is_leaf=True)  # arg1097_1
    buf1098 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1098, (36,), is_leaf=True)  # arg1098_1
    buf1099 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1099, (36,), is_leaf=True)  # arg1099_1
    buf1100 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1100, (36,), is_leaf=True)  # arg1100_1
    buf1101 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1101, (36, 36, 3, 3), is_leaf=True)  # arg1101_1
    buf1102 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1102, (36,), is_leaf=True)  # arg1102_1
    buf1103 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1103, (36,), is_leaf=True)  # arg1103_1
    buf1104 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1104, (36,), is_leaf=True)  # arg1104_1
    buf1105 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1105, (36,), is_leaf=True)  # arg1105_1
    buf1106 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1106, (36, 36, 3, 3), is_leaf=True)  # arg1106_1
    buf1107 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1107, (36,), is_leaf=True)  # arg1107_1
    buf1108 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1108, (36,), is_leaf=True)  # arg1108_1
    buf1109 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1109, (36,), is_leaf=True)  # arg1109_1
    buf1110 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1110, (36,), is_leaf=True)  # arg1110_1
    buf1111 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1111, (36, 36, 3, 3), is_leaf=True)  # arg1111_1
    buf1112 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1112, (36,), is_leaf=True)  # arg1112_1
    buf1113 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1113, (36,), is_leaf=True)  # arg1113_1
    buf1114 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1114, (36,), is_leaf=True)  # arg1114_1
    buf1115 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1115, (36,), is_leaf=True)  # arg1115_1
    buf1116 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1116, (36, 36, 3, 3), is_leaf=True)  # arg1116_1
    buf1117 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1117, (36,), is_leaf=True)  # arg1117_1
    buf1118 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1118, (36,), is_leaf=True)  # arg1118_1
    buf1119 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1119, (36,), is_leaf=True)  # arg1119_1
    buf1120 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1120, (36,), is_leaf=True)  # arg1120_1
    buf1121 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1121, (36, 36, 3, 3), is_leaf=True)  # arg1121_1
    buf1122 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1122, (36,), is_leaf=True)  # arg1122_1
    buf1123 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1123, (36,), is_leaf=True)  # arg1123_1
    buf1124 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1124, (36,), is_leaf=True)  # arg1124_1
    buf1125 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1125, (36,), is_leaf=True)  # arg1125_1
    buf1126 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1126, (72, 72, 3, 3), is_leaf=True)  # arg1126_1
    buf1127 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1127, (72,), is_leaf=True)  # arg1127_1
    buf1128 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1128, (72,), is_leaf=True)  # arg1128_1
    buf1129 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1129, (72,), is_leaf=True)  # arg1129_1
    buf1130 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1130, (72,), is_leaf=True)  # arg1130_1
    buf1131 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1131, (72, 72, 3, 3), is_leaf=True)  # arg1131_1
    buf1132 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1132, (72,), is_leaf=True)  # arg1132_1
    buf1133 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1133, (72,), is_leaf=True)  # arg1133_1
    buf1134 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1134, (72,), is_leaf=True)  # arg1134_1
    buf1135 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1135, (72,), is_leaf=True)  # arg1135_1
    buf1136 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1136, (72, 72, 3, 3), is_leaf=True)  # arg1136_1
    buf1137 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1137, (72,), is_leaf=True)  # arg1137_1
    buf1138 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1138, (72,), is_leaf=True)  # arg1138_1
    buf1139 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1139, (72,), is_leaf=True)  # arg1139_1
    buf1140 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1140, (72,), is_leaf=True)  # arg1140_1
    buf1141 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1141, (72, 72, 3, 3), is_leaf=True)  # arg1141_1
    buf1142 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1142, (72,), is_leaf=True)  # arg1142_1
    buf1143 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1143, (72,), is_leaf=True)  # arg1143_1
    buf1144 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1144, (72,), is_leaf=True)  # arg1144_1
    buf1145 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1145, (72,), is_leaf=True)  # arg1145_1
    buf1146 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1146, (72, 72, 3, 3), is_leaf=True)  # arg1146_1
    buf1147 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1147, (72,), is_leaf=True)  # arg1147_1
    buf1148 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1148, (72,), is_leaf=True)  # arg1148_1
    buf1149 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1149, (72,), is_leaf=True)  # arg1149_1
    buf1150 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1150, (72,), is_leaf=True)  # arg1150_1
    buf1151 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1151, (72, 72, 3, 3), is_leaf=True)  # arg1151_1
    buf1152 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1152, (72,), is_leaf=True)  # arg1152_1
    buf1153 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1153, (72,), is_leaf=True)  # arg1153_1
    buf1154 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1154, (72,), is_leaf=True)  # arg1154_1
    buf1155 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1155, (72,), is_leaf=True)  # arg1155_1
    buf1156 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1156, (72, 72, 3, 3), is_leaf=True)  # arg1156_1
    buf1157 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1157, (72,), is_leaf=True)  # arg1157_1
    buf1158 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1158, (72,), is_leaf=True)  # arg1158_1
    buf1159 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1159, (72,), is_leaf=True)  # arg1159_1
    buf1160 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1160, (72,), is_leaf=True)  # arg1160_1
    buf1161 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1161, (72, 72, 3, 3), is_leaf=True)  # arg1161_1
    buf1162 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1162, (72,), is_leaf=True)  # arg1162_1
    buf1163 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1163, (72,), is_leaf=True)  # arg1163_1
    buf1164 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1164, (72,), is_leaf=True)  # arg1164_1
    buf1165 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1165, (72,), is_leaf=True)  # arg1165_1
    buf1166 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1166, (144, 144, 3, 3), is_leaf=True)  # arg1166_1
    buf1167 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1167, (144,), is_leaf=True)  # arg1167_1
    buf1168 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1168, (144,), is_leaf=True)  # arg1168_1
    buf1169 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1169, (144,), is_leaf=True)  # arg1169_1
    buf1170 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1170, (144,), is_leaf=True)  # arg1170_1
    buf1171 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1171, (144, 144, 3, 3), is_leaf=True)  # arg1171_1
    buf1172 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1172, (144,), is_leaf=True)  # arg1172_1
    buf1173 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1173, (144,), is_leaf=True)  # arg1173_1
    buf1174 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1174, (144,), is_leaf=True)  # arg1174_1
    buf1175 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1175, (144,), is_leaf=True)  # arg1175_1
    buf1176 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1176, (144, 144, 3, 3), is_leaf=True)  # arg1176_1
    buf1177 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1177, (144,), is_leaf=True)  # arg1177_1
    buf1178 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1178, (144,), is_leaf=True)  # arg1178_1
    buf1179 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1179, (144,), is_leaf=True)  # arg1179_1
    buf1180 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1180, (144,), is_leaf=True)  # arg1180_1
    buf1181 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1181, (144, 144, 3, 3), is_leaf=True)  # arg1181_1
    buf1182 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1182, (144,), is_leaf=True)  # arg1182_1
    buf1183 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1183, (144,), is_leaf=True)  # arg1183_1
    buf1184 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1184, (144,), is_leaf=True)  # arg1184_1
    buf1185 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1185, (144,), is_leaf=True)  # arg1185_1
    buf1186 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1186, (144, 144, 3, 3), is_leaf=True)  # arg1186_1
    buf1187 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1187, (144,), is_leaf=True)  # arg1187_1
    buf1188 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1188, (144,), is_leaf=True)  # arg1188_1
    buf1189 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1189, (144,), is_leaf=True)  # arg1189_1
    buf1190 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1190, (144,), is_leaf=True)  # arg1190_1
    buf1191 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1191, (144, 144, 3, 3), is_leaf=True)  # arg1191_1
    buf1192 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1192, (144,), is_leaf=True)  # arg1192_1
    buf1193 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1193, (144,), is_leaf=True)  # arg1193_1
    buf1194 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1194, (144,), is_leaf=True)  # arg1194_1
    buf1195 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1195, (144,), is_leaf=True)  # arg1195_1
    buf1196 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1196, (144, 144, 3, 3), is_leaf=True)  # arg1196_1
    buf1197 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1197, (144,), is_leaf=True)  # arg1197_1
    buf1198 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1198, (144,), is_leaf=True)  # arg1198_1
    buf1199 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1199, (144,), is_leaf=True)  # arg1199_1
    buf1200 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1200, (144,), is_leaf=True)  # arg1200_1
    buf1201 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1201, (144, 144, 3, 3), is_leaf=True)  # arg1201_1
    buf1202 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1202, (144,), is_leaf=True)  # arg1202_1
    buf1203 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1203, (144,), is_leaf=True)  # arg1203_1
    buf1204 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1204, (144,), is_leaf=True)  # arg1204_1
    buf1205 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1205, (144,), is_leaf=True)  # arg1205_1
    buf1206 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf1206, (18, 36, 1, 1), is_leaf=True)  # arg1206_1
    buf1207 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1207, (18,), is_leaf=True)  # arg1207_1
    buf1208 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1208, (18,), is_leaf=True)  # arg1208_1
    buf1209 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1209, (18,), is_leaf=True)  # arg1209_1
    buf1210 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1210, (18,), is_leaf=True)  # arg1210_1
    buf1211 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf1211, (18, 72, 1, 1), is_leaf=True)  # arg1211_1
    buf1212 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1212, (18,), is_leaf=True)  # arg1212_1
    buf1213 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1213, (18,), is_leaf=True)  # arg1213_1
    buf1214 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1214, (18,), is_leaf=True)  # arg1214_1
    buf1215 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1215, (18,), is_leaf=True)  # arg1215_1
    buf1216 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf1216, (18, 144, 1, 1), is_leaf=True)  # arg1216_1
    buf1217 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1217, (18,), is_leaf=True)  # arg1217_1
    buf1218 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1218, (18,), is_leaf=True)  # arg1218_1
    buf1219 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1219, (18,), is_leaf=True)  # arg1219_1
    buf1220 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1220, (18,), is_leaf=True)  # arg1220_1
    buf1221 = reader.storage(None, 23328, device=device(type='cuda', index=0))
    reader.tensor(buf1221, (36, 18, 3, 3), is_leaf=True)  # arg1221_1
    buf1222 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1222, (36,), is_leaf=True)  # arg1222_1
    buf1223 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1223, (36,), is_leaf=True)  # arg1223_1
    buf1224 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1224, (36,), is_leaf=True)  # arg1224_1
    buf1225 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1225, (36,), is_leaf=True)  # arg1225_1
    buf1226 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf1226, (36, 72, 1, 1), is_leaf=True)  # arg1226_1
    buf1227 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1227, (36,), is_leaf=True)  # arg1227_1
    buf1228 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1228, (36,), is_leaf=True)  # arg1228_1
    buf1229 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1229, (36,), is_leaf=True)  # arg1229_1
    buf1230 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1230, (36,), is_leaf=True)  # arg1230_1
    buf1231 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf1231, (36, 144, 1, 1), is_leaf=True)  # arg1231_1
    buf1232 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1232, (36,), is_leaf=True)  # arg1232_1
    buf1233 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1233, (36,), is_leaf=True)  # arg1233_1
    buf1234 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1234, (36,), is_leaf=True)  # arg1234_1
    buf1235 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1235, (36,), is_leaf=True)  # arg1235_1
    buf1236 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1236, (18, 18, 3, 3), is_leaf=True)  # arg1236_1
    buf1237 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1237, (18,), is_leaf=True)  # arg1237_1
    buf1238 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1238, (18,), is_leaf=True)  # arg1238_1
    buf1239 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1239, (18,), is_leaf=True)  # arg1239_1
    buf1240 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1240, (18,), is_leaf=True)  # arg1240_1
    buf1241 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1241, (72, 18, 3, 3), is_leaf=True)  # arg1241_1
    buf1242 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1242, (72,), is_leaf=True)  # arg1242_1
    buf1243 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1243, (72,), is_leaf=True)  # arg1243_1
    buf1244 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1244, (72,), is_leaf=True)  # arg1244_1
    buf1245 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1245, (72,), is_leaf=True)  # arg1245_1
    buf1246 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf1246, (72, 36, 3, 3), is_leaf=True)  # arg1246_1
    buf1247 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1247, (72,), is_leaf=True)  # arg1247_1
    buf1248 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1248, (72,), is_leaf=True)  # arg1248_1
    buf1249 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1249, (72,), is_leaf=True)  # arg1249_1
    buf1250 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1250, (72,), is_leaf=True)  # arg1250_1
    buf1251 = reader.storage(None, 41472, device=device(type='cuda', index=0))
    reader.tensor(buf1251, (72, 144, 1, 1), is_leaf=True)  # arg1251_1
    buf1252 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1252, (72,), is_leaf=True)  # arg1252_1
    buf1253 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1253, (72,), is_leaf=True)  # arg1253_1
    buf1254 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1254, (72,), is_leaf=True)  # arg1254_1
    buf1255 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1255, (72,), is_leaf=True)  # arg1255_1
    buf1256 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1256, (18, 18, 3, 3), is_leaf=True)  # arg1256_1
    buf1257 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1257, (18,), is_leaf=True)  # arg1257_1
    buf1258 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1258, (18,), is_leaf=True)  # arg1258_1
    buf1259 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1259, (18,), is_leaf=True)  # arg1259_1
    buf1260 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1260, (18,), is_leaf=True)  # arg1260_1
    buf1261 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1261, (18, 18, 3, 3), is_leaf=True)  # arg1261_1
    buf1262 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1262, (18,), is_leaf=True)  # arg1262_1
    buf1263 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1263, (18,), is_leaf=True)  # arg1263_1
    buf1264 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1264, (18,), is_leaf=True)  # arg1264_1
    buf1265 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1265, (18,), is_leaf=True)  # arg1265_1
    buf1266 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf1266, (144, 18, 3, 3), is_leaf=True)  # arg1266_1
    buf1267 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1267, (144,), is_leaf=True)  # arg1267_1
    buf1268 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1268, (144,), is_leaf=True)  # arg1268_1
    buf1269 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1269, (144,), is_leaf=True)  # arg1269_1
    buf1270 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1270, (144,), is_leaf=True)  # arg1270_1
    buf1271 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1271, (36, 36, 3, 3), is_leaf=True)  # arg1271_1
    buf1272 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1272, (36,), is_leaf=True)  # arg1272_1
    buf1273 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1273, (36,), is_leaf=True)  # arg1273_1
    buf1274 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1274, (36,), is_leaf=True)  # arg1274_1
    buf1275 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1275, (36,), is_leaf=True)  # arg1275_1
    buf1276 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1276, (144, 36, 3, 3), is_leaf=True)  # arg1276_1
    buf1277 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1277, (144,), is_leaf=True)  # arg1277_1
    buf1278 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1278, (144,), is_leaf=True)  # arg1278_1
    buf1279 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1279, (144,), is_leaf=True)  # arg1279_1
    buf1280 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1280, (144,), is_leaf=True)  # arg1280_1
    buf1281 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf1281, (144, 72, 3, 3), is_leaf=True)  # arg1281_1
    buf1282 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1282, (144,), is_leaf=True)  # arg1282_1
    buf1283 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1283, (144,), is_leaf=True)  # arg1283_1
    buf1284 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1284, (144,), is_leaf=True)  # arg1284_1
    buf1285 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1285, (144,), is_leaf=True)  # arg1285_1
    buf1286 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1286, (18, 18, 3, 3), is_leaf=True)  # arg1286_1
    buf1287 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1287, (18,), is_leaf=True)  # arg1287_1
    buf1288 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1288, (18,), is_leaf=True)  # arg1288_1
    buf1289 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1289, (18,), is_leaf=True)  # arg1289_1
    buf1290 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1290, (18,), is_leaf=True)  # arg1290_1
    buf1291 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1291, (18, 18, 3, 3), is_leaf=True)  # arg1291_1
    buf1292 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1292, (18,), is_leaf=True)  # arg1292_1
    buf1293 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1293, (18,), is_leaf=True)  # arg1293_1
    buf1294 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1294, (18,), is_leaf=True)  # arg1294_1
    buf1295 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1295, (18,), is_leaf=True)  # arg1295_1
    buf1296 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1296, (18, 18, 3, 3), is_leaf=True)  # arg1296_1
    buf1297 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1297, (18,), is_leaf=True)  # arg1297_1
    buf1298 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1298, (18,), is_leaf=True)  # arg1298_1
    buf1299 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1299, (18,), is_leaf=True)  # arg1299_1
    buf1300 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1300, (18,), is_leaf=True)  # arg1300_1
    buf1301 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1301, (18, 18, 3, 3), is_leaf=True)  # arg1301_1
    buf1302 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1302, (18,), is_leaf=True)  # arg1302_1
    buf1303 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1303, (18,), is_leaf=True)  # arg1303_1
    buf1304 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1304, (18,), is_leaf=True)  # arg1304_1
    buf1305 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1305, (18,), is_leaf=True)  # arg1305_1
    buf1306 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1306, (18, 18, 3, 3), is_leaf=True)  # arg1306_1
    buf1307 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1307, (18,), is_leaf=True)  # arg1307_1
    buf1308 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1308, (18,), is_leaf=True)  # arg1308_1
    buf1309 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1309, (18,), is_leaf=True)  # arg1309_1
    buf1310 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1310, (18,), is_leaf=True)  # arg1310_1
    buf1311 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1311, (18, 18, 3, 3), is_leaf=True)  # arg1311_1
    buf1312 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1312, (18,), is_leaf=True)  # arg1312_1
    buf1313 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1313, (18,), is_leaf=True)  # arg1313_1
    buf1314 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1314, (18,), is_leaf=True)  # arg1314_1
    buf1315 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1315, (18,), is_leaf=True)  # arg1315_1
    buf1316 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1316, (18, 18, 3, 3), is_leaf=True)  # arg1316_1
    buf1317 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1317, (18,), is_leaf=True)  # arg1317_1
    buf1318 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1318, (18,), is_leaf=True)  # arg1318_1
    buf1319 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1319, (18,), is_leaf=True)  # arg1319_1
    buf1320 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1320, (18,), is_leaf=True)  # arg1320_1
    buf1321 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1321, (18, 18, 3, 3), is_leaf=True)  # arg1321_1
    buf1322 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1322, (18,), is_leaf=True)  # arg1322_1
    buf1323 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1323, (18,), is_leaf=True)  # arg1323_1
    buf1324 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1324, (18,), is_leaf=True)  # arg1324_1
    buf1325 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1325, (18,), is_leaf=True)  # arg1325_1
    buf1326 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1326, (36, 36, 3, 3), is_leaf=True)  # arg1326_1
    buf1327 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1327, (36,), is_leaf=True)  # arg1327_1
    buf1328 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1328, (36,), is_leaf=True)  # arg1328_1
    buf1329 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1329, (36,), is_leaf=True)  # arg1329_1
    buf1330 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1330, (36,), is_leaf=True)  # arg1330_1
    buf1331 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1331, (36, 36, 3, 3), is_leaf=True)  # arg1331_1
    buf1332 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1332, (36,), is_leaf=True)  # arg1332_1
    buf1333 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1333, (36,), is_leaf=True)  # arg1333_1
    buf1334 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1334, (36,), is_leaf=True)  # arg1334_1
    buf1335 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1335, (36,), is_leaf=True)  # arg1335_1
    buf1336 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1336, (36, 36, 3, 3), is_leaf=True)  # arg1336_1
    buf1337 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1337, (36,), is_leaf=True)  # arg1337_1
    buf1338 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1338, (36,), is_leaf=True)  # arg1338_1
    buf1339 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1339, (36,), is_leaf=True)  # arg1339_1
    buf1340 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1340, (36,), is_leaf=True)  # arg1340_1
    buf1341 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1341, (36, 36, 3, 3), is_leaf=True)  # arg1341_1
    buf1342 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1342, (36,), is_leaf=True)  # arg1342_1
    buf1343 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1343, (36,), is_leaf=True)  # arg1343_1
    buf1344 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1344, (36,), is_leaf=True)  # arg1344_1
    buf1345 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1345, (36,), is_leaf=True)  # arg1345_1
    buf1346 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1346, (36, 36, 3, 3), is_leaf=True)  # arg1346_1
    buf1347 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1347, (36,), is_leaf=True)  # arg1347_1
    buf1348 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1348, (36,), is_leaf=True)  # arg1348_1
    buf1349 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1349, (36,), is_leaf=True)  # arg1349_1
    buf1350 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1350, (36,), is_leaf=True)  # arg1350_1
    buf1351 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1351, (36, 36, 3, 3), is_leaf=True)  # arg1351_1
    buf1352 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1352, (36,), is_leaf=True)  # arg1352_1
    buf1353 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1353, (36,), is_leaf=True)  # arg1353_1
    buf1354 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1354, (36,), is_leaf=True)  # arg1354_1
    buf1355 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1355, (36,), is_leaf=True)  # arg1355_1
    buf1356 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1356, (36, 36, 3, 3), is_leaf=True)  # arg1356_1
    buf1357 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1357, (36,), is_leaf=True)  # arg1357_1
    buf1358 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1358, (36,), is_leaf=True)  # arg1358_1
    buf1359 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1359, (36,), is_leaf=True)  # arg1359_1
    buf1360 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1360, (36,), is_leaf=True)  # arg1360_1
    buf1361 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1361, (36, 36, 3, 3), is_leaf=True)  # arg1361_1
    buf1362 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1362, (36,), is_leaf=True)  # arg1362_1
    buf1363 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1363, (36,), is_leaf=True)  # arg1363_1
    buf1364 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1364, (36,), is_leaf=True)  # arg1364_1
    buf1365 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1365, (36,), is_leaf=True)  # arg1365_1
    buf1366 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1366, (72, 72, 3, 3), is_leaf=True)  # arg1366_1
    buf1367 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1367, (72,), is_leaf=True)  # arg1367_1
    buf1368 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1368, (72,), is_leaf=True)  # arg1368_1
    buf1369 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1369, (72,), is_leaf=True)  # arg1369_1
    buf1370 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1370, (72,), is_leaf=True)  # arg1370_1
    buf1371 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1371, (72, 72, 3, 3), is_leaf=True)  # arg1371_1
    buf1372 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1372, (72,), is_leaf=True)  # arg1372_1
    buf1373 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1373, (72,), is_leaf=True)  # arg1373_1
    buf1374 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1374, (72,), is_leaf=True)  # arg1374_1
    buf1375 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1375, (72,), is_leaf=True)  # arg1375_1
    buf1376 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1376, (72, 72, 3, 3), is_leaf=True)  # arg1376_1
    buf1377 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1377, (72,), is_leaf=True)  # arg1377_1
    buf1378 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1378, (72,), is_leaf=True)  # arg1378_1
    buf1379 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1379, (72,), is_leaf=True)  # arg1379_1
    buf1380 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1380, (72,), is_leaf=True)  # arg1380_1
    buf1381 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1381, (72, 72, 3, 3), is_leaf=True)  # arg1381_1
    buf1382 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1382, (72,), is_leaf=True)  # arg1382_1
    buf1383 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1383, (72,), is_leaf=True)  # arg1383_1
    buf1384 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1384, (72,), is_leaf=True)  # arg1384_1
    buf1385 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1385, (72,), is_leaf=True)  # arg1385_1
    buf1386 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1386, (72, 72, 3, 3), is_leaf=True)  # arg1386_1
    buf1387 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1387, (72,), is_leaf=True)  # arg1387_1
    buf1388 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1388, (72,), is_leaf=True)  # arg1388_1
    buf1389 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1389, (72,), is_leaf=True)  # arg1389_1
    buf1390 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1390, (72,), is_leaf=True)  # arg1390_1
    buf1391 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1391, (72, 72, 3, 3), is_leaf=True)  # arg1391_1
    buf1392 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1392, (72,), is_leaf=True)  # arg1392_1
    buf1393 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1393, (72,), is_leaf=True)  # arg1393_1
    buf1394 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1394, (72,), is_leaf=True)  # arg1394_1
    buf1395 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1395, (72,), is_leaf=True)  # arg1395_1
    buf1396 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1396, (72, 72, 3, 3), is_leaf=True)  # arg1396_1
    buf1397 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1397, (72,), is_leaf=True)  # arg1397_1
    buf1398 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1398, (72,), is_leaf=True)  # arg1398_1
    buf1399 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1399, (72,), is_leaf=True)  # arg1399_1
    buf1400 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1400, (72,), is_leaf=True)  # arg1400_1
    buf1401 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1401, (72, 72, 3, 3), is_leaf=True)  # arg1401_1
    buf1402 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1402, (72,), is_leaf=True)  # arg1402_1
    buf1403 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1403, (72,), is_leaf=True)  # arg1403_1
    buf1404 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1404, (72,), is_leaf=True)  # arg1404_1
    buf1405 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1405, (72,), is_leaf=True)  # arg1405_1
    buf1406 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1406, (144, 144, 3, 3), is_leaf=True)  # arg1406_1
    buf1407 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1407, (144,), is_leaf=True)  # arg1407_1
    buf1408 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1408, (144,), is_leaf=True)  # arg1408_1
    buf1409 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1409, (144,), is_leaf=True)  # arg1409_1
    buf1410 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1410, (144,), is_leaf=True)  # arg1410_1
    buf1411 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1411, (144, 144, 3, 3), is_leaf=True)  # arg1411_1
    buf1412 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1412, (144,), is_leaf=True)  # arg1412_1
    buf1413 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1413, (144,), is_leaf=True)  # arg1413_1
    buf1414 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1414, (144,), is_leaf=True)  # arg1414_1
    buf1415 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1415, (144,), is_leaf=True)  # arg1415_1
    buf1416 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1416, (144, 144, 3, 3), is_leaf=True)  # arg1416_1
    buf1417 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1417, (144,), is_leaf=True)  # arg1417_1
    buf1418 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1418, (144,), is_leaf=True)  # arg1418_1
    buf1419 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1419, (144,), is_leaf=True)  # arg1419_1
    buf1420 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1420, (144,), is_leaf=True)  # arg1420_1
    buf1421 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1421, (144, 144, 3, 3), is_leaf=True)  # arg1421_1
    buf1422 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1422, (144,), is_leaf=True)  # arg1422_1
    buf1423 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1423, (144,), is_leaf=True)  # arg1423_1
    buf1424 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1424, (144,), is_leaf=True)  # arg1424_1
    buf1425 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1425, (144,), is_leaf=True)  # arg1425_1
    buf1426 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1426, (144, 144, 3, 3), is_leaf=True)  # arg1426_1
    buf1427 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1427, (144,), is_leaf=True)  # arg1427_1
    buf1428 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1428, (144,), is_leaf=True)  # arg1428_1
    buf1429 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1429, (144,), is_leaf=True)  # arg1429_1
    buf1430 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1430, (144,), is_leaf=True)  # arg1430_1
    buf1431 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1431, (144, 144, 3, 3), is_leaf=True)  # arg1431_1
    buf1432 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1432, (144,), is_leaf=True)  # arg1432_1
    buf1433 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1433, (144,), is_leaf=True)  # arg1433_1
    buf1434 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1434, (144,), is_leaf=True)  # arg1434_1
    buf1435 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1435, (144,), is_leaf=True)  # arg1435_1
    buf1436 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1436, (144, 144, 3, 3), is_leaf=True)  # arg1436_1
    buf1437 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1437, (144,), is_leaf=True)  # arg1437_1
    buf1438 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1438, (144,), is_leaf=True)  # arg1438_1
    buf1439 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1439, (144,), is_leaf=True)  # arg1439_1
    buf1440 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1440, (144,), is_leaf=True)  # arg1440_1
    buf1441 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf1441, (144, 144, 3, 3), is_leaf=True)  # arg1441_1
    buf1442 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1442, (144,), is_leaf=True)  # arg1442_1
    buf1443 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1443, (144,), is_leaf=True)  # arg1443_1
    buf1444 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1444, (144,), is_leaf=True)  # arg1444_1
    buf1445 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1445, (144,), is_leaf=True)  # arg1445_1
    buf1446 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf1446, (18, 36, 1, 1), is_leaf=True)  # arg1446_1
    buf1447 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1447, (18,), is_leaf=True)  # arg1447_1
    buf1448 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1448, (18,), is_leaf=True)  # arg1448_1
    buf1449 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1449, (18,), is_leaf=True)  # arg1449_1
    buf1450 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1450, (18,), is_leaf=True)  # arg1450_1
    buf1451 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf1451, (18, 72, 1, 1), is_leaf=True)  # arg1451_1
    buf1452 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1452, (18,), is_leaf=True)  # arg1452_1
    buf1453 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1453, (18,), is_leaf=True)  # arg1453_1
    buf1454 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1454, (18,), is_leaf=True)  # arg1454_1
    buf1455 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1455, (18,), is_leaf=True)  # arg1455_1
    buf1456 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf1456, (18, 144, 1, 1), is_leaf=True)  # arg1456_1
    buf1457 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1457, (18,), is_leaf=True)  # arg1457_1
    buf1458 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1458, (18,), is_leaf=True)  # arg1458_1
    buf1459 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1459, (18,), is_leaf=True)  # arg1459_1
    buf1460 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1460, (18,), is_leaf=True)  # arg1460_1
    buf1461 = reader.storage(None, 23328, device=device(type='cuda', index=0))
    reader.tensor(buf1461, (36, 18, 3, 3), is_leaf=True)  # arg1461_1
    buf1462 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1462, (36,), is_leaf=True)  # arg1462_1
    buf1463 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1463, (36,), is_leaf=True)  # arg1463_1
    buf1464 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1464, (36,), is_leaf=True)  # arg1464_1
    buf1465 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1465, (36,), is_leaf=True)  # arg1465_1
    buf1466 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf1466, (36, 72, 1, 1), is_leaf=True)  # arg1466_1
    buf1467 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1467, (36,), is_leaf=True)  # arg1467_1
    buf1468 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1468, (36,), is_leaf=True)  # arg1468_1
    buf1469 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1469, (36,), is_leaf=True)  # arg1469_1
    buf1470 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1470, (36,), is_leaf=True)  # arg1470_1
    buf1471 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf1471, (36, 144, 1, 1), is_leaf=True)  # arg1471_1
    buf1472 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1472, (36,), is_leaf=True)  # arg1472_1
    buf1473 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1473, (36,), is_leaf=True)  # arg1473_1
    buf1474 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1474, (36,), is_leaf=True)  # arg1474_1
    buf1475 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1475, (36,), is_leaf=True)  # arg1475_1
    buf1476 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1476, (18, 18, 3, 3), is_leaf=True)  # arg1476_1
    buf1477 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1477, (18,), is_leaf=True)  # arg1477_1
    buf1478 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1478, (18,), is_leaf=True)  # arg1478_1
    buf1479 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1479, (18,), is_leaf=True)  # arg1479_1
    buf1480 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1480, (18,), is_leaf=True)  # arg1480_1
    buf1481 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1481, (72, 18, 3, 3), is_leaf=True)  # arg1481_1
    buf1482 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1482, (72,), is_leaf=True)  # arg1482_1
    buf1483 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1483, (72,), is_leaf=True)  # arg1483_1
    buf1484 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1484, (72,), is_leaf=True)  # arg1484_1
    buf1485 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1485, (72,), is_leaf=True)  # arg1485_1
    buf1486 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf1486, (72, 36, 3, 3), is_leaf=True)  # arg1486_1
    buf1487 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1487, (72,), is_leaf=True)  # arg1487_1
    buf1488 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1488, (72,), is_leaf=True)  # arg1488_1
    buf1489 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1489, (72,), is_leaf=True)  # arg1489_1
    buf1490 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1490, (72,), is_leaf=True)  # arg1490_1
    buf1491 = reader.storage(None, 41472, device=device(type='cuda', index=0))
    reader.tensor(buf1491, (72, 144, 1, 1), is_leaf=True)  # arg1491_1
    buf1492 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1492, (72,), is_leaf=True)  # arg1492_1
    buf1493 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1493, (72,), is_leaf=True)  # arg1493_1
    buf1494 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1494, (72,), is_leaf=True)  # arg1494_1
    buf1495 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf1495, (72,), is_leaf=True)  # arg1495_1
    buf1496 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1496, (18, 18, 3, 3), is_leaf=True)  # arg1496_1
    buf1497 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1497, (18,), is_leaf=True)  # arg1497_1
    buf1498 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1498, (18,), is_leaf=True)  # arg1498_1
    buf1499 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1499, (18,), is_leaf=True)  # arg1499_1
    buf1500 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1500, (18,), is_leaf=True)  # arg1500_1
    buf1501 = reader.storage(None, 11664, device=device(type='cuda', index=0))
    reader.tensor(buf1501, (18, 18, 3, 3), is_leaf=True)  # arg1501_1
    buf1502 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1502, (18,), is_leaf=True)  # arg1502_1
    buf1503 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1503, (18,), is_leaf=True)  # arg1503_1
    buf1504 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1504, (18,), is_leaf=True)  # arg1504_1
    buf1505 = reader.storage(None, 72, device=device(type='cuda', index=0))
    reader.tensor(buf1505, (18,), is_leaf=True)  # arg1505_1
    buf1506 = reader.storage(None, 93312, device=device(type='cuda', index=0))
    reader.tensor(buf1506, (144, 18, 3, 3), is_leaf=True)  # arg1506_1
    buf1507 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1507, (144,), is_leaf=True)  # arg1507_1
    buf1508 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1508, (144,), is_leaf=True)  # arg1508_1
    buf1509 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1509, (144,), is_leaf=True)  # arg1509_1
    buf1510 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1510, (144,), is_leaf=True)  # arg1510_1
    buf1511 = reader.storage(None, 46656, device=device(type='cuda', index=0))
    reader.tensor(buf1511, (36, 36, 3, 3), is_leaf=True)  # arg1511_1
    buf1512 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1512, (36,), is_leaf=True)  # arg1512_1
    buf1513 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1513, (36,), is_leaf=True)  # arg1513_1
    buf1514 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1514, (36,), is_leaf=True)  # arg1514_1
    buf1515 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf1515, (36,), is_leaf=True)  # arg1515_1
    buf1516 = reader.storage(None, 186624, device=device(type='cuda', index=0))
    reader.tensor(buf1516, (144, 36, 3, 3), is_leaf=True)  # arg1516_1
    buf1517 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1517, (144,), is_leaf=True)  # arg1517_1
    buf1518 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1518, (144,), is_leaf=True)  # arg1518_1
    buf1519 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1519, (144,), is_leaf=True)  # arg1519_1
    buf1520 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1520, (144,), is_leaf=True)  # arg1520_1
    buf1521 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf1521, (144, 72, 3, 3), is_leaf=True)  # arg1521_1
    buf1522 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1522, (144,), is_leaf=True)  # arg1522_1
    buf1523 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1523, (144,), is_leaf=True)  # arg1523_1
    buf1524 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1524, (144,), is_leaf=True)  # arg1524_1
    buf1525 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf1525, (144,), is_leaf=True)  # arg1525_1
    buf1526 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf1526, (32, 18, 1, 1), is_leaf=True)  # arg1526_1
    buf1527 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1527, (32,), is_leaf=True)  # arg1527_1
    buf1528 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1528, (32,), is_leaf=True)  # arg1528_1
    buf1529 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1529, (32,), is_leaf=True)  # arg1529_1
    buf1530 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1530, (32,), is_leaf=True)  # arg1530_1
    buf1531 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf1531, (32, 32, 3, 3), is_leaf=True)  # arg1531_1
    buf1532 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1532, (32,), is_leaf=True)  # arg1532_1
    buf1533 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1533, (32,), is_leaf=True)  # arg1533_1
    buf1534 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1534, (32,), is_leaf=True)  # arg1534_1
    buf1535 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf1535, (32,), is_leaf=True)  # arg1535_1
    buf1536 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf1536, (128, 32, 1, 1), is_leaf=True)  # arg1536_1
    buf1537 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1537, (128,), is_leaf=True)  # arg1537_1
    buf1538 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1538, (128,), is_leaf=True)  # arg1538_1
    buf1539 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1539, (128,), is_leaf=True)  # arg1539_1
    buf1540 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1540, (128,), is_leaf=True)  # arg1540_1
    buf1541 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf1541, (128, 18, 1, 1), is_leaf=True)  # arg1541_1
    buf1542 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1542, (128,), is_leaf=True)  # arg1542_1
    buf1543 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1543, (128,), is_leaf=True)  # arg1543_1
    buf1544 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1544, (128,), is_leaf=True)  # arg1544_1
    buf1545 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1545, (128,), is_leaf=True)  # arg1545_1
    buf1546 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf1546, (64, 36, 1, 1), is_leaf=True)  # arg1546_1
    buf1547 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1547, (64,), is_leaf=True)  # arg1547_1
    buf1548 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1548, (64,), is_leaf=True)  # arg1548_1
    buf1549 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1549, (64,), is_leaf=True)  # arg1549_1
    buf1550 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1550, (64,), is_leaf=True)  # arg1550_1
    buf1551 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf1551, (64, 64, 3, 3), is_leaf=True)  # arg1551_1
    buf1552 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1552, (64,), is_leaf=True)  # arg1552_1
    buf1553 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1553, (64,), is_leaf=True)  # arg1553_1
    buf1554 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1554, (64,), is_leaf=True)  # arg1554_1
    buf1555 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1555, (64,), is_leaf=True)  # arg1555_1
    buf1556 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf1556, (256, 64, 1, 1), is_leaf=True)  # arg1556_1
    buf1557 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1557, (256,), is_leaf=True)  # arg1557_1
    buf1558 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1558, (256,), is_leaf=True)  # arg1558_1
    buf1559 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1559, (256,), is_leaf=True)  # arg1559_1
    buf1560 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1560, (256,), is_leaf=True)  # arg1560_1
    buf1561 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf1561, (256, 36, 1, 1), is_leaf=True)  # arg1561_1
    buf1562 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1562, (256,), is_leaf=True)  # arg1562_1
    buf1563 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1563, (256,), is_leaf=True)  # arg1563_1
    buf1564 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1564, (256,), is_leaf=True)  # arg1564_1
    buf1565 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1565, (256,), is_leaf=True)  # arg1565_1
    buf1566 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf1566, (256, 128, 3, 3), is_leaf=True)  # arg1566_1
    buf1567 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1567, (256,), is_leaf=True)  # arg1567_1
    buf1568 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1568, (256,), is_leaf=True)  # arg1568_1
    buf1569 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1569, (256,), is_leaf=True)  # arg1569_1
    buf1570 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1570, (256,), is_leaf=True)  # arg1570_1
    buf1571 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1571, (256,), is_leaf=True)  # arg1571_1
    buf1572 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf1572, (128, 72, 1, 1), is_leaf=True)  # arg1572_1
    buf1573 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1573, (128,), is_leaf=True)  # arg1573_1
    buf1574 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1574, (128,), is_leaf=True)  # arg1574_1
    buf1575 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1575, (128,), is_leaf=True)  # arg1575_1
    buf1576 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1576, (128,), is_leaf=True)  # arg1576_1
    buf1577 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf1577, (128, 128, 3, 3), is_leaf=True)  # arg1577_1
    buf1578 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1578, (128,), is_leaf=True)  # arg1578_1
    buf1579 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1579, (128,), is_leaf=True)  # arg1579_1
    buf1580 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1580, (128,), is_leaf=True)  # arg1580_1
    buf1581 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1581, (128,), is_leaf=True)  # arg1581_1
    buf1582 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1582, (512, 128, 1, 1), is_leaf=True)  # arg1582_1
    buf1583 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1583, (512,), is_leaf=True)  # arg1583_1
    buf1584 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1584, (512,), is_leaf=True)  # arg1584_1
    buf1585 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1585, (512,), is_leaf=True)  # arg1585_1
    buf1586 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1586, (512,), is_leaf=True)  # arg1586_1
    buf1587 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf1587, (512, 72, 1, 1), is_leaf=True)  # arg1587_1
    buf1588 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1588, (512,), is_leaf=True)  # arg1588_1
    buf1589 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1589, (512,), is_leaf=True)  # arg1589_1
    buf1590 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1590, (512,), is_leaf=True)  # arg1590_1
    buf1591 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1591, (512,), is_leaf=True)  # arg1591_1
    buf1592 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf1592, (512, 256, 3, 3), is_leaf=True)  # arg1592_1
    buf1593 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1593, (512,), is_leaf=True)  # arg1593_1
    buf1594 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1594, (512,), is_leaf=True)  # arg1594_1
    buf1595 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1595, (512,), is_leaf=True)  # arg1595_1
    buf1596 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1596, (512,), is_leaf=True)  # arg1596_1
    buf1597 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf1597, (512,), is_leaf=True)  # arg1597_1
    buf1598 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf1598, (256, 144, 1, 1), is_leaf=True)  # arg1598_1
    buf1599 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1599, (256,), is_leaf=True)  # arg1599_1
    buf1600 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1600, (256,), is_leaf=True)  # arg1600_1
    buf1601 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1601, (256,), is_leaf=True)  # arg1601_1
    buf1602 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1602, (256,), is_leaf=True)  # arg1602_1
    buf1603 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf1603, (256, 256, 3, 3), is_leaf=True)  # arg1603_1
    buf1604 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1604, (256,), is_leaf=True)  # arg1604_1
    buf1605 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1605, (256,), is_leaf=True)  # arg1605_1
    buf1606 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1606, (256,), is_leaf=True)  # arg1606_1
    buf1607 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf1607, (256,), is_leaf=True)  # arg1607_1
    buf1608 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf1608, (1024, 256, 1, 1), is_leaf=True)  # arg1608_1
    buf1609 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1609, (1024,), is_leaf=True)  # arg1609_1
    buf1610 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1610, (1024,), is_leaf=True)  # arg1610_1
    buf1611 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1611, (1024,), is_leaf=True)  # arg1611_1
    buf1612 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1612, (1024,), is_leaf=True)  # arg1612_1
    buf1613 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf1613, (1024, 144, 1, 1), is_leaf=True)  # arg1613_1
    buf1614 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1614, (1024,), is_leaf=True)  # arg1614_1
    buf1615 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1615, (1024,), is_leaf=True)  # arg1615_1
    buf1616 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1616, (1024,), is_leaf=True)  # arg1616_1
    buf1617 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1617, (1024,), is_leaf=True)  # arg1617_1
    buf1618 = reader.storage(None, 18874368, device=device(type='cuda', index=0))
    reader.tensor(buf1618, (1024, 512, 3, 3), is_leaf=True)  # arg1618_1
    buf1619 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1619, (1024,), is_leaf=True)  # arg1619_1
    buf1620 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1620, (1024,), is_leaf=True)  # arg1620_1
    buf1621 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1621, (1024,), is_leaf=True)  # arg1621_1
    buf1622 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1622, (1024,), is_leaf=True)  # arg1622_1
    buf1623 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1623, (1024,), is_leaf=True)  # arg1623_1
    buf1624 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf1624, (2048, 1024, 1, 1), is_leaf=True)  # arg1624_1
    buf1625 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1625, (2048,), is_leaf=True)  # arg1625_1
    buf1626 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1626, (2048,), is_leaf=True)  # arg1626_1
    buf1627 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1627, (2048,), is_leaf=True)  # arg1627_1
    buf1628 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1628, (2048,), is_leaf=True)  # arg1628_1
    buf1629 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1629, (2048,), is_leaf=True)  # arg1629_1
    buf1630 = reader.storage(None, 8192000, device=device(type='cuda', index=0))
    reader.tensor(buf1630, (1000, 2048), is_leaf=True)  # arg1630_1
    buf1631 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf1631, (1000,), is_leaf=True)  # arg1631_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)