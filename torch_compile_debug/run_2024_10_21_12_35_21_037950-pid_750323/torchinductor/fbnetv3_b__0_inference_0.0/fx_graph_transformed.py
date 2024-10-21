class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[8, 3, 256, 256]", arg2_1: "f32[16]", arg3_1: "f32[16]", arg4_1: "f32[16]", arg5_1: "f32[16]", arg6_1: "f32[16, 1, 3, 3]", arg7_1: "f32[16]", arg8_1: "f32[16]", arg9_1: "f32[16]", arg10_1: "f32[16]", arg11_1: "f32[16, 16, 1, 1]", arg12_1: "f32[16]", arg13_1: "f32[16]", arg14_1: "f32[16]", arg15_1: "f32[16]", arg16_1: "f32[16, 1, 3, 3]", arg17_1: "f32[16]", arg18_1: "f32[16]", arg19_1: "f32[16]", arg20_1: "f32[16]", arg21_1: "f32[16, 16, 1, 1]", arg22_1: "f32[16]", arg23_1: "f32[16]", arg24_1: "f32[16]", arg25_1: "f32[16]", arg26_1: "f32[64, 16, 1, 1]", arg27_1: "f32[64]", arg28_1: "f32[64]", arg29_1: "f32[64]", arg30_1: "f32[64]", arg31_1: "f32[64, 1, 5, 5]", arg32_1: "f32[64]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[24, 64, 1, 1]", arg37_1: "f32[24]", arg38_1: "f32[24]", arg39_1: "f32[24]", arg40_1: "f32[24]", arg41_1: "f32[48, 24, 1, 1]", arg42_1: "f32[48]", arg43_1: "f32[48]", arg44_1: "f32[48]", arg45_1: "f32[48]", arg46_1: "f32[48, 1, 5, 5]", arg47_1: "f32[48]", arg48_1: "f32[48]", arg49_1: "f32[48]", arg50_1: "f32[48]", arg51_1: "f32[24, 48, 1, 1]", arg52_1: "f32[24]", arg53_1: "f32[24]", arg54_1: "f32[24]", arg55_1: "f32[24]", arg56_1: "f32[48, 24, 1, 1]", arg57_1: "f32[48]", arg58_1: "f32[48]", arg59_1: "f32[48]", arg60_1: "f32[48]", arg61_1: "f32[48, 1, 5, 5]", arg62_1: "f32[48]", arg63_1: "f32[48]", arg64_1: "f32[48]", arg65_1: "f32[48]", arg66_1: "f32[24, 48, 1, 1]", arg67_1: "f32[24]", arg68_1: "f32[24]", arg69_1: "f32[24]", arg70_1: "f32[24]", arg71_1: "f32[48, 24, 1, 1]", arg72_1: "f32[48]", arg73_1: "f32[48]", arg74_1: "f32[48]", arg75_1: "f32[48]", arg76_1: "f32[48, 1, 5, 5]", arg77_1: "f32[48]", arg78_1: "f32[48]", arg79_1: "f32[48]", arg80_1: "f32[48]", arg81_1: "f32[24, 48, 1, 1]", arg82_1: "f32[24]", arg83_1: "f32[24]", arg84_1: "f32[24]", arg85_1: "f32[24]", arg86_1: "f32[120, 24, 1, 1]", arg87_1: "f32[120]", arg88_1: "f32[120]", arg89_1: "f32[120]", arg90_1: "f32[120]", arg91_1: "f32[120, 1, 5, 5]", arg92_1: "f32[120]", arg93_1: "f32[120]", arg94_1: "f32[120]", arg95_1: "f32[120]", arg96_1: "f32[8, 120, 1, 1]", arg97_1: "f32[8]", arg98_1: "f32[120, 8, 1, 1]", arg99_1: "f32[120]", arg100_1: "f32[40, 120, 1, 1]", arg101_1: "f32[40]", arg102_1: "f32[40]", arg103_1: "f32[40]", arg104_1: "f32[40]", arg105_1: "f32[120, 40, 1, 1]", arg106_1: "f32[120]", arg107_1: "f32[120]", arg108_1: "f32[120]", arg109_1: "f32[120]", arg110_1: "f32[120, 1, 5, 5]", arg111_1: "f32[120]", arg112_1: "f32[120]", arg113_1: "f32[120]", arg114_1: "f32[120]", arg115_1: "f32[16, 120, 1, 1]", arg116_1: "f32[16]", arg117_1: "f32[120, 16, 1, 1]", arg118_1: "f32[120]", arg119_1: "f32[40, 120, 1, 1]", arg120_1: "f32[40]", arg121_1: "f32[40]", arg122_1: "f32[40]", arg123_1: "f32[40]", arg124_1: "f32[120, 40, 1, 1]", arg125_1: "f32[120]", arg126_1: "f32[120]", arg127_1: "f32[120]", arg128_1: "f32[120]", arg129_1: "f32[120, 1, 5, 5]", arg130_1: "f32[120]", arg131_1: "f32[120]", arg132_1: "f32[120]", arg133_1: "f32[120]", arg134_1: "f32[16, 120, 1, 1]", arg135_1: "f32[16]", arg136_1: "f32[120, 16, 1, 1]", arg137_1: "f32[120]", arg138_1: "f32[40, 120, 1, 1]", arg139_1: "f32[40]", arg140_1: "f32[40]", arg141_1: "f32[40]", arg142_1: "f32[40]", arg143_1: "f32[120, 40, 1, 1]", arg144_1: "f32[120]", arg145_1: "f32[120]", arg146_1: "f32[120]", arg147_1: "f32[120]", arg148_1: "f32[120, 1, 5, 5]", arg149_1: "f32[120]", arg150_1: "f32[120]", arg151_1: "f32[120]", arg152_1: "f32[120]", arg153_1: "f32[16, 120, 1, 1]", arg154_1: "f32[16]", arg155_1: "f32[120, 16, 1, 1]", arg156_1: "f32[120]", arg157_1: "f32[40, 120, 1, 1]", arg158_1: "f32[40]", arg159_1: "f32[40]", arg160_1: "f32[40]", arg161_1: "f32[40]", arg162_1: "f32[120, 40, 1, 1]", arg163_1: "f32[120]", arg164_1: "f32[120]", arg165_1: "f32[120]", arg166_1: "f32[120]", arg167_1: "f32[120, 1, 5, 5]", arg168_1: "f32[120]", arg169_1: "f32[120]", arg170_1: "f32[120]", arg171_1: "f32[120]", arg172_1: "f32[16, 120, 1, 1]", arg173_1: "f32[16]", arg174_1: "f32[120, 16, 1, 1]", arg175_1: "f32[120]", arg176_1: "f32[40, 120, 1, 1]", arg177_1: "f32[40]", arg178_1: "f32[40]", arg179_1: "f32[40]", arg180_1: "f32[40]", arg181_1: "f32[200, 40, 1, 1]", arg182_1: "f32[200]", arg183_1: "f32[200]", arg184_1: "f32[200]", arg185_1: "f32[200]", arg186_1: "f32[200, 1, 5, 5]", arg187_1: "f32[200]", arg188_1: "f32[200]", arg189_1: "f32[200]", arg190_1: "f32[200]", arg191_1: "f32[72, 200, 1, 1]", arg192_1: "f32[72]", arg193_1: "f32[72]", arg194_1: "f32[72]", arg195_1: "f32[72]", arg196_1: "f32[216, 72, 1, 1]", arg197_1: "f32[216]", arg198_1: "f32[216]", arg199_1: "f32[216]", arg200_1: "f32[216]", arg201_1: "f32[216, 1, 3, 3]", arg202_1: "f32[216]", arg203_1: "f32[216]", arg204_1: "f32[216]", arg205_1: "f32[216]", arg206_1: "f32[72, 216, 1, 1]", arg207_1: "f32[72]", arg208_1: "f32[72]", arg209_1: "f32[72]", arg210_1: "f32[72]", arg211_1: "f32[216, 72, 1, 1]", arg212_1: "f32[216]", arg213_1: "f32[216]", arg214_1: "f32[216]", arg215_1: "f32[216]", arg216_1: "f32[216, 1, 3, 3]", arg217_1: "f32[216]", arg218_1: "f32[216]", arg219_1: "f32[216]", arg220_1: "f32[216]", arg221_1: "f32[72, 216, 1, 1]", arg222_1: "f32[72]", arg223_1: "f32[72]", arg224_1: "f32[72]", arg225_1: "f32[72]", arg226_1: "f32[216, 72, 1, 1]", arg227_1: "f32[216]", arg228_1: "f32[216]", arg229_1: "f32[216]", arg230_1: "f32[216]", arg231_1: "f32[216, 1, 3, 3]", arg232_1: "f32[216]", arg233_1: "f32[216]", arg234_1: "f32[216]", arg235_1: "f32[216]", arg236_1: "f32[72, 216, 1, 1]", arg237_1: "f32[72]", arg238_1: "f32[72]", arg239_1: "f32[72]", arg240_1: "f32[72]", arg241_1: "f32[216, 72, 1, 1]", arg242_1: "f32[216]", arg243_1: "f32[216]", arg244_1: "f32[216]", arg245_1: "f32[216]", arg246_1: "f32[216, 1, 3, 3]", arg247_1: "f32[216]", arg248_1: "f32[216]", arg249_1: "f32[216]", arg250_1: "f32[216]", arg251_1: "f32[72, 216, 1, 1]", arg252_1: "f32[72]", arg253_1: "f32[72]", arg254_1: "f32[72]", arg255_1: "f32[72]", arg256_1: "f32[360, 72, 1, 1]", arg257_1: "f32[360]", arg258_1: "f32[360]", arg259_1: "f32[360]", arg260_1: "f32[360]", arg261_1: "f32[360, 1, 3, 3]", arg262_1: "f32[360]", arg263_1: "f32[360]", arg264_1: "f32[360]", arg265_1: "f32[360]", arg266_1: "f32[24, 360, 1, 1]", arg267_1: "f32[24]", arg268_1: "f32[360, 24, 1, 1]", arg269_1: "f32[360]", arg270_1: "f32[120, 360, 1, 1]", arg271_1: "f32[120]", arg272_1: "f32[120]", arg273_1: "f32[120]", arg274_1: "f32[120]", arg275_1: "f32[360, 120, 1, 1]", arg276_1: "f32[360]", arg277_1: "f32[360]", arg278_1: "f32[360]", arg279_1: "f32[360]", arg280_1: "f32[360, 1, 5, 5]", arg281_1: "f32[360]", arg282_1: "f32[360]", arg283_1: "f32[360]", arg284_1: "f32[360]", arg285_1: "f32[32, 360, 1, 1]", arg286_1: "f32[32]", arg287_1: "f32[360, 32, 1, 1]", arg288_1: "f32[360]", arg289_1: "f32[120, 360, 1, 1]", arg290_1: "f32[120]", arg291_1: "f32[120]", arg292_1: "f32[120]", arg293_1: "f32[120]", arg294_1: "f32[360, 120, 1, 1]", arg295_1: "f32[360]", arg296_1: "f32[360]", arg297_1: "f32[360]", arg298_1: "f32[360]", arg299_1: "f32[360, 1, 5, 5]", arg300_1: "f32[360]", arg301_1: "f32[360]", arg302_1: "f32[360]", arg303_1: "f32[360]", arg304_1: "f32[32, 360, 1, 1]", arg305_1: "f32[32]", arg306_1: "f32[360, 32, 1, 1]", arg307_1: "f32[360]", arg308_1: "f32[120, 360, 1, 1]", arg309_1: "f32[120]", arg310_1: "f32[120]", arg311_1: "f32[120]", arg312_1: "f32[120]", arg313_1: "f32[360, 120, 1, 1]", arg314_1: "f32[360]", arg315_1: "f32[360]", arg316_1: "f32[360]", arg317_1: "f32[360]", arg318_1: "f32[360, 1, 5, 5]", arg319_1: "f32[360]", arg320_1: "f32[360]", arg321_1: "f32[360]", arg322_1: "f32[360]", arg323_1: "f32[32, 360, 1, 1]", arg324_1: "f32[32]", arg325_1: "f32[360, 32, 1, 1]", arg326_1: "f32[360]", arg327_1: "f32[120, 360, 1, 1]", arg328_1: "f32[120]", arg329_1: "f32[120]", arg330_1: "f32[120]", arg331_1: "f32[120]", arg332_1: "f32[360, 120, 1, 1]", arg333_1: "f32[360]", arg334_1: "f32[360]", arg335_1: "f32[360]", arg336_1: "f32[360]", arg337_1: "f32[360, 1, 5, 5]", arg338_1: "f32[360]", arg339_1: "f32[360]", arg340_1: "f32[360]", arg341_1: "f32[360]", arg342_1: "f32[32, 360, 1, 1]", arg343_1: "f32[32]", arg344_1: "f32[360, 32, 1, 1]", arg345_1: "f32[360]", arg346_1: "f32[120, 360, 1, 1]", arg347_1: "f32[120]", arg348_1: "f32[120]", arg349_1: "f32[120]", arg350_1: "f32[120]", arg351_1: "f32[360, 120, 1, 1]", arg352_1: "f32[360]", arg353_1: "f32[360]", arg354_1: "f32[360]", arg355_1: "f32[360]", arg356_1: "f32[360, 1, 5, 5]", arg357_1: "f32[360]", arg358_1: "f32[360]", arg359_1: "f32[360]", arg360_1: "f32[360]", arg361_1: "f32[32, 360, 1, 1]", arg362_1: "f32[32]", arg363_1: "f32[360, 32, 1, 1]", arg364_1: "f32[360]", arg365_1: "f32[120, 360, 1, 1]", arg366_1: "f32[120]", arg367_1: "f32[120]", arg368_1: "f32[120]", arg369_1: "f32[120]", arg370_1: "f32[720, 120, 1, 1]", arg371_1: "f32[720]", arg372_1: "f32[720]", arg373_1: "f32[720]", arg374_1: "f32[720]", arg375_1: "f32[720, 1, 3, 3]", arg376_1: "f32[720]", arg377_1: "f32[720]", arg378_1: "f32[720]", arg379_1: "f32[720]", arg380_1: "f32[32, 720, 1, 1]", arg381_1: "f32[32]", arg382_1: "f32[720, 32, 1, 1]", arg383_1: "f32[720]", arg384_1: "f32[184, 720, 1, 1]", arg385_1: "f32[184]", arg386_1: "f32[184]", arg387_1: "f32[184]", arg388_1: "f32[184]", arg389_1: "f32[736, 184, 1, 1]", arg390_1: "f32[736]", arg391_1: "f32[736]", arg392_1: "f32[736]", arg393_1: "f32[736]", arg394_1: "f32[736, 1, 5, 5]", arg395_1: "f32[736]", arg396_1: "f32[736]", arg397_1: "f32[736]", arg398_1: "f32[736]", arg399_1: "f32[48, 736, 1, 1]", arg400_1: "f32[48]", arg401_1: "f32[736, 48, 1, 1]", arg402_1: "f32[736]", arg403_1: "f32[184, 736, 1, 1]", arg404_1: "f32[184]", arg405_1: "f32[184]", arg406_1: "f32[184]", arg407_1: "f32[184]", arg408_1: "f32[736, 184, 1, 1]", arg409_1: "f32[736]", arg410_1: "f32[736]", arg411_1: "f32[736]", arg412_1: "f32[736]", arg413_1: "f32[736, 1, 5, 5]", arg414_1: "f32[736]", arg415_1: "f32[736]", arg416_1: "f32[736]", arg417_1: "f32[736]", arg418_1: "f32[48, 736, 1, 1]", arg419_1: "f32[48]", arg420_1: "f32[736, 48, 1, 1]", arg421_1: "f32[736]", arg422_1: "f32[184, 736, 1, 1]", arg423_1: "f32[184]", arg424_1: "f32[184]", arg425_1: "f32[184]", arg426_1: "f32[184]", arg427_1: "f32[736, 184, 1, 1]", arg428_1: "f32[736]", arg429_1: "f32[736]", arg430_1: "f32[736]", arg431_1: "f32[736]", arg432_1: "f32[736, 1, 5, 5]", arg433_1: "f32[736]", arg434_1: "f32[736]", arg435_1: "f32[736]", arg436_1: "f32[736]", arg437_1: "f32[48, 736, 1, 1]", arg438_1: "f32[48]", arg439_1: "f32[736, 48, 1, 1]", arg440_1: "f32[736]", arg441_1: "f32[184, 736, 1, 1]", arg442_1: "f32[184]", arg443_1: "f32[184]", arg444_1: "f32[184]", arg445_1: "f32[184]", arg446_1: "f32[736, 184, 1, 1]", arg447_1: "f32[736]", arg448_1: "f32[736]", arg449_1: "f32[736]", arg450_1: "f32[736]", arg451_1: "f32[736, 1, 5, 5]", arg452_1: "f32[736]", arg453_1: "f32[736]", arg454_1: "f32[736]", arg455_1: "f32[736]", arg456_1: "f32[48, 736, 1, 1]", arg457_1: "f32[48]", arg458_1: "f32[736, 48, 1, 1]", arg459_1: "f32[736]", arg460_1: "f32[184, 736, 1, 1]", arg461_1: "f32[184]", arg462_1: "f32[184]", arg463_1: "f32[184]", arg464_1: "f32[184]", arg465_1: "f32[736, 184, 1, 1]", arg466_1: "f32[736]", arg467_1: "f32[736]", arg468_1: "f32[736]", arg469_1: "f32[736]", arg470_1: "f32[736, 1, 5, 5]", arg471_1: "f32[736]", arg472_1: "f32[736]", arg473_1: "f32[736]", arg474_1: "f32[736]", arg475_1: "f32[48, 736, 1, 1]", arg476_1: "f32[48]", arg477_1: "f32[736, 48, 1, 1]", arg478_1: "f32[736]", arg479_1: "f32[184, 736, 1, 1]", arg480_1: "f32[184]", arg481_1: "f32[184]", arg482_1: "f32[184]", arg483_1: "f32[184]", arg484_1: "f32[1104, 184, 1, 1]", arg485_1: "f32[1104]", arg486_1: "f32[1104]", arg487_1: "f32[1104]", arg488_1: "f32[1104]", arg489_1: "f32[1104, 1, 5, 5]", arg490_1: "f32[1104]", arg491_1: "f32[1104]", arg492_1: "f32[1104]", arg493_1: "f32[1104]", arg494_1: "f32[48, 1104, 1, 1]", arg495_1: "f32[48]", arg496_1: "f32[1104, 48, 1, 1]", arg497_1: "f32[1104]", arg498_1: "f32[224, 1104, 1, 1]", arg499_1: "f32[224]", arg500_1: "f32[224]", arg501_1: "f32[224]", arg502_1: "f32[224]", arg503_1: "f32[1344, 224, 1, 1]", arg504_1: "f32[1344]", arg505_1: "f32[1344]", arg506_1: "f32[1344]", arg507_1: "f32[1344]", arg508_1: "f32[1984, 1344, 1, 1]", arg509_1: "f32[1000, 1984]", arg510_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:251 in forward_features, code: x = self.conv_stem(x)
        convolution_124: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_696: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_697: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        sub_87: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_697);  convolution_124 = unsqueeze_697 = None
        add_292: "f32[16]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_87: "f32[16]" = torch.ops.aten.sqrt.default(add_292);  add_292 = None
        reciprocal_87: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_356: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_698: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_356, -1);  mul_356 = None
        unsqueeze_699: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        mul_357: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_701: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_358: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_701);  mul_357 = unsqueeze_701 = None
        unsqueeze_702: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_703: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_293: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_703);  mul_358 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_294: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_293, 3)
        clamp_min_95: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_min.default(add_294, 0);  add_294 = None
        clamp_max_95: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_95, 6);  clamp_min_95 = None
        mul_359: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_293, clamp_max_95);  add_293 = clamp_max_95 = None
        div_95: "f32[8, 16, 128, 128]" = torch.ops.aten.div.Tensor(mul_359, 6);  mul_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_125: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(div_95, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_704: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_705: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        sub_88: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_705);  convolution_125 = unsqueeze_705 = None
        add_295: "f32[16]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_88: "f32[16]" = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_88: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_360: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_706: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_707: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        mul_361: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_709: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_362: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_709);  mul_361 = unsqueeze_709 = None
        unsqueeze_710: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_711: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_296: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_711);  mul_362 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_297: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_296, 3)
        clamp_min_96: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_min.default(add_297, 0);  add_297 = None
        clamp_max_96: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_96, 6);  clamp_min_96 = None
        mul_363: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_296, clamp_max_96);  add_296 = clamp_max_96 = None
        div_96: "f32[8, 16, 128, 128]" = torch.ops.aten.div.Tensor(mul_363, 6);  mul_363 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_126: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(div_96, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_96 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_712: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_713: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        sub_89: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_713);  convolution_126 = unsqueeze_713 = None
        add_298: "f32[16]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_89: "f32[16]" = torch.ops.aten.sqrt.default(add_298);  add_298 = None
        reciprocal_89: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_364: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_714: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_364, -1);  mul_364 = None
        unsqueeze_715: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        mul_365: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_717: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_366: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_365, unsqueeze_717);  mul_365 = unsqueeze_717 = None
        unsqueeze_718: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_719: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_299: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_366, unsqueeze_719);  mul_366 = unsqueeze_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:197 in forward, code: x = self.drop_path(x) + shortcut
        add_300: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_299, div_95);  add_299 = div_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_127: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(add_300, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_720: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_721: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        sub_90: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_721);  convolution_127 = unsqueeze_721 = None
        add_301: "f32[16]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_90: "f32[16]" = torch.ops.aten.sqrt.default(add_301);  add_301 = None
        reciprocal_90: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_367: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_722: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_367, -1);  mul_367 = None
        unsqueeze_723: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        mul_368: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_725: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_369: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_368, unsqueeze_725);  mul_368 = unsqueeze_725 = None
        unsqueeze_726: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_727: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_302: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_369, unsqueeze_727);  mul_369 = unsqueeze_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_303: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_302, 3)
        clamp_min_97: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_min.default(add_303, 0);  add_303 = None
        clamp_max_97: "f32[8, 16, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_97, 6);  clamp_min_97 = None
        mul_370: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_302, clamp_max_97);  add_302 = clamp_max_97 = None
        div_97: "f32[8, 16, 128, 128]" = torch.ops.aten.div.Tensor(mul_370, 6);  mul_370 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_128: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(div_97, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_97 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_728: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_729: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        sub_91: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_729);  convolution_128 = unsqueeze_729 = None
        add_304: "f32[16]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_91: "f32[16]" = torch.ops.aten.sqrt.default(add_304);  add_304 = None
        reciprocal_91: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_371: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_730: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_371, -1);  mul_371 = None
        unsqueeze_731: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        mul_372: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_733: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_373: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_372, unsqueeze_733);  mul_372 = unsqueeze_733 = None
        unsqueeze_734: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_735: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_305: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_373, unsqueeze_735);  mul_373 = unsqueeze_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:197 in forward, code: x = self.drop_path(x) + shortcut
        add_306: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(add_305, add_300);  add_305 = add_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_129: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(add_306, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_306 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_736: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_737: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        sub_92: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_737);  convolution_129 = unsqueeze_737 = None
        add_307: "f32[64]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_92: "f32[64]" = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_92: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_374: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_738: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_374, -1);  mul_374 = None
        unsqueeze_739: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        mul_375: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_741: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_376: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_375, unsqueeze_741);  mul_375 = unsqueeze_741 = None
        unsqueeze_742: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_743: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_308: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_376, unsqueeze_743);  mul_376 = unsqueeze_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_309: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(add_308, 3)
        clamp_min_98: "f32[8, 64, 128, 128]" = torch.ops.aten.clamp_min.default(add_309, 0);  add_309 = None
        clamp_max_98: "f32[8, 64, 128, 128]" = torch.ops.aten.clamp_max.default(clamp_min_98, 6);  clamp_min_98 = None
        mul_377: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_308, clamp_max_98);  add_308 = clamp_max_98 = None
        div_98: "f32[8, 64, 128, 128]" = torch.ops.aten.div.Tensor(mul_377, 6);  mul_377 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_130: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(div_98, arg31_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 64);  div_98 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_744: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_745: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        sub_93: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_745);  convolution_130 = unsqueeze_745 = None
        add_310: "f32[64]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_93: "f32[64]" = torch.ops.aten.sqrt.default(add_310);  add_310 = None
        reciprocal_93: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_378: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_746: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_747: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        mul_379: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_749: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_380: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_749);  mul_379 = unsqueeze_749 = None
        unsqueeze_750: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_751: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_311: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_751);  mul_380 = unsqueeze_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_312: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_311, 3)
        clamp_min_99: "f32[8, 64, 64, 64]" = torch.ops.aten.clamp_min.default(add_312, 0);  add_312 = None
        clamp_max_99: "f32[8, 64, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_99, 6);  clamp_min_99 = None
        mul_381: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_311, clamp_max_99);  add_311 = clamp_max_99 = None
        div_99: "f32[8, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_381, 6);  mul_381 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_131: "f32[8, 24, 64, 64]" = torch.ops.aten.convolution.default(div_99, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_99 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_752: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_753: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        sub_94: "f32[8, 24, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_753);  convolution_131 = unsqueeze_753 = None
        add_313: "f32[24]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_94: "f32[24]" = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_94: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_382: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_754: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_382, -1);  mul_382 = None
        unsqueeze_755: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        mul_383: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_757: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_384: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(mul_383, unsqueeze_757);  mul_383 = unsqueeze_757 = None
        unsqueeze_758: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_759: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_314: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_759);  mul_384 = unsqueeze_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_132: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(add_314, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_760: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_761: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        sub_95: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_761);  convolution_132 = unsqueeze_761 = None
        add_315: "f32[48]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_95: "f32[48]" = torch.ops.aten.sqrt.default(add_315);  add_315 = None
        reciprocal_95: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_385: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_762: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_385, -1);  mul_385 = None
        unsqueeze_763: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        mul_386: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_765: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_387: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_386, unsqueeze_765);  mul_386 = unsqueeze_765 = None
        unsqueeze_766: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_767: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_316: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_387, unsqueeze_767);  mul_387 = unsqueeze_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_317: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_316, 3)
        clamp_min_100: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_317, 0);  add_317 = None
        clamp_max_100: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_100, 6);  clamp_min_100 = None
        mul_388: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_316, clamp_max_100);  add_316 = clamp_max_100 = None
        div_100: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_388, 6);  mul_388 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_133: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(div_100, arg46_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_100 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_768: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_769: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        sub_96: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_769);  convolution_133 = unsqueeze_769 = None
        add_318: "f32[48]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_96: "f32[48]" = torch.ops.aten.sqrt.default(add_318);  add_318 = None
        reciprocal_96: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_389: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_770: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_389, -1);  mul_389 = None
        unsqueeze_771: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        mul_390: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_773: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_391: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_390, unsqueeze_773);  mul_390 = unsqueeze_773 = None
        unsqueeze_774: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_775: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_319: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_775);  mul_391 = unsqueeze_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_320: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_319, 3)
        clamp_min_101: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_320, 0);  add_320 = None
        clamp_max_101: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_101, 6);  clamp_min_101 = None
        mul_392: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_319, clamp_max_101);  add_319 = clamp_max_101 = None
        div_101: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_392, 6);  mul_392 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_134: "f32[8, 24, 64, 64]" = torch.ops.aten.convolution.default(div_101, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_101 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_776: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_777: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        sub_97: "f32[8, 24, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_777);  convolution_134 = unsqueeze_777 = None
        add_321: "f32[24]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_97: "f32[24]" = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_97: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_393: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_778: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
        unsqueeze_779: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        mul_394: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_781: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_395: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_781);  mul_394 = unsqueeze_781 = None
        unsqueeze_782: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_783: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_322: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(mul_395, unsqueeze_783);  mul_395 = unsqueeze_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_323: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(add_322, add_314);  add_322 = add_314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_135: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(add_323, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_784: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_785: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        sub_98: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_785);  convolution_135 = unsqueeze_785 = None
        add_324: "f32[48]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_98: "f32[48]" = torch.ops.aten.sqrt.default(add_324);  add_324 = None
        reciprocal_98: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_396: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_786: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
        unsqueeze_787: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        mul_397: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_789: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_398: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_789);  mul_397 = unsqueeze_789 = None
        unsqueeze_790: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_791: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_325: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_791);  mul_398 = unsqueeze_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_326: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_325, 3)
        clamp_min_102: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_326, 0);  add_326 = None
        clamp_max_102: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_102, 6);  clamp_min_102 = None
        mul_399: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_325, clamp_max_102);  add_325 = clamp_max_102 = None
        div_102: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_399, 6);  mul_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_136: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(div_102, arg61_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_102 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_792: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_793: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        sub_99: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_793);  convolution_136 = unsqueeze_793 = None
        add_327: "f32[48]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_99: "f32[48]" = torch.ops.aten.sqrt.default(add_327);  add_327 = None
        reciprocal_99: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_400: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_794: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_400, -1);  mul_400 = None
        unsqueeze_795: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        mul_401: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_797: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_402: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_401, unsqueeze_797);  mul_401 = unsqueeze_797 = None
        unsqueeze_798: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_799: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_328: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_402, unsqueeze_799);  mul_402 = unsqueeze_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_329: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_328, 3)
        clamp_min_103: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_329, 0);  add_329 = None
        clamp_max_103: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_103, 6);  clamp_min_103 = None
        mul_403: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_328, clamp_max_103);  add_328 = clamp_max_103 = None
        div_103: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_403, 6);  mul_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_137: "f32[8, 24, 64, 64]" = torch.ops.aten.convolution.default(div_103, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_103 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_800: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_801: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        sub_100: "f32[8, 24, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_801);  convolution_137 = unsqueeze_801 = None
        add_330: "f32[24]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_100: "f32[24]" = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_100: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_404: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_802: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_404, -1);  mul_404 = None
        unsqueeze_803: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        mul_405: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_805: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_406: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_805);  mul_405 = unsqueeze_805 = None
        unsqueeze_806: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_807: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_331: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_807);  mul_406 = unsqueeze_807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_332: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(add_331, add_323);  add_331 = add_323 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_138: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(add_332, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_808: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_809: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        sub_101: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_809);  convolution_138 = unsqueeze_809 = None
        add_333: "f32[48]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_101: "f32[48]" = torch.ops.aten.sqrt.default(add_333);  add_333 = None
        reciprocal_101: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_407: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_810: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_811: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        mul_408: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_813: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_409: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_813);  mul_408 = unsqueeze_813 = None
        unsqueeze_814: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_815: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_334: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_815);  mul_409 = unsqueeze_815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_335: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_334, 3)
        clamp_min_104: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_335, 0);  add_335 = None
        clamp_max_104: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_104, 6);  clamp_min_104 = None
        mul_410: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_334, clamp_max_104);  add_334 = clamp_max_104 = None
        div_104: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_410, 6);  mul_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_139: "f32[8, 48, 64, 64]" = torch.ops.aten.convolution.default(div_104, arg76_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  div_104 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_816: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_817: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        sub_102: "f32[8, 48, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_817);  convolution_139 = unsqueeze_817 = None
        add_336: "f32[48]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_102: "f32[48]" = torch.ops.aten.sqrt.default(add_336);  add_336 = None
        reciprocal_102: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_411: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_818: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_819: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        mul_412: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_821: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_413: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_821);  mul_412 = unsqueeze_821 = None
        unsqueeze_822: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_823: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_337: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(mul_413, unsqueeze_823);  mul_413 = unsqueeze_823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_338: "f32[8, 48, 64, 64]" = torch.ops.aten.add.Tensor(add_337, 3)
        clamp_min_105: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_min.default(add_338, 0);  add_338 = None
        clamp_max_105: "f32[8, 48, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_105, 6);  clamp_min_105 = None
        mul_414: "f32[8, 48, 64, 64]" = torch.ops.aten.mul.Tensor(add_337, clamp_max_105);  add_337 = clamp_max_105 = None
        div_105: "f32[8, 48, 64, 64]" = torch.ops.aten.div.Tensor(mul_414, 6);  mul_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_140: "f32[8, 24, 64, 64]" = torch.ops.aten.convolution.default(div_105, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_105 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_824: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_825: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        sub_103: "f32[8, 24, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_825);  convolution_140 = unsqueeze_825 = None
        add_339: "f32[24]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_103: "f32[24]" = torch.ops.aten.sqrt.default(add_339);  add_339 = None
        reciprocal_103: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_415: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_826: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_415, -1);  mul_415 = None
        unsqueeze_827: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        mul_416: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_829: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_417: "f32[8, 24, 64, 64]" = torch.ops.aten.mul.Tensor(mul_416, unsqueeze_829);  mul_416 = unsqueeze_829 = None
        unsqueeze_830: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_831: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_340: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(mul_417, unsqueeze_831);  mul_417 = unsqueeze_831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_341: "f32[8, 24, 64, 64]" = torch.ops.aten.add.Tensor(add_340, add_332);  add_340 = add_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_141: "f32[8, 120, 64, 64]" = torch.ops.aten.convolution.default(add_341, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_341 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_832: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_833: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        sub_104: "f32[8, 120, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_833);  convolution_141 = unsqueeze_833 = None
        add_342: "f32[120]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_104: "f32[120]" = torch.ops.aten.sqrt.default(add_342);  add_342 = None
        reciprocal_104: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_418: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_834: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_418, -1);  mul_418 = None
        unsqueeze_835: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        mul_419: "f32[8, 120, 64, 64]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_837: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_420: "f32[8, 120, 64, 64]" = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_837);  mul_419 = unsqueeze_837 = None
        unsqueeze_838: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_839: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_343: "f32[8, 120, 64, 64]" = torch.ops.aten.add.Tensor(mul_420, unsqueeze_839);  mul_420 = unsqueeze_839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_344: "f32[8, 120, 64, 64]" = torch.ops.aten.add.Tensor(add_343, 3)
        clamp_min_106: "f32[8, 120, 64, 64]" = torch.ops.aten.clamp_min.default(add_344, 0);  add_344 = None
        clamp_max_106: "f32[8, 120, 64, 64]" = torch.ops.aten.clamp_max.default(clamp_min_106, 6);  clamp_min_106 = None
        mul_421: "f32[8, 120, 64, 64]" = torch.ops.aten.mul.Tensor(add_343, clamp_max_106);  add_343 = clamp_max_106 = None
        div_106: "f32[8, 120, 64, 64]" = torch.ops.aten.div.Tensor(mul_421, 6);  mul_421 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_142: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_106, arg91_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 120);  div_106 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_840: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_841: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        sub_105: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_841);  convolution_142 = unsqueeze_841 = None
        add_345: "f32[120]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_105: "f32[120]" = torch.ops.aten.sqrt.default(add_345);  add_345 = None
        reciprocal_105: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_422: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_842: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_422, -1);  mul_422 = None
        unsqueeze_843: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        mul_423: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_845: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_424: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_423, unsqueeze_845);  mul_423 = unsqueeze_845 = None
        unsqueeze_846: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_847: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_346: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_424, unsqueeze_847);  mul_424 = unsqueeze_847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_347: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_346, 3)
        clamp_min_107: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_347, 0);  add_347 = None
        clamp_max_107: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_107, 6);  clamp_min_107 = None
        mul_425: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_346, clamp_max_107);  add_346 = clamp_max_107 = None
        div_107: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_425, 6);  mul_425 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_19: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_107, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_143: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_19, arg96_1, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg96_1 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_348: "f32[8, 8, 1, 1]" = torch.ops.aten.add.Tensor(convolution_143, 3)
        clamp_min_108: "f32[8, 8, 1, 1]" = torch.ops.aten.clamp_min.default(add_348, 0);  add_348 = None
        clamp_max_108: "f32[8, 8, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_108, 6);  clamp_min_108 = None
        mul_426: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_143, clamp_max_108);  convolution_143 = clamp_max_108 = None
        div_108: "f32[8, 8, 1, 1]" = torch.ops.aten.div.Tensor(mul_426, 6);  mul_426 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_144: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_108, arg98_1, arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_108 = arg98_1 = arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_349: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_144, 3);  convolution_144 = None
        clamp_min_109: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_349, 0);  add_349 = None
        clamp_max_109: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_109, 6);  clamp_min_109 = None
        div_109: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_109, 6);  clamp_max_109 = None
        mul_427: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_107, div_109);  div_107 = div_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_145: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_427, arg100_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_427 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_848: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_849: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        sub_106: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_849);  convolution_145 = unsqueeze_849 = None
        add_350: "f32[40]" = torch.ops.aten.add.Tensor(arg102_1, 1e-05);  arg102_1 = None
        sqrt_106: "f32[40]" = torch.ops.aten.sqrt.default(add_350);  add_350 = None
        reciprocal_106: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_428: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_850: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_428, -1);  mul_428 = None
        unsqueeze_851: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        mul_429: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_853: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_430: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_429, unsqueeze_853);  mul_429 = unsqueeze_853 = None
        unsqueeze_854: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_855: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_351: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_430, unsqueeze_855);  mul_430 = unsqueeze_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_146: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(add_351, arg105_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg105_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_856: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_857: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        sub_107: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_857);  convolution_146 = unsqueeze_857 = None
        add_352: "f32[120]" = torch.ops.aten.add.Tensor(arg107_1, 1e-05);  arg107_1 = None
        sqrt_107: "f32[120]" = torch.ops.aten.sqrt.default(add_352);  add_352 = None
        reciprocal_107: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_431: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_858: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_431, -1);  mul_431 = None
        unsqueeze_859: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        mul_432: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_861: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_433: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_432, unsqueeze_861);  mul_432 = unsqueeze_861 = None
        unsqueeze_862: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_863: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_353: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_863);  mul_433 = unsqueeze_863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_354: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_353, 3)
        clamp_min_110: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_354, 0);  add_354 = None
        clamp_max_110: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_110, 6);  clamp_min_110 = None
        mul_434: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_353, clamp_max_110);  add_353 = clamp_max_110 = None
        div_110: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_434, 6);  mul_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_147: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_110, arg110_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_110 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_864: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_865: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        sub_108: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_865);  convolution_147 = unsqueeze_865 = None
        add_355: "f32[120]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_108: "f32[120]" = torch.ops.aten.sqrt.default(add_355);  add_355 = None
        reciprocal_108: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_435: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_866: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
        unsqueeze_867: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        mul_436: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_869: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_437: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_869);  mul_436 = unsqueeze_869 = None
        unsqueeze_870: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_871: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_356: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_871);  mul_437 = unsqueeze_871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_357: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_356, 3)
        clamp_min_111: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_357, 0);  add_357 = None
        clamp_max_111: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_111, 6);  clamp_min_111 = None
        mul_438: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_356, clamp_max_111);  add_356 = clamp_max_111 = None
        div_111: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_438, 6);  mul_438 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_20: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_111, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_148: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_20, arg115_1, arg116_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg115_1 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_358: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_148, 3)
        clamp_min_112: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_358, 0);  add_358 = None
        clamp_max_112: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_112, 6);  clamp_min_112 = None
        mul_439: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_148, clamp_max_112);  convolution_148 = clamp_max_112 = None
        div_112: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_439, 6);  mul_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_149: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_112, arg117_1, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_112 = arg117_1 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_359: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_149, 3);  convolution_149 = None
        clamp_min_113: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_359, 0);  add_359 = None
        clamp_max_113: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_113, 6);  clamp_min_113 = None
        div_113: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_113, 6);  clamp_max_113 = None
        mul_440: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_111, div_113);  div_111 = div_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_150: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_440, arg119_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_440 = arg119_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_872: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_873: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        sub_109: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_873);  convolution_150 = unsqueeze_873 = None
        add_360: "f32[40]" = torch.ops.aten.add.Tensor(arg121_1, 1e-05);  arg121_1 = None
        sqrt_109: "f32[40]" = torch.ops.aten.sqrt.default(add_360);  add_360 = None
        reciprocal_109: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_441: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_874: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
        unsqueeze_875: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        mul_442: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_877: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_443: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_877);  mul_442 = unsqueeze_877 = None
        unsqueeze_878: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_879: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_361: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_443, unsqueeze_879);  mul_443 = unsqueeze_879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_362: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(add_361, add_351);  add_361 = add_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_151: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(add_362, arg124_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_880: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_881: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        sub_110: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_881);  convolution_151 = unsqueeze_881 = None
        add_363: "f32[120]" = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
        sqrt_110: "f32[120]" = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_110: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_444: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_882: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_883: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        mul_445: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_885: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_446: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_885);  mul_445 = unsqueeze_885 = None
        unsqueeze_886: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_887: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_364: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_887);  mul_446 = unsqueeze_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_365: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_364, 3)
        clamp_min_114: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_365, 0);  add_365 = None
        clamp_max_114: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_114, 6);  clamp_min_114 = None
        mul_447: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_364, clamp_max_114);  add_364 = clamp_max_114 = None
        div_114: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_447, 6);  mul_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_152: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_114, arg129_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_114 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_888: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_889: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        sub_111: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_889);  convolution_152 = unsqueeze_889 = None
        add_366: "f32[120]" = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_111: "f32[120]" = torch.ops.aten.sqrt.default(add_366);  add_366 = None
        reciprocal_111: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_448: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_890: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_448, -1);  mul_448 = None
        unsqueeze_891: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        mul_449: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_893: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_450: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_449, unsqueeze_893);  mul_449 = unsqueeze_893 = None
        unsqueeze_894: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_895: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_367: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_450, unsqueeze_895);  mul_450 = unsqueeze_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_368: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_367, 3)
        clamp_min_115: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_368, 0);  add_368 = None
        clamp_max_115: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_115, 6);  clamp_min_115 = None
        mul_451: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_367, clamp_max_115);  add_367 = clamp_max_115 = None
        div_115: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_451, 6);  mul_451 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_21: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_115, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_153: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_21, arg134_1, arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg134_1 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_369: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_153, 3)
        clamp_min_116: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_369, 0);  add_369 = None
        clamp_max_116: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_116, 6);  clamp_min_116 = None
        mul_452: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_153, clamp_max_116);  convolution_153 = clamp_max_116 = None
        div_116: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_452, 6);  mul_452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_154: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_116, arg136_1, arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_116 = arg136_1 = arg137_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_370: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_154, 3);  convolution_154 = None
        clamp_min_117: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_370, 0);  add_370 = None
        clamp_max_117: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_117, 6);  clamp_min_117 = None
        div_117: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_117, 6);  clamp_max_117 = None
        mul_453: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_115, div_117);  div_115 = div_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_155: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_453, arg138_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_453 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_896: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_897: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        sub_112: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_897);  convolution_155 = unsqueeze_897 = None
        add_371: "f32[40]" = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_112: "f32[40]" = torch.ops.aten.sqrt.default(add_371);  add_371 = None
        reciprocal_112: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_454: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_898: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_454, -1);  mul_454 = None
        unsqueeze_899: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        mul_455: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_901: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_456: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_901);  mul_455 = unsqueeze_901 = None
        unsqueeze_902: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_903: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_372: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_456, unsqueeze_903);  mul_456 = unsqueeze_903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_373: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(add_372, add_362);  add_372 = add_362 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_156: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(add_373, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_904: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_905: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        sub_113: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_905);  convolution_156 = unsqueeze_905 = None
        add_374: "f32[120]" = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_113: "f32[120]" = torch.ops.aten.sqrt.default(add_374);  add_374 = None
        reciprocal_113: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_457: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_906: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_457, -1);  mul_457 = None
        unsqueeze_907: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        mul_458: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_909: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_459: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_458, unsqueeze_909);  mul_458 = unsqueeze_909 = None
        unsqueeze_910: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_911: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_375: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_459, unsqueeze_911);  mul_459 = unsqueeze_911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_376: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_375, 3)
        clamp_min_118: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_376, 0);  add_376 = None
        clamp_max_118: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_118, 6);  clamp_min_118 = None
        mul_460: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_375, clamp_max_118);  add_375 = clamp_max_118 = None
        div_118: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_460, 6);  mul_460 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_157: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_118, arg148_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_118 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_912: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_913: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        sub_114: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_913);  convolution_157 = unsqueeze_913 = None
        add_377: "f32[120]" = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_114: "f32[120]" = torch.ops.aten.sqrt.default(add_377);  add_377 = None
        reciprocal_114: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_461: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_914: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_461, -1);  mul_461 = None
        unsqueeze_915: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        mul_462: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_917: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_463: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_917);  mul_462 = unsqueeze_917 = None
        unsqueeze_918: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_919: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_378: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_463, unsqueeze_919);  mul_463 = unsqueeze_919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_379: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_378, 3)
        clamp_min_119: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_379, 0);  add_379 = None
        clamp_max_119: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_119, 6);  clamp_min_119 = None
        mul_464: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_378, clamp_max_119);  add_378 = clamp_max_119 = None
        div_119: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_464, 6);  mul_464 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_22: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_119, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_158: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_22, arg153_1, arg154_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg153_1 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_380: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_158, 3)
        clamp_min_120: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_380, 0);  add_380 = None
        clamp_max_120: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_120, 6);  clamp_min_120 = None
        mul_465: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_158, clamp_max_120);  convolution_158 = clamp_max_120 = None
        div_120: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_465, 6);  mul_465 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_159: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_120, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_120 = arg155_1 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_381: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_159, 3);  convolution_159 = None
        clamp_min_121: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_381, 0);  add_381 = None
        clamp_max_121: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_121, 6);  clamp_min_121 = None
        div_121: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_121, 6);  clamp_max_121 = None
        mul_466: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_119, div_121);  div_119 = div_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_160: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_466, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_466 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_920: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_921: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        sub_115: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_921);  convolution_160 = unsqueeze_921 = None
        add_382: "f32[40]" = torch.ops.aten.add.Tensor(arg159_1, 1e-05);  arg159_1 = None
        sqrt_115: "f32[40]" = torch.ops.aten.sqrt.default(add_382);  add_382 = None
        reciprocal_115: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_467: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_922: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_467, -1);  mul_467 = None
        unsqueeze_923: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        mul_468: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_925: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_469: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_468, unsqueeze_925);  mul_468 = unsqueeze_925 = None
        unsqueeze_926: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_927: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_383: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_469, unsqueeze_927);  mul_469 = unsqueeze_927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_384: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(add_383, add_373);  add_383 = add_373 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_161: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(add_384, arg162_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_928: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_929: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        sub_116: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_929);  convolution_161 = unsqueeze_929 = None
        add_385: "f32[120]" = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_116: "f32[120]" = torch.ops.aten.sqrt.default(add_385);  add_385 = None
        reciprocal_116: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_470: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_930: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_470, -1);  mul_470 = None
        unsqueeze_931: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        mul_471: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_933: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_472: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_471, unsqueeze_933);  mul_471 = unsqueeze_933 = None
        unsqueeze_934: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_935: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_386: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_472, unsqueeze_935);  mul_472 = unsqueeze_935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_387: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_386, 3)
        clamp_min_122: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_387, 0);  add_387 = None
        clamp_max_122: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_122, 6);  clamp_min_122 = None
        mul_473: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_386, clamp_max_122);  add_386 = clamp_max_122 = None
        div_122: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_473, 6);  mul_473 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_162: "f32[8, 120, 32, 32]" = torch.ops.aten.convolution.default(div_122, arg167_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  div_122 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_936: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_937: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        sub_117: "f32[8, 120, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_937);  convolution_162 = unsqueeze_937 = None
        add_388: "f32[120]" = torch.ops.aten.add.Tensor(arg169_1, 1e-05);  arg169_1 = None
        sqrt_117: "f32[120]" = torch.ops.aten.sqrt.default(add_388);  add_388 = None
        reciprocal_117: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_474: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_938: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_474, -1);  mul_474 = None
        unsqueeze_939: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        mul_475: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_941: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_476: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(mul_475, unsqueeze_941);  mul_475 = unsqueeze_941 = None
        unsqueeze_942: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_943: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_389: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(mul_476, unsqueeze_943);  mul_476 = unsqueeze_943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_390: "f32[8, 120, 32, 32]" = torch.ops.aten.add.Tensor(add_389, 3)
        clamp_min_123: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_min.default(add_390, 0);  add_390 = None
        clamp_max_123: "f32[8, 120, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_123, 6);  clamp_min_123 = None
        mul_477: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(add_389, clamp_max_123);  add_389 = clamp_max_123 = None
        div_123: "f32[8, 120, 32, 32]" = torch.ops.aten.div.Tensor(mul_477, 6);  mul_477 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_23: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_123, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_163: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_23, arg172_1, arg173_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg172_1 = arg173_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_391: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_163, 3)
        clamp_min_124: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_391, 0);  add_391 = None
        clamp_max_124: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_124, 6);  clamp_min_124 = None
        mul_478: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_163, clamp_max_124);  convolution_163 = clamp_max_124 = None
        div_124: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_478, 6);  mul_478 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_164: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_124, arg174_1, arg175_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_124 = arg174_1 = arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_392: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_164, 3);  convolution_164 = None
        clamp_min_125: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_392, 0);  add_392 = None
        clamp_max_125: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_125, 6);  clamp_min_125 = None
        div_125: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_125, 6);  clamp_max_125 = None
        mul_479: "f32[8, 120, 32, 32]" = torch.ops.aten.mul.Tensor(div_123, div_125);  div_123 = div_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_165: "f32[8, 40, 32, 32]" = torch.ops.aten.convolution.default(mul_479, arg176_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_479 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_944: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_945: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        sub_118: "f32[8, 40, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_945);  convolution_165 = unsqueeze_945 = None
        add_393: "f32[40]" = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_118: "f32[40]" = torch.ops.aten.sqrt.default(add_393);  add_393 = None
        reciprocal_118: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_480: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_946: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_480, -1);  mul_480 = None
        unsqueeze_947: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        mul_481: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_949: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_482: "f32[8, 40, 32, 32]" = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_949);  mul_481 = unsqueeze_949 = None
        unsqueeze_950: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_951: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_394: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_951);  mul_482 = unsqueeze_951 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_395: "f32[8, 40, 32, 32]" = torch.ops.aten.add.Tensor(add_394, add_384);  add_394 = add_384 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_166: "f32[8, 200, 32, 32]" = torch.ops.aten.convolution.default(add_395, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_395 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_952: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_953: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        sub_119: "f32[8, 200, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_953);  convolution_166 = unsqueeze_953 = None
        add_396: "f32[200]" = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_119: "f32[200]" = torch.ops.aten.sqrt.default(add_396);  add_396 = None
        reciprocal_119: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_483: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_954: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_483, -1);  mul_483 = None
        unsqueeze_955: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        mul_484: "f32[8, 200, 32, 32]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_957: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_485: "f32[8, 200, 32, 32]" = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_957);  mul_484 = unsqueeze_957 = None
        unsqueeze_958: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_959: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_397: "f32[8, 200, 32, 32]" = torch.ops.aten.add.Tensor(mul_485, unsqueeze_959);  mul_485 = unsqueeze_959 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_398: "f32[8, 200, 32, 32]" = torch.ops.aten.add.Tensor(add_397, 3)
        clamp_min_126: "f32[8, 200, 32, 32]" = torch.ops.aten.clamp_min.default(add_398, 0);  add_398 = None
        clamp_max_126: "f32[8, 200, 32, 32]" = torch.ops.aten.clamp_max.default(clamp_min_126, 6);  clamp_min_126 = None
        mul_486: "f32[8, 200, 32, 32]" = torch.ops.aten.mul.Tensor(add_397, clamp_max_126);  add_397 = clamp_max_126 = None
        div_126: "f32[8, 200, 32, 32]" = torch.ops.aten.div.Tensor(mul_486, 6);  mul_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_167: "f32[8, 200, 16, 16]" = torch.ops.aten.convolution.default(div_126, arg186_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 200);  div_126 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_960: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_961: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        sub_120: "f32[8, 200, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_961);  convolution_167 = unsqueeze_961 = None
        add_399: "f32[200]" = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_120: "f32[200]" = torch.ops.aten.sqrt.default(add_399);  add_399 = None
        reciprocal_120: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_487: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_962: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_487, -1);  mul_487 = None
        unsqueeze_963: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        mul_488: "f32[8, 200, 16, 16]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_965: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_489: "f32[8, 200, 16, 16]" = torch.ops.aten.mul.Tensor(mul_488, unsqueeze_965);  mul_488 = unsqueeze_965 = None
        unsqueeze_966: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_967: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_400: "f32[8, 200, 16, 16]" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_967);  mul_489 = unsqueeze_967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_401: "f32[8, 200, 16, 16]" = torch.ops.aten.add.Tensor(add_400, 3)
        clamp_min_127: "f32[8, 200, 16, 16]" = torch.ops.aten.clamp_min.default(add_401, 0);  add_401 = None
        clamp_max_127: "f32[8, 200, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_127, 6);  clamp_min_127 = None
        mul_490: "f32[8, 200, 16, 16]" = torch.ops.aten.mul.Tensor(add_400, clamp_max_127);  add_400 = clamp_max_127 = None
        div_127: "f32[8, 200, 16, 16]" = torch.ops.aten.div.Tensor(mul_490, 6);  mul_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_168: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_127, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_127 = arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_968: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_969: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        sub_121: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_969);  convolution_168 = unsqueeze_969 = None
        add_402: "f32[72]" = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_121: "f32[72]" = torch.ops.aten.sqrt.default(add_402);  add_402 = None
        reciprocal_121: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_491: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_970: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_491, -1);  mul_491 = None
        unsqueeze_971: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        mul_492: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_973: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_493: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_492, unsqueeze_973);  mul_492 = unsqueeze_973 = None
        unsqueeze_974: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_975: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_403: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_493, unsqueeze_975);  mul_493 = unsqueeze_975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_169: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(add_403, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_976: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_977: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        sub_122: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_977);  convolution_169 = unsqueeze_977 = None
        add_404: "f32[216]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_122: "f32[216]" = torch.ops.aten.sqrt.default(add_404);  add_404 = None
        reciprocal_122: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_494: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_978: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_494, -1);  mul_494 = None
        unsqueeze_979: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        mul_495: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_981: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_496: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_495, unsqueeze_981);  mul_495 = unsqueeze_981 = None
        unsqueeze_982: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_983: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_405: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_983);  mul_496 = unsqueeze_983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_406: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_405, 3)
        clamp_min_128: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_406, 0);  add_406 = None
        clamp_max_128: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_128, 6);  clamp_min_128 = None
        mul_497: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_405, clamp_max_128);  add_405 = clamp_max_128 = None
        div_128: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_497, 6);  mul_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_170: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(div_128, arg201_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_128 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_984: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_985: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        sub_123: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_985);  convolution_170 = unsqueeze_985 = None
        add_407: "f32[216]" = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_123: "f32[216]" = torch.ops.aten.sqrt.default(add_407);  add_407 = None
        reciprocal_123: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_498: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_986: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_987: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        mul_499: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_989: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_500: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_989);  mul_499 = unsqueeze_989 = None
        unsqueeze_990: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_991: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_408: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_500, unsqueeze_991);  mul_500 = unsqueeze_991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_409: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_408, 3)
        clamp_min_129: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_409, 0);  add_409 = None
        clamp_max_129: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_129, 6);  clamp_min_129 = None
        mul_501: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_408, clamp_max_129);  add_408 = clamp_max_129 = None
        div_129: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_501, 6);  mul_501 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_171: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_129, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_129 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_992: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_993: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        sub_124: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_993);  convolution_171 = unsqueeze_993 = None
        add_410: "f32[72]" = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_124: "f32[72]" = torch.ops.aten.sqrt.default(add_410);  add_410 = None
        reciprocal_124: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_502: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_994: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_502, -1);  mul_502 = None
        unsqueeze_995: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        mul_503: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_997: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_504: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_503, unsqueeze_997);  mul_503 = unsqueeze_997 = None
        unsqueeze_998: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_999: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_411: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_504, unsqueeze_999);  mul_504 = unsqueeze_999 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_412: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(add_411, add_403);  add_411 = add_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_172: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(add_412, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg211_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1000: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_1001: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        sub_125: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1001);  convolution_172 = unsqueeze_1001 = None
        add_413: "f32[216]" = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_125: "f32[216]" = torch.ops.aten.sqrt.default(add_413);  add_413 = None
        reciprocal_125: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_505: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1002: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_505, -1);  mul_505 = None
        unsqueeze_1003: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        mul_506: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_1005: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_507: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_506, unsqueeze_1005);  mul_506 = unsqueeze_1005 = None
        unsqueeze_1006: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_1007: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_414: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_507, unsqueeze_1007);  mul_507 = unsqueeze_1007 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_415: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_414, 3)
        clamp_min_130: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_415, 0);  add_415 = None
        clamp_max_130: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_130, 6);  clamp_min_130 = None
        mul_508: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_414, clamp_max_130);  add_414 = clamp_max_130 = None
        div_130: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_508, 6);  mul_508 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_173: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(div_130, arg216_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_130 = arg216_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1008: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_1009: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        sub_126: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1009);  convolution_173 = unsqueeze_1009 = None
        add_416: "f32[216]" = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_126: "f32[216]" = torch.ops.aten.sqrt.default(add_416);  add_416 = None
        reciprocal_126: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_509: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1010: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_509, -1);  mul_509 = None
        unsqueeze_1011: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        mul_510: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_1013: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_511: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_510, unsqueeze_1013);  mul_510 = unsqueeze_1013 = None
        unsqueeze_1014: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_1015: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_417: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_511, unsqueeze_1015);  mul_511 = unsqueeze_1015 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_418: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_417, 3)
        clamp_min_131: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_418, 0);  add_418 = None
        clamp_max_131: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_131, 6);  clamp_min_131 = None
        mul_512: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_417, clamp_max_131);  add_417 = clamp_max_131 = None
        div_131: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_512, 6);  mul_512 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_174: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_131, arg221_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_131 = arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1016: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_1017: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        sub_127: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_174, unsqueeze_1017);  convolution_174 = unsqueeze_1017 = None
        add_419: "f32[72]" = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_127: "f32[72]" = torch.ops.aten.sqrt.default(add_419);  add_419 = None
        reciprocal_127: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_513: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1018: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_513, -1);  mul_513 = None
        unsqueeze_1019: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        mul_514: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_1021: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_515: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_514, unsqueeze_1021);  mul_514 = unsqueeze_1021 = None
        unsqueeze_1022: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_1023: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_420: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_515, unsqueeze_1023);  mul_515 = unsqueeze_1023 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_421: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(add_420, add_412);  add_420 = add_412 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_175: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(add_421, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1024: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_1025: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        sub_128: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_1025);  convolution_175 = unsqueeze_1025 = None
        add_422: "f32[216]" = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_128: "f32[216]" = torch.ops.aten.sqrt.default(add_422);  add_422 = None
        reciprocal_128: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_516: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1026: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_516, -1);  mul_516 = None
        unsqueeze_1027: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        mul_517: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_1029: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_518: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_1029);  mul_517 = unsqueeze_1029 = None
        unsqueeze_1030: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_1031: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_423: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_518, unsqueeze_1031);  mul_518 = unsqueeze_1031 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_424: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_423, 3)
        clamp_min_132: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_424, 0);  add_424 = None
        clamp_max_132: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_132, 6);  clamp_min_132 = None
        mul_519: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_423, clamp_max_132);  add_423 = clamp_max_132 = None
        div_132: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_519, 6);  mul_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_176: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(div_132, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_132 = arg231_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1032: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_1033: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        sub_129: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1033);  convolution_176 = unsqueeze_1033 = None
        add_425: "f32[216]" = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_129: "f32[216]" = torch.ops.aten.sqrt.default(add_425);  add_425 = None
        reciprocal_129: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_520: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1034: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_520, -1);  mul_520 = None
        unsqueeze_1035: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        mul_521: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_1037: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_522: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_521, unsqueeze_1037);  mul_521 = unsqueeze_1037 = None
        unsqueeze_1038: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_1039: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_426: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_522, unsqueeze_1039);  mul_522 = unsqueeze_1039 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_427: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_426, 3)
        clamp_min_133: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_427, 0);  add_427 = None
        clamp_max_133: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_133, 6);  clamp_min_133 = None
        mul_523: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_426, clamp_max_133);  add_426 = clamp_max_133 = None
        div_133: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_523, 6);  mul_523 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_177: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_133, arg236_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_133 = arg236_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1040: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1041: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        sub_130: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_1041);  convolution_177 = unsqueeze_1041 = None
        add_428: "f32[72]" = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_130: "f32[72]" = torch.ops.aten.sqrt.default(add_428);  add_428 = None
        reciprocal_130: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_524: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1042: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_524, -1);  mul_524 = None
        unsqueeze_1043: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        mul_525: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_1045: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_526: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_1045);  mul_525 = unsqueeze_1045 = None
        unsqueeze_1046: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_1047: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_429: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_526, unsqueeze_1047);  mul_526 = unsqueeze_1047 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_430: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(add_429, add_421);  add_429 = add_421 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_178: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(add_430, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1048: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1049: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        sub_131: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_1049);  convolution_178 = unsqueeze_1049 = None
        add_431: "f32[216]" = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_131: "f32[216]" = torch.ops.aten.sqrt.default(add_431);  add_431 = None
        reciprocal_131: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_527: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1050: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_527, -1);  mul_527 = None
        unsqueeze_1051: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        mul_528: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_1053: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_529: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_1053);  mul_528 = unsqueeze_1053 = None
        unsqueeze_1054: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_1055: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_432: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_529, unsqueeze_1055);  mul_529 = unsqueeze_1055 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_433: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_432, 3)
        clamp_min_134: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_433, 0);  add_433 = None
        clamp_max_134: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_134, 6);  clamp_min_134 = None
        mul_530: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_432, clamp_max_134);  add_432 = clamp_max_134 = None
        div_134: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_530, 6);  mul_530 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_179: "f32[8, 216, 16, 16]" = torch.ops.aten.convolution.default(div_134, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216);  div_134 = arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1056: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1057: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        sub_132: "f32[8, 216, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1057);  convolution_179 = unsqueeze_1057 = None
        add_434: "f32[216]" = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_132: "f32[216]" = torch.ops.aten.sqrt.default(add_434);  add_434 = None
        reciprocal_132: "f32[216]" = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_531: "f32[216]" = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1058: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(mul_531, -1);  mul_531 = None
        unsqueeze_1059: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        mul_532: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_1061: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_533: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_1061);  mul_532 = unsqueeze_1061 = None
        unsqueeze_1062: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_1063: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_435: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(mul_533, unsqueeze_1063);  mul_533 = unsqueeze_1063 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_436: "f32[8, 216, 16, 16]" = torch.ops.aten.add.Tensor(add_435, 3)
        clamp_min_135: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_min.default(add_436, 0);  add_436 = None
        clamp_max_135: "f32[8, 216, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_135, 6);  clamp_min_135 = None
        mul_534: "f32[8, 216, 16, 16]" = torch.ops.aten.mul.Tensor(add_435, clamp_max_135);  add_435 = clamp_max_135 = None
        div_135: "f32[8, 216, 16, 16]" = torch.ops.aten.div.Tensor(mul_534, 6);  mul_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_180: "f32[8, 72, 16, 16]" = torch.ops.aten.convolution.default(div_135, arg251_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_135 = arg251_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1064: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_1065: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        sub_133: "f32[8, 72, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_1065);  convolution_180 = unsqueeze_1065 = None
        add_437: "f32[72]" = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_133: "f32[72]" = torch.ops.aten.sqrt.default(add_437);  add_437 = None
        reciprocal_133: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_535: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1066: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_535, -1);  mul_535 = None
        unsqueeze_1067: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        mul_536: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1069: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_537: "f32[8, 72, 16, 16]" = torch.ops.aten.mul.Tensor(mul_536, unsqueeze_1069);  mul_536 = unsqueeze_1069 = None
        unsqueeze_1070: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_1071: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_438: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(mul_537, unsqueeze_1071);  mul_537 = unsqueeze_1071 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_439: "f32[8, 72, 16, 16]" = torch.ops.aten.add.Tensor(add_438, add_430);  add_438 = add_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_181: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_439, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_439 = arg256_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1072: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1073: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
        sub_134: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1073);  convolution_181 = unsqueeze_1073 = None
        add_440: "f32[360]" = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_134: "f32[360]" = torch.ops.aten.sqrt.default(add_440);  add_440 = None
        reciprocal_134: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
        mul_538: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
        unsqueeze_1074: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_538, -1);  mul_538 = None
        unsqueeze_1075: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
        mul_539: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
        unsqueeze_1076: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1077: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
        mul_540: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_1077);  mul_539 = unsqueeze_1077 = None
        unsqueeze_1078: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_1079: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
        add_441: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_540, unsqueeze_1079);  mul_540 = unsqueeze_1079 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_442: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_441, 3)
        clamp_min_136: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_442, 0);  add_442 = None
        clamp_max_136: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_136, 6);  clamp_min_136 = None
        mul_541: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_441, clamp_max_136);  add_441 = clamp_max_136 = None
        div_136: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_541, 6);  mul_541 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_182: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_136, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 360);  div_136 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1080: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1081: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
        sub_135: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1081);  convolution_182 = unsqueeze_1081 = None
        add_443: "f32[360]" = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_135: "f32[360]" = torch.ops.aten.sqrt.default(add_443);  add_443 = None
        reciprocal_135: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
        mul_542: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
        unsqueeze_1082: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_542, -1);  mul_542 = None
        unsqueeze_1083: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
        mul_543: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
        unsqueeze_1084: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1085: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
        mul_544: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_543, unsqueeze_1085);  mul_543 = unsqueeze_1085 = None
        unsqueeze_1086: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_1087: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
        add_444: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_544, unsqueeze_1087);  mul_544 = unsqueeze_1087 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_445: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_444, 3)
        clamp_min_137: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_445, 0);  add_445 = None
        clamp_max_137: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_137, 6);  clamp_min_137 = None
        mul_545: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_444, clamp_max_137);  add_444 = clamp_max_137 = None
        div_137: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_545, 6);  mul_545 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_24: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_137, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_183: "f32[8, 24, 1, 1]" = torch.ops.aten.convolution.default(mean_24, arg266_1, arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg266_1 = arg267_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_446: "f32[8, 24, 1, 1]" = torch.ops.aten.add.Tensor(convolution_183, 3)
        clamp_min_138: "f32[8, 24, 1, 1]" = torch.ops.aten.clamp_min.default(add_446, 0);  add_446 = None
        clamp_max_138: "f32[8, 24, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_138, 6);  clamp_min_138 = None
        mul_546: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_183, clamp_max_138);  convolution_183 = clamp_max_138 = None
        div_138: "f32[8, 24, 1, 1]" = torch.ops.aten.div.Tensor(mul_546, 6);  mul_546 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_184: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_138, arg268_1, arg269_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_138 = arg268_1 = arg269_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_447: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_184, 3);  convolution_184 = None
        clamp_min_139: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_447, 0);  add_447 = None
        clamp_max_139: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_139, 6);  clamp_min_139 = None
        div_139: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_139, 6);  clamp_max_139 = None
        mul_547: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_137, div_139);  div_137 = div_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_185: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_547, arg270_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_547 = arg270_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1088: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_1089: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
        sub_136: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_1089);  convolution_185 = unsqueeze_1089 = None
        add_448: "f32[120]" = torch.ops.aten.add.Tensor(arg272_1, 1e-05);  arg272_1 = None
        sqrt_136: "f32[120]" = torch.ops.aten.sqrt.default(add_448);  add_448 = None
        reciprocal_136: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
        mul_548: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
        unsqueeze_1090: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_548, -1);  mul_548 = None
        unsqueeze_1091: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
        mul_549: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
        unsqueeze_1092: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_1093: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
        mul_550: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_549, unsqueeze_1093);  mul_549 = unsqueeze_1093 = None
        unsqueeze_1094: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1095: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
        add_449: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_550, unsqueeze_1095);  mul_550 = unsqueeze_1095 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_186: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_449, arg275_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg275_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1096: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1097: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
        sub_137: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_1097);  convolution_186 = unsqueeze_1097 = None
        add_450: "f32[360]" = torch.ops.aten.add.Tensor(arg277_1, 1e-05);  arg277_1 = None
        sqrt_137: "f32[360]" = torch.ops.aten.sqrt.default(add_450);  add_450 = None
        reciprocal_137: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
        mul_551: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
        unsqueeze_1098: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_551, -1);  mul_551 = None
        unsqueeze_1099: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
        mul_552: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
        unsqueeze_1100: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_1101: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
        mul_553: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_552, unsqueeze_1101);  mul_552 = unsqueeze_1101 = None
        unsqueeze_1102: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1103: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
        add_451: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_553, unsqueeze_1103);  mul_553 = unsqueeze_1103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_452: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_451, 3)
        clamp_min_140: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_452, 0);  add_452 = None
        clamp_max_140: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_140, 6);  clamp_min_140 = None
        mul_554: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_451, clamp_max_140);  add_451 = clamp_max_140 = None
        div_140: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_554, 6);  mul_554 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_187: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_140, arg280_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_140 = arg280_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1104: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
        unsqueeze_1105: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
        sub_138: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1105);  convolution_187 = unsqueeze_1105 = None
        add_453: "f32[360]" = torch.ops.aten.add.Tensor(arg282_1, 1e-05);  arg282_1 = None
        sqrt_138: "f32[360]" = torch.ops.aten.sqrt.default(add_453);  add_453 = None
        reciprocal_138: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
        mul_555: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
        unsqueeze_1106: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_555, -1);  mul_555 = None
        unsqueeze_1107: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
        mul_556: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
        unsqueeze_1108: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_1109: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
        mul_557: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_556, unsqueeze_1109);  mul_556 = unsqueeze_1109 = None
        unsqueeze_1110: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_1111: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
        add_454: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_557, unsqueeze_1111);  mul_557 = unsqueeze_1111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_455: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_454, 3)
        clamp_min_141: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_455, 0);  add_455 = None
        clamp_max_141: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_141, 6);  clamp_min_141 = None
        mul_558: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_454, clamp_max_141);  add_454 = clamp_max_141 = None
        div_141: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_558, 6);  mul_558 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_25: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_141, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_188: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_25, arg285_1, arg286_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg285_1 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_456: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_188, 3)
        clamp_min_142: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_456, 0);  add_456 = None
        clamp_max_142: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_142, 6);  clamp_min_142 = None
        mul_559: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_188, clamp_max_142);  convolution_188 = clamp_max_142 = None
        div_142: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_559, 6);  mul_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_189: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_142, arg287_1, arg288_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_142 = arg287_1 = arg288_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_457: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_189, 3);  convolution_189 = None
        clamp_min_143: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_457, 0);  add_457 = None
        clamp_max_143: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_143, 6);  clamp_min_143 = None
        div_143: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_143, 6);  clamp_max_143 = None
        mul_560: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_141, div_143);  div_141 = div_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_190: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_560, arg289_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_560 = arg289_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1112: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1113: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        sub_139: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1113);  convolution_190 = unsqueeze_1113 = None
        add_458: "f32[120]" = torch.ops.aten.add.Tensor(arg291_1, 1e-05);  arg291_1 = None
        sqrt_139: "f32[120]" = torch.ops.aten.sqrt.default(add_458);  add_458 = None
        reciprocal_139: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_561: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1114: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_561, -1);  mul_561 = None
        unsqueeze_1115: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        mul_562: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
        unsqueeze_1116: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_1117: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_563: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_562, unsqueeze_1117);  mul_562 = unsqueeze_1117 = None
        unsqueeze_1118: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_1119: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_459: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_563, unsqueeze_1119);  mul_563 = unsqueeze_1119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_460: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_459, add_449);  add_459 = add_449 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_191: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_460, arg294_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg294_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1120: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_1121: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        sub_140: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_1121);  convolution_191 = unsqueeze_1121 = None
        add_461: "f32[360]" = torch.ops.aten.add.Tensor(arg296_1, 1e-05);  arg296_1 = None
        sqrt_140: "f32[360]" = torch.ops.aten.sqrt.default(add_461);  add_461 = None
        reciprocal_140: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_564: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1122: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_564, -1);  mul_564 = None
        unsqueeze_1123: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        mul_565: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
        unsqueeze_1124: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_1125: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_566: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_565, unsqueeze_1125);  mul_565 = unsqueeze_1125 = None
        unsqueeze_1126: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_1127: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_462: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_1127);  mul_566 = unsqueeze_1127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_463: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_462, 3)
        clamp_min_144: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_463, 0);  add_463 = None
        clamp_max_144: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_144, 6);  clamp_min_144 = None
        mul_567: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_462, clamp_max_144);  add_462 = clamp_max_144 = None
        div_144: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_567, 6);  mul_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_192: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_144, arg299_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_144 = arg299_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1128: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_1129: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        sub_141: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_192, unsqueeze_1129);  convolution_192 = unsqueeze_1129 = None
        add_464: "f32[360]" = torch.ops.aten.add.Tensor(arg301_1, 1e-05);  arg301_1 = None
        sqrt_141: "f32[360]" = torch.ops.aten.sqrt.default(add_464);  add_464 = None
        reciprocal_141: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_568: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1130: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_568, -1);  mul_568 = None
        unsqueeze_1131: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        mul_569: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
        unsqueeze_1132: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_1133: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_570: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_569, unsqueeze_1133);  mul_569 = unsqueeze_1133 = None
        unsqueeze_1134: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_1135: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_465: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_570, unsqueeze_1135);  mul_570 = unsqueeze_1135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_466: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_465, 3)
        clamp_min_145: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_466, 0);  add_466 = None
        clamp_max_145: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_145, 6);  clamp_min_145 = None
        mul_571: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_465, clamp_max_145);  add_465 = clamp_max_145 = None
        div_145: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_571, 6);  mul_571 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_26: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_145, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_193: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_26, arg304_1, arg305_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg304_1 = arg305_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_467: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_193, 3)
        clamp_min_146: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_467, 0);  add_467 = None
        clamp_max_146: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_146, 6);  clamp_min_146 = None
        mul_572: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_193, clamp_max_146);  convolution_193 = clamp_max_146 = None
        div_146: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_572, 6);  mul_572 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_194: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_146, arg306_1, arg307_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_146 = arg306_1 = arg307_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_468: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_194, 3);  convolution_194 = None
        clamp_min_147: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_468, 0);  add_468 = None
        clamp_max_147: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_147, 6);  clamp_min_147 = None
        div_147: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_147, 6);  clamp_max_147 = None
        mul_573: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_145, div_147);  div_145 = div_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_195: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_573, arg308_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_573 = arg308_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1136: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1137: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        sub_142: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_195, unsqueeze_1137);  convolution_195 = unsqueeze_1137 = None
        add_469: "f32[120]" = torch.ops.aten.add.Tensor(arg310_1, 1e-05);  arg310_1 = None
        sqrt_142: "f32[120]" = torch.ops.aten.sqrt.default(add_469);  add_469 = None
        reciprocal_142: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_574: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1138: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_574, -1);  mul_574 = None
        unsqueeze_1139: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        mul_575: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
        unsqueeze_1140: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_1141: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_576: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_575, unsqueeze_1141);  mul_575 = unsqueeze_1141 = None
        unsqueeze_1142: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_1143: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_470: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_576, unsqueeze_1143);  mul_576 = unsqueeze_1143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_471: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_470, add_460);  add_470 = add_460 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_196: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_471, arg313_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg313_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1144: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1145: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        sub_143: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1145);  convolution_196 = unsqueeze_1145 = None
        add_472: "f32[360]" = torch.ops.aten.add.Tensor(arg315_1, 1e-05);  arg315_1 = None
        sqrt_143: "f32[360]" = torch.ops.aten.sqrt.default(add_472);  add_472 = None
        reciprocal_143: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_577: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1146: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_577, -1);  mul_577 = None
        unsqueeze_1147: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        mul_578: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
        unsqueeze_1148: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg316_1, -1);  arg316_1 = None
        unsqueeze_1149: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_579: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_578, unsqueeze_1149);  mul_578 = unsqueeze_1149 = None
        unsqueeze_1150: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1151: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_473: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_579, unsqueeze_1151);  mul_579 = unsqueeze_1151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_474: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_473, 3)
        clamp_min_148: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_474, 0);  add_474 = None
        clamp_max_148: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_148, 6);  clamp_min_148 = None
        mul_580: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_473, clamp_max_148);  add_473 = clamp_max_148 = None
        div_148: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_580, 6);  mul_580 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_197: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_148, arg318_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_148 = arg318_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1152: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1153: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        sub_144: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_197, unsqueeze_1153);  convolution_197 = unsqueeze_1153 = None
        add_475: "f32[360]" = torch.ops.aten.add.Tensor(arg320_1, 1e-05);  arg320_1 = None
        sqrt_144: "f32[360]" = torch.ops.aten.sqrt.default(add_475);  add_475 = None
        reciprocal_144: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_581: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1154: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_581, -1);  mul_581 = None
        unsqueeze_1155: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        mul_582: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
        unsqueeze_1156: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_1157: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_583: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_582, unsqueeze_1157);  mul_582 = unsqueeze_1157 = None
        unsqueeze_1158: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1159: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_476: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_583, unsqueeze_1159);  mul_583 = unsqueeze_1159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_477: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_476, 3)
        clamp_min_149: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_477, 0);  add_477 = None
        clamp_max_149: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_149, 6);  clamp_min_149 = None
        mul_584: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_476, clamp_max_149);  add_476 = clamp_max_149 = None
        div_149: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_584, 6);  mul_584 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_27: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_149, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_198: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_27, arg323_1, arg324_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_27 = arg323_1 = arg324_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_478: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_198, 3)
        clamp_min_150: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_478, 0);  add_478 = None
        clamp_max_150: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_150, 6);  clamp_min_150 = None
        mul_585: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_198, clamp_max_150);  convolution_198 = clamp_max_150 = None
        div_150: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_585, 6);  mul_585 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_199: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_150, arg325_1, arg326_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_150 = arg325_1 = arg326_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_479: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_199, 3);  convolution_199 = None
        clamp_min_151: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_479, 0);  add_479 = None
        clamp_max_151: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_151, 6);  clamp_min_151 = None
        div_151: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_151, 6);  clamp_max_151 = None
        mul_586: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_149, div_151);  div_149 = div_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_200: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_586, arg327_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_586 = arg327_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1160: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_1161: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        sub_145: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_200, unsqueeze_1161);  convolution_200 = unsqueeze_1161 = None
        add_480: "f32[120]" = torch.ops.aten.add.Tensor(arg329_1, 1e-05);  arg329_1 = None
        sqrt_145: "f32[120]" = torch.ops.aten.sqrt.default(add_480);  add_480 = None
        reciprocal_145: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_587: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1162: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_587, -1);  mul_587 = None
        unsqueeze_1163: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        mul_588: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
        unsqueeze_1164: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_1165: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_589: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_1165);  mul_588 = unsqueeze_1165 = None
        unsqueeze_1166: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_1167: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_481: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_589, unsqueeze_1167);  mul_589 = unsqueeze_1167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_482: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_481, add_471);  add_481 = add_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_201: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_482, arg332_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg332_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1168: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_1169: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        sub_146: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_201, unsqueeze_1169);  convolution_201 = unsqueeze_1169 = None
        add_483: "f32[360]" = torch.ops.aten.add.Tensor(arg334_1, 1e-05);  arg334_1 = None
        sqrt_146: "f32[360]" = torch.ops.aten.sqrt.default(add_483);  add_483 = None
        reciprocal_146: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_590: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1170: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_590, -1);  mul_590 = None
        unsqueeze_1171: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        mul_591: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
        unsqueeze_1172: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_1173: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_592: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_591, unsqueeze_1173);  mul_591 = unsqueeze_1173 = None
        unsqueeze_1174: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_1175: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_484: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_592, unsqueeze_1175);  mul_592 = unsqueeze_1175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_485: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_484, 3)
        clamp_min_152: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_485, 0);  add_485 = None
        clamp_max_152: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_152, 6);  clamp_min_152 = None
        mul_593: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_484, clamp_max_152);  add_484 = clamp_max_152 = None
        div_152: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_593, 6);  mul_593 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_202: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_152, arg337_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_152 = arg337_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1176: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg338_1, -1);  arg338_1 = None
        unsqueeze_1177: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        sub_147: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1177);  convolution_202 = unsqueeze_1177 = None
        add_486: "f32[360]" = torch.ops.aten.add.Tensor(arg339_1, 1e-05);  arg339_1 = None
        sqrt_147: "f32[360]" = torch.ops.aten.sqrt.default(add_486);  add_486 = None
        reciprocal_147: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_594: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1178: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_594, -1);  mul_594 = None
        unsqueeze_1179: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        mul_595: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
        unsqueeze_1180: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_1181: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_596: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_1181);  mul_595 = unsqueeze_1181 = None
        unsqueeze_1182: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
        unsqueeze_1183: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_487: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_596, unsqueeze_1183);  mul_596 = unsqueeze_1183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_488: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_487, 3)
        clamp_min_153: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_488, 0);  add_488 = None
        clamp_max_153: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_153, 6);  clamp_min_153 = None
        mul_597: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_487, clamp_max_153);  add_487 = clamp_max_153 = None
        div_153: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_597, 6);  mul_597 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_28: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_153, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_203: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_28, arg342_1, arg343_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_28 = arg342_1 = arg343_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_489: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_203, 3)
        clamp_min_154: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_489, 0);  add_489 = None
        clamp_max_154: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_154, 6);  clamp_min_154 = None
        mul_598: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_203, clamp_max_154);  convolution_203 = clamp_max_154 = None
        div_154: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_598, 6);  mul_598 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_204: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_154, arg344_1, arg345_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_154 = arg344_1 = arg345_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_490: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_204, 3);  convolution_204 = None
        clamp_min_155: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_490, 0);  add_490 = None
        clamp_max_155: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_155, 6);  clamp_min_155 = None
        div_155: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_155, 6);  clamp_max_155 = None
        mul_599: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_153, div_155);  div_153 = div_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_205: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_599, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_599 = arg346_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1184: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_1185: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        sub_148: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_205, unsqueeze_1185);  convolution_205 = unsqueeze_1185 = None
        add_491: "f32[120]" = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_148: "f32[120]" = torch.ops.aten.sqrt.default(add_491);  add_491 = None
        reciprocal_148: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_600: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1186: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_600, -1);  mul_600 = None
        unsqueeze_1187: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        mul_601: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
        unsqueeze_1188: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1189: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_602: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_601, unsqueeze_1189);  mul_601 = unsqueeze_1189 = None
        unsqueeze_1190: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_1191: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_492: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_602, unsqueeze_1191);  mul_602 = unsqueeze_1191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_493: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_492, add_482);  add_492 = add_482 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_206: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(add_493, arg351_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg351_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1192: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1193: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        sub_149: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1193);  convolution_206 = unsqueeze_1193 = None
        add_494: "f32[360]" = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
        sqrt_149: "f32[360]" = torch.ops.aten.sqrt.default(add_494);  add_494 = None
        reciprocal_149: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_603: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1194: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_603, -1);  mul_603 = None
        unsqueeze_1195: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        mul_604: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1197: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_605: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_604, unsqueeze_1197);  mul_604 = unsqueeze_1197 = None
        unsqueeze_1198: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_1199: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_495: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_605, unsqueeze_1199);  mul_605 = unsqueeze_1199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_496: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_495, 3)
        clamp_min_156: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_496, 0);  add_496 = None
        clamp_max_156: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_156, 6);  clamp_min_156 = None
        mul_606: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_495, clamp_max_156);  add_495 = clamp_max_156 = None
        div_156: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_606, 6);  mul_606 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_207: "f32[8, 360, 16, 16]" = torch.ops.aten.convolution.default(div_156, arg356_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360);  div_156 = arg356_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1200: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1201: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        sub_150: "f32[8, 360, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_207, unsqueeze_1201);  convolution_207 = unsqueeze_1201 = None
        add_497: "f32[360]" = torch.ops.aten.add.Tensor(arg358_1, 1e-05);  arg358_1 = None
        sqrt_150: "f32[360]" = torch.ops.aten.sqrt.default(add_497);  add_497 = None
        reciprocal_150: "f32[360]" = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_607: "f32[360]" = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1202: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(mul_607, -1);  mul_607 = None
        unsqueeze_1203: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        mul_608: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1205: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_609: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(mul_608, unsqueeze_1205);  mul_608 = unsqueeze_1205 = None
        unsqueeze_1206: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_1207: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_498: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(mul_609, unsqueeze_1207);  mul_609 = unsqueeze_1207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_499: "f32[8, 360, 16, 16]" = torch.ops.aten.add.Tensor(add_498, 3)
        clamp_min_157: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_min.default(add_499, 0);  add_499 = None
        clamp_max_157: "f32[8, 360, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_157, 6);  clamp_min_157 = None
        mul_610: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(add_498, clamp_max_157);  add_498 = clamp_max_157 = None
        div_157: "f32[8, 360, 16, 16]" = torch.ops.aten.div.Tensor(mul_610, 6);  mul_610 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_29: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_157, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_208: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_29, arg361_1, arg362_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_29 = arg361_1 = arg362_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_500: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_208, 3)
        clamp_min_158: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_500, 0);  add_500 = None
        clamp_max_158: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_158, 6);  clamp_min_158 = None
        mul_611: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_208, clamp_max_158);  convolution_208 = clamp_max_158 = None
        div_158: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_611, 6);  mul_611 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_209: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_158, arg363_1, arg364_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_158 = arg363_1 = arg364_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_501: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_209, 3);  convolution_209 = None
        clamp_min_159: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_501, 0);  add_501 = None
        clamp_max_159: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_159, 6);  clamp_min_159 = None
        div_159: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_159, 6);  clamp_max_159 = None
        mul_612: "f32[8, 360, 16, 16]" = torch.ops.aten.mul.Tensor(div_157, div_159);  div_157 = div_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_210: "f32[8, 120, 16, 16]" = torch.ops.aten.convolution.default(mul_612, arg365_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_612 = arg365_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1208: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg366_1, -1);  arg366_1 = None
        unsqueeze_1209: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        sub_151: "f32[8, 120, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_210, unsqueeze_1209);  convolution_210 = unsqueeze_1209 = None
        add_502: "f32[120]" = torch.ops.aten.add.Tensor(arg367_1, 1e-05);  arg367_1 = None
        sqrt_151: "f32[120]" = torch.ops.aten.sqrt.default(add_502);  add_502 = None
        reciprocal_151: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_613: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1210: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_613, -1);  mul_613 = None
        unsqueeze_1211: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        mul_614: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
        unsqueeze_1213: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_615: "f32[8, 120, 16, 16]" = torch.ops.aten.mul.Tensor(mul_614, unsqueeze_1213);  mul_614 = unsqueeze_1213 = None
        unsqueeze_1214: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_1215: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_503: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_1215);  mul_615 = unsqueeze_1215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_504: "f32[8, 120, 16, 16]" = torch.ops.aten.add.Tensor(add_503, add_493);  add_503 = add_493 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_211: "f32[8, 720, 16, 16]" = torch.ops.aten.convolution.default(add_504, arg370_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_504 = arg370_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1216: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
        unsqueeze_1217: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        sub_152: "f32[8, 720, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_211, unsqueeze_1217);  convolution_211 = unsqueeze_1217 = None
        add_505: "f32[720]" = torch.ops.aten.add.Tensor(arg372_1, 1e-05);  arg372_1 = None
        sqrt_152: "f32[720]" = torch.ops.aten.sqrt.default(add_505);  add_505 = None
        reciprocal_152: "f32[720]" = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_616: "f32[720]" = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1218: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(mul_616, -1);  mul_616 = None
        unsqueeze_1219: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        mul_617: "f32[8, 720, 16, 16]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_1221: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_618: "f32[8, 720, 16, 16]" = torch.ops.aten.mul.Tensor(mul_617, unsqueeze_1221);  mul_617 = unsqueeze_1221 = None
        unsqueeze_1222: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1223: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_506: "f32[8, 720, 16, 16]" = torch.ops.aten.add.Tensor(mul_618, unsqueeze_1223);  mul_618 = unsqueeze_1223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_507: "f32[8, 720, 16, 16]" = torch.ops.aten.add.Tensor(add_506, 3)
        clamp_min_160: "f32[8, 720, 16, 16]" = torch.ops.aten.clamp_min.default(add_507, 0);  add_507 = None
        clamp_max_160: "f32[8, 720, 16, 16]" = torch.ops.aten.clamp_max.default(clamp_min_160, 6);  clamp_min_160 = None
        mul_619: "f32[8, 720, 16, 16]" = torch.ops.aten.mul.Tensor(add_506, clamp_max_160);  add_506 = clamp_max_160 = None
        div_160: "f32[8, 720, 16, 16]" = torch.ops.aten.div.Tensor(mul_619, 6);  mul_619 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_212: "f32[8, 720, 8, 8]" = torch.ops.aten.convolution.default(div_160, arg375_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 720);  div_160 = arg375_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1224: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
        unsqueeze_1225: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        sub_153: "f32[8, 720, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1225);  convolution_212 = unsqueeze_1225 = None
        add_508: "f32[720]" = torch.ops.aten.add.Tensor(arg377_1, 1e-05);  arg377_1 = None
        sqrt_153: "f32[720]" = torch.ops.aten.sqrt.default(add_508);  add_508 = None
        reciprocal_153: "f32[720]" = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_620: "f32[720]" = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1226: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(mul_620, -1);  mul_620 = None
        unsqueeze_1227: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        mul_621: "f32[8, 720, 8, 8]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_1229: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_622: "f32[8, 720, 8, 8]" = torch.ops.aten.mul.Tensor(mul_621, unsqueeze_1229);  mul_621 = unsqueeze_1229 = None
        unsqueeze_1230: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1231: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_509: "f32[8, 720, 8, 8]" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_1231);  mul_622 = unsqueeze_1231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_510: "f32[8, 720, 8, 8]" = torch.ops.aten.add.Tensor(add_509, 3)
        clamp_min_161: "f32[8, 720, 8, 8]" = torch.ops.aten.clamp_min.default(add_510, 0);  add_510 = None
        clamp_max_161: "f32[8, 720, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_161, 6);  clamp_min_161 = None
        mul_623: "f32[8, 720, 8, 8]" = torch.ops.aten.mul.Tensor(add_509, clamp_max_161);  add_509 = clamp_max_161 = None
        div_161: "f32[8, 720, 8, 8]" = torch.ops.aten.div.Tensor(mul_623, 6);  mul_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_30: "f32[8, 720, 1, 1]" = torch.ops.aten.mean.dim(div_161, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_213: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_30, arg380_1, arg381_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_30 = arg380_1 = arg381_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_511: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_213, 3)
        clamp_min_162: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_511, 0);  add_511 = None
        clamp_max_162: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_162, 6);  clamp_min_162 = None
        mul_624: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_213, clamp_max_162);  convolution_213 = clamp_max_162 = None
        div_162: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_624, 6);  mul_624 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_214: "f32[8, 720, 1, 1]" = torch.ops.aten.convolution.default(div_162, arg382_1, arg383_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_162 = arg382_1 = arg383_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_512: "f32[8, 720, 1, 1]" = torch.ops.aten.add.Tensor(convolution_214, 3);  convolution_214 = None
        clamp_min_163: "f32[8, 720, 1, 1]" = torch.ops.aten.clamp_min.default(add_512, 0);  add_512 = None
        clamp_max_163: "f32[8, 720, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_163, 6);  clamp_min_163 = None
        div_163: "f32[8, 720, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_163, 6);  clamp_max_163 = None
        mul_625: "f32[8, 720, 8, 8]" = torch.ops.aten.mul.Tensor(div_161, div_163);  div_161 = div_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_215: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_625, arg384_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_625 = arg384_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1232: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_1233: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        sub_154: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_215, unsqueeze_1233);  convolution_215 = unsqueeze_1233 = None
        add_513: "f32[184]" = torch.ops.aten.add.Tensor(arg386_1, 1e-05);  arg386_1 = None
        sqrt_154: "f32[184]" = torch.ops.aten.sqrt.default(add_513);  add_513 = None
        reciprocal_154: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_626: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1234: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_626, -1);  mul_626 = None
        unsqueeze_1235: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        mul_627: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg387_1, -1);  arg387_1 = None
        unsqueeze_1237: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_628: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_627, unsqueeze_1237);  mul_627 = unsqueeze_1237 = None
        unsqueeze_1238: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_1239: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_514: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_628, unsqueeze_1239);  mul_628 = unsqueeze_1239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_216: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_514, arg389_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg389_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1240: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_1241: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        sub_155: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1241);  convolution_216 = unsqueeze_1241 = None
        add_515: "f32[736]" = torch.ops.aten.add.Tensor(arg391_1, 1e-05);  arg391_1 = None
        sqrt_155: "f32[736]" = torch.ops.aten.sqrt.default(add_515);  add_515 = None
        reciprocal_155: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_629: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1242: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_629, -1);  mul_629 = None
        unsqueeze_1243: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        mul_630: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
        unsqueeze_1244: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg392_1, -1);  arg392_1 = None
        unsqueeze_1245: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_631: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_1245);  mul_630 = unsqueeze_1245 = None
        unsqueeze_1246: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg393_1, -1);  arg393_1 = None
        unsqueeze_1247: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_516: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_631, unsqueeze_1247);  mul_631 = unsqueeze_1247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_517: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_516, 3)
        clamp_min_164: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_517, 0);  add_517 = None
        clamp_max_164: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_164, 6);  clamp_min_164 = None
        mul_632: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_516, clamp_max_164);  add_516 = clamp_max_164 = None
        div_164: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_632, 6);  mul_632 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_217: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_164, arg394_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_164 = arg394_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1248: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_1249: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        sub_156: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_217, unsqueeze_1249);  convolution_217 = unsqueeze_1249 = None
        add_518: "f32[736]" = torch.ops.aten.add.Tensor(arg396_1, 1e-05);  arg396_1 = None
        sqrt_156: "f32[736]" = torch.ops.aten.sqrt.default(add_518);  add_518 = None
        reciprocal_156: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_633: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1250: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_1251: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        mul_634: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
        unsqueeze_1252: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_1253: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_635: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_1253);  mul_634 = unsqueeze_1253 = None
        unsqueeze_1254: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_1255: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_519: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_635, unsqueeze_1255);  mul_635 = unsqueeze_1255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_520: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_519, 3)
        clamp_min_165: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_520, 0);  add_520 = None
        clamp_max_165: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_165, 6);  clamp_min_165 = None
        mul_636: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_519, clamp_max_165);  add_519 = clamp_max_165 = None
        div_165: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_636, 6);  mul_636 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_31: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_165, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_218: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_31, arg399_1, arg400_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_31 = arg399_1 = arg400_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_521: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_218, 3)
        clamp_min_166: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_521, 0);  add_521 = None
        clamp_max_166: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_166, 6);  clamp_min_166 = None
        mul_637: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_218, clamp_max_166);  convolution_218 = clamp_max_166 = None
        div_166: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_637, 6);  mul_637 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_219: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_166, arg401_1, arg402_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_166 = arg401_1 = arg402_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_522: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_219, 3);  convolution_219 = None
        clamp_min_167: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_522, 0);  add_522 = None
        clamp_max_167: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_167, 6);  clamp_min_167 = None
        div_167: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_167, 6);  clamp_max_167 = None
        mul_638: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_165, div_167);  div_165 = div_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_220: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_638, arg403_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_638 = arg403_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1256: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_1257: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        sub_157: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_220, unsqueeze_1257);  convolution_220 = unsqueeze_1257 = None
        add_523: "f32[184]" = torch.ops.aten.add.Tensor(arg405_1, 1e-05);  arg405_1 = None
        sqrt_157: "f32[184]" = torch.ops.aten.sqrt.default(add_523);  add_523 = None
        reciprocal_157: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_639: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1258: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_639, -1);  mul_639 = None
        unsqueeze_1259: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        mul_640: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
        unsqueeze_1260: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
        unsqueeze_1261: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_641: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_640, unsqueeze_1261);  mul_640 = unsqueeze_1261 = None
        unsqueeze_1262: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg407_1, -1);  arg407_1 = None
        unsqueeze_1263: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_524: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_641, unsqueeze_1263);  mul_641 = unsqueeze_1263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_525: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_524, add_514);  add_524 = add_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_221: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_525, arg408_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg408_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1264: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_1265: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        sub_158: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_221, unsqueeze_1265);  convolution_221 = unsqueeze_1265 = None
        add_526: "f32[736]" = torch.ops.aten.add.Tensor(arg410_1, 1e-05);  arg410_1 = None
        sqrt_158: "f32[736]" = torch.ops.aten.sqrt.default(add_526);  add_526 = None
        reciprocal_158: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_642: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1266: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_642, -1);  mul_642 = None
        unsqueeze_1267: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        mul_643: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
        unsqueeze_1268: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg411_1, -1);  arg411_1 = None
        unsqueeze_1269: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_644: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_1269);  mul_643 = unsqueeze_1269 = None
        unsqueeze_1270: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg412_1, -1);  arg412_1 = None
        unsqueeze_1271: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_527: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_644, unsqueeze_1271);  mul_644 = unsqueeze_1271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_528: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_527, 3)
        clamp_min_168: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_528, 0);  add_528 = None
        clamp_max_168: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_168, 6);  clamp_min_168 = None
        mul_645: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_527, clamp_max_168);  add_527 = clamp_max_168 = None
        div_168: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_645, 6);  mul_645 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_222: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_168, arg413_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_168 = arg413_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1272: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_1273: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        sub_159: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_222, unsqueeze_1273);  convolution_222 = unsqueeze_1273 = None
        add_529: "f32[736]" = torch.ops.aten.add.Tensor(arg415_1, 1e-05);  arg415_1 = None
        sqrt_159: "f32[736]" = torch.ops.aten.sqrt.default(add_529);  add_529 = None
        reciprocal_159: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_646: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1274: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_646, -1);  mul_646 = None
        unsqueeze_1275: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        mul_647: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
        unsqueeze_1276: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
        unsqueeze_1277: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_648: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_647, unsqueeze_1277);  mul_647 = unsqueeze_1277 = None
        unsqueeze_1278: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg417_1, -1);  arg417_1 = None
        unsqueeze_1279: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_530: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_648, unsqueeze_1279);  mul_648 = unsqueeze_1279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_531: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_530, 3)
        clamp_min_169: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_531, 0);  add_531 = None
        clamp_max_169: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_169, 6);  clamp_min_169 = None
        mul_649: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_530, clamp_max_169);  add_530 = clamp_max_169 = None
        div_169: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_649, 6);  mul_649 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_32: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_169, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_223: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_32, arg418_1, arg419_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_32 = arg418_1 = arg419_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_532: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_223, 3)
        clamp_min_170: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_532, 0);  add_532 = None
        clamp_max_170: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_170, 6);  clamp_min_170 = None
        mul_650: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_223, clamp_max_170);  convolution_223 = clamp_max_170 = None
        div_170: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_650, 6);  mul_650 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_224: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_170, arg420_1, arg421_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_170 = arg420_1 = arg421_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_533: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_224, 3);  convolution_224 = None
        clamp_min_171: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_533, 0);  add_533 = None
        clamp_max_171: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_171, 6);  clamp_min_171 = None
        div_171: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_171, 6);  clamp_max_171 = None
        mul_651: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_169, div_171);  div_169 = div_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_225: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_651, arg422_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_651 = arg422_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1280: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_1281: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, -1);  unsqueeze_1280 = None
        sub_160: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_225, unsqueeze_1281);  convolution_225 = unsqueeze_1281 = None
        add_534: "f32[184]" = torch.ops.aten.add.Tensor(arg424_1, 1e-05);  arg424_1 = None
        sqrt_160: "f32[184]" = torch.ops.aten.sqrt.default(add_534);  add_534 = None
        reciprocal_160: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_160);  sqrt_160 = None
        mul_652: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_160, 1);  reciprocal_160 = None
        unsqueeze_1282: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_652, -1);  mul_652 = None
        unsqueeze_1283: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, -1);  unsqueeze_1282 = None
        mul_653: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_1283);  sub_160 = unsqueeze_1283 = None
        unsqueeze_1284: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_1285: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, -1);  unsqueeze_1284 = None
        mul_654: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_653, unsqueeze_1285);  mul_653 = unsqueeze_1285 = None
        unsqueeze_1286: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg426_1, -1);  arg426_1 = None
        unsqueeze_1287: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, -1);  unsqueeze_1286 = None
        add_535: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_654, unsqueeze_1287);  mul_654 = unsqueeze_1287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_536: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_535, add_525);  add_535 = add_525 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_226: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_536, arg427_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg427_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1288: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
        unsqueeze_1289: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, -1);  unsqueeze_1288 = None
        sub_161: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_226, unsqueeze_1289);  convolution_226 = unsqueeze_1289 = None
        add_537: "f32[736]" = torch.ops.aten.add.Tensor(arg429_1, 1e-05);  arg429_1 = None
        sqrt_161: "f32[736]" = torch.ops.aten.sqrt.default(add_537);  add_537 = None
        reciprocal_161: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_161);  sqrt_161 = None
        mul_655: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_161, 1);  reciprocal_161 = None
        unsqueeze_1290: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_655, -1);  mul_655 = None
        unsqueeze_1291: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, -1);  unsqueeze_1290 = None
        mul_656: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_1291);  sub_161 = unsqueeze_1291 = None
        unsqueeze_1292: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_1293: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, -1);  unsqueeze_1292 = None
        mul_657: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_656, unsqueeze_1293);  mul_656 = unsqueeze_1293 = None
        unsqueeze_1294: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
        unsqueeze_1295: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, -1);  unsqueeze_1294 = None
        add_538: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_1295);  mul_657 = unsqueeze_1295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_539: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_538, 3)
        clamp_min_172: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_539, 0);  add_539 = None
        clamp_max_172: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_172, 6);  clamp_min_172 = None
        mul_658: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_538, clamp_max_172);  add_538 = clamp_max_172 = None
        div_172: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_658, 6);  mul_658 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_227: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_172, arg432_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_172 = arg432_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1296: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg433_1, -1);  arg433_1 = None
        unsqueeze_1297: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, -1);  unsqueeze_1296 = None
        sub_162: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_227, unsqueeze_1297);  convolution_227 = unsqueeze_1297 = None
        add_540: "f32[736]" = torch.ops.aten.add.Tensor(arg434_1, 1e-05);  arg434_1 = None
        sqrt_162: "f32[736]" = torch.ops.aten.sqrt.default(add_540);  add_540 = None
        reciprocal_162: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_162);  sqrt_162 = None
        mul_659: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_162, 1);  reciprocal_162 = None
        unsqueeze_1298: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_659, -1);  mul_659 = None
        unsqueeze_1299: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, -1);  unsqueeze_1298 = None
        mul_660: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_1299);  sub_162 = unsqueeze_1299 = None
        unsqueeze_1300: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg435_1, -1);  arg435_1 = None
        unsqueeze_1301: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, -1);  unsqueeze_1300 = None
        mul_661: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_660, unsqueeze_1301);  mul_660 = unsqueeze_1301 = None
        unsqueeze_1302: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg436_1, -1);  arg436_1 = None
        unsqueeze_1303: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, -1);  unsqueeze_1302 = None
        add_541: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_661, unsqueeze_1303);  mul_661 = unsqueeze_1303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_542: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_541, 3)
        clamp_min_173: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_542, 0);  add_542 = None
        clamp_max_173: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_173, 6);  clamp_min_173 = None
        mul_662: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_541, clamp_max_173);  add_541 = clamp_max_173 = None
        div_173: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_662, 6);  mul_662 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_33: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_173, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_228: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_33, arg437_1, arg438_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_33 = arg437_1 = arg438_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_543: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_228, 3)
        clamp_min_174: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_543, 0);  add_543 = None
        clamp_max_174: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_174, 6);  clamp_min_174 = None
        mul_663: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_228, clamp_max_174);  convolution_228 = clamp_max_174 = None
        div_174: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_663, 6);  mul_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_229: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_174, arg439_1, arg440_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_174 = arg439_1 = arg440_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_544: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_229, 3);  convolution_229 = None
        clamp_min_175: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_544, 0);  add_544 = None
        clamp_max_175: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_175, 6);  clamp_min_175 = None
        div_175: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_175, 6);  clamp_max_175 = None
        mul_664: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_173, div_175);  div_173 = div_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_230: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_664, arg441_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_664 = arg441_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1304: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg442_1, -1);  arg442_1 = None
        unsqueeze_1305: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, -1);  unsqueeze_1304 = None
        sub_163: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_230, unsqueeze_1305);  convolution_230 = unsqueeze_1305 = None
        add_545: "f32[184]" = torch.ops.aten.add.Tensor(arg443_1, 1e-05);  arg443_1 = None
        sqrt_163: "f32[184]" = torch.ops.aten.sqrt.default(add_545);  add_545 = None
        reciprocal_163: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_163);  sqrt_163 = None
        mul_665: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_163, 1);  reciprocal_163 = None
        unsqueeze_1306: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_665, -1);  mul_665 = None
        unsqueeze_1307: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, -1);  unsqueeze_1306 = None
        mul_666: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_1307);  sub_163 = unsqueeze_1307 = None
        unsqueeze_1308: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_1309: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, -1);  unsqueeze_1308 = None
        mul_667: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_666, unsqueeze_1309);  mul_666 = unsqueeze_1309 = None
        unsqueeze_1310: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
        unsqueeze_1311: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, -1);  unsqueeze_1310 = None
        add_546: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_667, unsqueeze_1311);  mul_667 = unsqueeze_1311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_547: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_546, add_536);  add_546 = add_536 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_231: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_547, arg446_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg446_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1312: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg447_1, -1);  arg447_1 = None
        unsqueeze_1313: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, -1);  unsqueeze_1312 = None
        sub_164: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_231, unsqueeze_1313);  convolution_231 = unsqueeze_1313 = None
        add_548: "f32[736]" = torch.ops.aten.add.Tensor(arg448_1, 1e-05);  arg448_1 = None
        sqrt_164: "f32[736]" = torch.ops.aten.sqrt.default(add_548);  add_548 = None
        reciprocal_164: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_164);  sqrt_164 = None
        mul_668: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_164, 1);  reciprocal_164 = None
        unsqueeze_1314: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_668, -1);  mul_668 = None
        unsqueeze_1315: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, -1);  unsqueeze_1314 = None
        mul_669: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_1315);  sub_164 = unsqueeze_1315 = None
        unsqueeze_1316: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_1317: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, -1);  unsqueeze_1316 = None
        mul_670: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_669, unsqueeze_1317);  mul_669 = unsqueeze_1317 = None
        unsqueeze_1318: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
        unsqueeze_1319: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, -1);  unsqueeze_1318 = None
        add_549: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_670, unsqueeze_1319);  mul_670 = unsqueeze_1319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_550: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_549, 3)
        clamp_min_176: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_550, 0);  add_550 = None
        clamp_max_176: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_176, 6);  clamp_min_176 = None
        mul_671: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_549, clamp_max_176);  add_549 = clamp_max_176 = None
        div_176: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_671, 6);  mul_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_232: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_176, arg451_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_176 = arg451_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1320: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg452_1, -1);  arg452_1 = None
        unsqueeze_1321: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, -1);  unsqueeze_1320 = None
        sub_165: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_232, unsqueeze_1321);  convolution_232 = unsqueeze_1321 = None
        add_551: "f32[736]" = torch.ops.aten.add.Tensor(arg453_1, 1e-05);  arg453_1 = None
        sqrt_165: "f32[736]" = torch.ops.aten.sqrt.default(add_551);  add_551 = None
        reciprocal_165: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_165);  sqrt_165 = None
        mul_672: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_165, 1);  reciprocal_165 = None
        unsqueeze_1322: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_672, -1);  mul_672 = None
        unsqueeze_1323: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, -1);  unsqueeze_1322 = None
        mul_673: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_1323);  sub_165 = unsqueeze_1323 = None
        unsqueeze_1324: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_1325: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, -1);  unsqueeze_1324 = None
        mul_674: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_673, unsqueeze_1325);  mul_673 = unsqueeze_1325 = None
        unsqueeze_1326: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_1327: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, -1);  unsqueeze_1326 = None
        add_552: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_674, unsqueeze_1327);  mul_674 = unsqueeze_1327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_553: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_552, 3)
        clamp_min_177: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_553, 0);  add_553 = None
        clamp_max_177: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_177, 6);  clamp_min_177 = None
        mul_675: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_552, clamp_max_177);  add_552 = clamp_max_177 = None
        div_177: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_675, 6);  mul_675 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_34: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_177, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_233: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_34, arg456_1, arg457_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_34 = arg456_1 = arg457_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_554: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_233, 3)
        clamp_min_178: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_554, 0);  add_554 = None
        clamp_max_178: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_178, 6);  clamp_min_178 = None
        mul_676: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_233, clamp_max_178);  convolution_233 = clamp_max_178 = None
        div_178: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_676, 6);  mul_676 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_234: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_178, arg458_1, arg459_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_178 = arg458_1 = arg459_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_555: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_234, 3);  convolution_234 = None
        clamp_min_179: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_555, 0);  add_555 = None
        clamp_max_179: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_179, 6);  clamp_min_179 = None
        div_179: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_179, 6);  clamp_max_179 = None
        mul_677: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_177, div_179);  div_177 = div_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_235: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_677, arg460_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_677 = arg460_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1328: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg461_1, -1);  arg461_1 = None
        unsqueeze_1329: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, -1);  unsqueeze_1328 = None
        sub_166: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_235, unsqueeze_1329);  convolution_235 = unsqueeze_1329 = None
        add_556: "f32[184]" = torch.ops.aten.add.Tensor(arg462_1, 1e-05);  arg462_1 = None
        sqrt_166: "f32[184]" = torch.ops.aten.sqrt.default(add_556);  add_556 = None
        reciprocal_166: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_166);  sqrt_166 = None
        mul_678: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_166, 1);  reciprocal_166 = None
        unsqueeze_1330: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_678, -1);  mul_678 = None
        unsqueeze_1331: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, -1);  unsqueeze_1330 = None
        mul_679: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_1331);  sub_166 = unsqueeze_1331 = None
        unsqueeze_1332: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg463_1, -1);  arg463_1 = None
        unsqueeze_1333: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, -1);  unsqueeze_1332 = None
        mul_680: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_1333);  mul_679 = unsqueeze_1333 = None
        unsqueeze_1334: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_1335: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, -1);  unsqueeze_1334 = None
        add_557: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_680, unsqueeze_1335);  mul_680 = unsqueeze_1335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_558: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_557, add_547);  add_557 = add_547 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_236: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(add_558, arg465_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg465_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1336: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg466_1, -1);  arg466_1 = None
        unsqueeze_1337: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, -1);  unsqueeze_1336 = None
        sub_167: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_236, unsqueeze_1337);  convolution_236 = unsqueeze_1337 = None
        add_559: "f32[736]" = torch.ops.aten.add.Tensor(arg467_1, 1e-05);  arg467_1 = None
        sqrt_167: "f32[736]" = torch.ops.aten.sqrt.default(add_559);  add_559 = None
        reciprocal_167: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_167);  sqrt_167 = None
        mul_681: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_167, 1);  reciprocal_167 = None
        unsqueeze_1338: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_681, -1);  mul_681 = None
        unsqueeze_1339: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, -1);  unsqueeze_1338 = None
        mul_682: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_1339);  sub_167 = unsqueeze_1339 = None
        unsqueeze_1340: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg468_1, -1);  arg468_1 = None
        unsqueeze_1341: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, -1);  unsqueeze_1340 = None
        mul_683: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_682, unsqueeze_1341);  mul_682 = unsqueeze_1341 = None
        unsqueeze_1342: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_1343: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, -1);  unsqueeze_1342 = None
        add_560: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_683, unsqueeze_1343);  mul_683 = unsqueeze_1343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_561: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_560, 3)
        clamp_min_180: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_561, 0);  add_561 = None
        clamp_max_180: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_180, 6);  clamp_min_180 = None
        mul_684: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_560, clamp_max_180);  add_560 = clamp_max_180 = None
        div_180: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_684, 6);  mul_684 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_237: "f32[8, 736, 8, 8]" = torch.ops.aten.convolution.default(div_180, arg470_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736);  div_180 = arg470_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1344: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg471_1, -1);  arg471_1 = None
        unsqueeze_1345: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, -1);  unsqueeze_1344 = None
        sub_168: "f32[8, 736, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_237, unsqueeze_1345);  convolution_237 = unsqueeze_1345 = None
        add_562: "f32[736]" = torch.ops.aten.add.Tensor(arg472_1, 1e-05);  arg472_1 = None
        sqrt_168: "f32[736]" = torch.ops.aten.sqrt.default(add_562);  add_562 = None
        reciprocal_168: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_168);  sqrt_168 = None
        mul_685: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_168, 1);  reciprocal_168 = None
        unsqueeze_1346: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_685, -1);  mul_685 = None
        unsqueeze_1347: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, -1);  unsqueeze_1346 = None
        mul_686: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_1347);  sub_168 = unsqueeze_1347 = None
        unsqueeze_1348: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
        unsqueeze_1349: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, -1);  unsqueeze_1348 = None
        mul_687: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(mul_686, unsqueeze_1349);  mul_686 = unsqueeze_1349 = None
        unsqueeze_1350: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_1351: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, -1);  unsqueeze_1350 = None
        add_563: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(mul_687, unsqueeze_1351);  mul_687 = unsqueeze_1351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_564: "f32[8, 736, 8, 8]" = torch.ops.aten.add.Tensor(add_563, 3)
        clamp_min_181: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_min.default(add_564, 0);  add_564 = None
        clamp_max_181: "f32[8, 736, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_181, 6);  clamp_min_181 = None
        mul_688: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(add_563, clamp_max_181);  add_563 = clamp_max_181 = None
        div_181: "f32[8, 736, 8, 8]" = torch.ops.aten.div.Tensor(mul_688, 6);  mul_688 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_35: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_181, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_238: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_35, arg475_1, arg476_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_35 = arg475_1 = arg476_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_565: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_238, 3)
        clamp_min_182: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_565, 0);  add_565 = None
        clamp_max_182: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_182, 6);  clamp_min_182 = None
        mul_689: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_238, clamp_max_182);  convolution_238 = clamp_max_182 = None
        div_182: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_689, 6);  mul_689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_239: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_182, arg477_1, arg478_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_182 = arg477_1 = arg478_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_566: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_239, 3);  convolution_239 = None
        clamp_min_183: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_566, 0);  add_566 = None
        clamp_max_183: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_183, 6);  clamp_min_183 = None
        div_183: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_183, 6);  clamp_max_183 = None
        mul_690: "f32[8, 736, 8, 8]" = torch.ops.aten.mul.Tensor(div_181, div_183);  div_181 = div_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_240: "f32[8, 184, 8, 8]" = torch.ops.aten.convolution.default(mul_690, arg479_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_690 = arg479_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1352: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
        unsqueeze_1353: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, -1);  unsqueeze_1352 = None
        sub_169: "f32[8, 184, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_1353);  convolution_240 = unsqueeze_1353 = None
        add_567: "f32[184]" = torch.ops.aten.add.Tensor(arg481_1, 1e-05);  arg481_1 = None
        sqrt_169: "f32[184]" = torch.ops.aten.sqrt.default(add_567);  add_567 = None
        reciprocal_169: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_169);  sqrt_169 = None
        mul_691: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_169, 1);  reciprocal_169 = None
        unsqueeze_1354: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_691, -1);  mul_691 = None
        unsqueeze_1355: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, -1);  unsqueeze_1354 = None
        mul_692: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_1355);  sub_169 = unsqueeze_1355 = None
        unsqueeze_1356: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg482_1, -1);  arg482_1 = None
        unsqueeze_1357: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, -1);  unsqueeze_1356 = None
        mul_693: "f32[8, 184, 8, 8]" = torch.ops.aten.mul.Tensor(mul_692, unsqueeze_1357);  mul_692 = unsqueeze_1357 = None
        unsqueeze_1358: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg483_1, -1);  arg483_1 = None
        unsqueeze_1359: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, -1);  unsqueeze_1358 = None
        add_568: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(mul_693, unsqueeze_1359);  mul_693 = unsqueeze_1359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_569: "f32[8, 184, 8, 8]" = torch.ops.aten.add.Tensor(add_568, add_558);  add_568 = add_558 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_241: "f32[8, 1104, 8, 8]" = torch.ops.aten.convolution.default(add_569, arg484_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_569 = arg484_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1360: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
        unsqueeze_1361: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, -1);  unsqueeze_1360 = None
        sub_170: "f32[8, 1104, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_241, unsqueeze_1361);  convolution_241 = unsqueeze_1361 = None
        add_570: "f32[1104]" = torch.ops.aten.add.Tensor(arg486_1, 1e-05);  arg486_1 = None
        sqrt_170: "f32[1104]" = torch.ops.aten.sqrt.default(add_570);  add_570 = None
        reciprocal_170: "f32[1104]" = torch.ops.aten.reciprocal.default(sqrt_170);  sqrt_170 = None
        mul_694: "f32[1104]" = torch.ops.aten.mul.Tensor(reciprocal_170, 1);  reciprocal_170 = None
        unsqueeze_1362: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(mul_694, -1);  mul_694 = None
        unsqueeze_1363: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, -1);  unsqueeze_1362 = None
        mul_695: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_1363);  sub_170 = unsqueeze_1363 = None
        unsqueeze_1364: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg487_1, -1);  arg487_1 = None
        unsqueeze_1365: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, -1);  unsqueeze_1364 = None
        mul_696: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(mul_695, unsqueeze_1365);  mul_695 = unsqueeze_1365 = None
        unsqueeze_1366: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg488_1, -1);  arg488_1 = None
        unsqueeze_1367: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, -1);  unsqueeze_1366 = None
        add_571: "f32[8, 1104, 8, 8]" = torch.ops.aten.add.Tensor(mul_696, unsqueeze_1367);  mul_696 = unsqueeze_1367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_572: "f32[8, 1104, 8, 8]" = torch.ops.aten.add.Tensor(add_571, 3)
        clamp_min_184: "f32[8, 1104, 8, 8]" = torch.ops.aten.clamp_min.default(add_572, 0);  add_572 = None
        clamp_max_184: "f32[8, 1104, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_184, 6);  clamp_min_184 = None
        mul_697: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(add_571, clamp_max_184);  add_571 = clamp_max_184 = None
        div_184: "f32[8, 1104, 8, 8]" = torch.ops.aten.div.Tensor(mul_697, 6);  mul_697 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_242: "f32[8, 1104, 8, 8]" = torch.ops.aten.convolution.default(div_184, arg489_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1104);  div_184 = arg489_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1368: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_1369: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, -1);  unsqueeze_1368 = None
        sub_171: "f32[8, 1104, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_242, unsqueeze_1369);  convolution_242 = unsqueeze_1369 = None
        add_573: "f32[1104]" = torch.ops.aten.add.Tensor(arg491_1, 1e-05);  arg491_1 = None
        sqrt_171: "f32[1104]" = torch.ops.aten.sqrt.default(add_573);  add_573 = None
        reciprocal_171: "f32[1104]" = torch.ops.aten.reciprocal.default(sqrt_171);  sqrt_171 = None
        mul_698: "f32[1104]" = torch.ops.aten.mul.Tensor(reciprocal_171, 1);  reciprocal_171 = None
        unsqueeze_1370: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(mul_698, -1);  mul_698 = None
        unsqueeze_1371: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, -1);  unsqueeze_1370 = None
        mul_699: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_1371);  sub_171 = unsqueeze_1371 = None
        unsqueeze_1372: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg492_1, -1);  arg492_1 = None
        unsqueeze_1373: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, -1);  unsqueeze_1372 = None
        mul_700: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(mul_699, unsqueeze_1373);  mul_699 = unsqueeze_1373 = None
        unsqueeze_1374: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(arg493_1, -1);  arg493_1 = None
        unsqueeze_1375: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, -1);  unsqueeze_1374 = None
        add_574: "f32[8, 1104, 8, 8]" = torch.ops.aten.add.Tensor(mul_700, unsqueeze_1375);  mul_700 = unsqueeze_1375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_575: "f32[8, 1104, 8, 8]" = torch.ops.aten.add.Tensor(add_574, 3)
        clamp_min_185: "f32[8, 1104, 8, 8]" = torch.ops.aten.clamp_min.default(add_575, 0);  add_575 = None
        clamp_max_185: "f32[8, 1104, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_185, 6);  clamp_min_185 = None
        mul_701: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(add_574, clamp_max_185);  add_574 = clamp_max_185 = None
        div_185: "f32[8, 1104, 8, 8]" = torch.ops.aten.div.Tensor(mul_701, 6);  mul_701 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_36: "f32[8, 1104, 1, 1]" = torch.ops.aten.mean.dim(div_185, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_243: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_36, arg494_1, arg495_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_36 = arg494_1 = arg495_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        add_576: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_243, 3)
        clamp_min_186: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_576, 0);  add_576 = None
        clamp_max_186: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_186, 6);  clamp_min_186 = None
        mul_702: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_243, clamp_max_186);  convolution_243 = clamp_max_186 = None
        div_186: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_702, 6);  mul_702 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_244: "f32[8, 1104, 1, 1]" = torch.ops.aten.convolution.default(div_186, arg496_1, arg497_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_186 = arg496_1 = arg497_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_577: "f32[8, 1104, 1, 1]" = torch.ops.aten.add.Tensor(convolution_244, 3);  convolution_244 = None
        clamp_min_187: "f32[8, 1104, 1, 1]" = torch.ops.aten.clamp_min.default(add_577, 0);  add_577 = None
        clamp_max_187: "f32[8, 1104, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_187, 6);  clamp_min_187 = None
        div_187: "f32[8, 1104, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_187, 6);  clamp_max_187 = None
        mul_703: "f32[8, 1104, 8, 8]" = torch.ops.aten.mul.Tensor(div_185, div_187);  div_185 = div_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_245: "f32[8, 224, 8, 8]" = torch.ops.aten.convolution.default(mul_703, arg498_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_703 = arg498_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1376: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_1377: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, -1);  unsqueeze_1376 = None
        sub_172: "f32[8, 224, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_245, unsqueeze_1377);  convolution_245 = unsqueeze_1377 = None
        add_578: "f32[224]" = torch.ops.aten.add.Tensor(arg500_1, 1e-05);  arg500_1 = None
        sqrt_172: "f32[224]" = torch.ops.aten.sqrt.default(add_578);  add_578 = None
        reciprocal_172: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_172);  sqrt_172 = None
        mul_704: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_172, 1);  reciprocal_172 = None
        unsqueeze_1378: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_704, -1);  mul_704 = None
        unsqueeze_1379: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, -1);  unsqueeze_1378 = None
        mul_705: "f32[8, 224, 8, 8]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_1379);  sub_172 = unsqueeze_1379 = None
        unsqueeze_1380: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg501_1, -1);  arg501_1 = None
        unsqueeze_1381: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, -1);  unsqueeze_1380 = None
        mul_706: "f32[8, 224, 8, 8]" = torch.ops.aten.mul.Tensor(mul_705, unsqueeze_1381);  mul_705 = unsqueeze_1381 = None
        unsqueeze_1382: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
        unsqueeze_1383: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, -1);  unsqueeze_1382 = None
        add_579: "f32[8, 224, 8, 8]" = torch.ops.aten.add.Tensor(mul_706, unsqueeze_1383);  mul_706 = unsqueeze_1383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:111 in forward, code: x = self.conv(x)
        convolution_246: "f32[8, 1344, 8, 8]" = torch.ops.aten.convolution.default(add_579, arg503_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_579 = arg503_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_1384: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
        unsqueeze_1385: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, -1);  unsqueeze_1384 = None
        sub_173: "f32[8, 1344, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_246, unsqueeze_1385);  convolution_246 = unsqueeze_1385 = None
        add_580: "f32[1344]" = torch.ops.aten.add.Tensor(arg505_1, 1e-05);  arg505_1 = None
        sqrt_173: "f32[1344]" = torch.ops.aten.sqrt.default(add_580);  add_580 = None
        reciprocal_173: "f32[1344]" = torch.ops.aten.reciprocal.default(sqrt_173);  sqrt_173 = None
        mul_707: "f32[1344]" = torch.ops.aten.mul.Tensor(reciprocal_173, 1);  reciprocal_173 = None
        unsqueeze_1386: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(mul_707, -1);  mul_707 = None
        unsqueeze_1387: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, -1);  unsqueeze_1386 = None
        mul_708: "f32[8, 1344, 8, 8]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_1387);  sub_173 = unsqueeze_1387 = None
        unsqueeze_1388: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg506_1, -1);  arg506_1 = None
        unsqueeze_1389: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, -1);  unsqueeze_1388 = None
        mul_709: "f32[8, 1344, 8, 8]" = torch.ops.aten.mul.Tensor(mul_708, unsqueeze_1389);  mul_708 = unsqueeze_1389 = None
        unsqueeze_1390: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(arg507_1, -1);  arg507_1 = None
        unsqueeze_1391: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, -1);  unsqueeze_1390 = None
        add_581: "f32[8, 1344, 8, 8]" = torch.ops.aten.add.Tensor(mul_709, unsqueeze_1391);  mul_709 = unsqueeze_1391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_582: "f32[8, 1344, 8, 8]" = torch.ops.aten.add.Tensor(add_581, 3)
        clamp_min_188: "f32[8, 1344, 8, 8]" = torch.ops.aten.clamp_min.default(add_582, 0);  add_582 = None
        clamp_max_188: "f32[8, 1344, 8, 8]" = torch.ops.aten.clamp_max.default(clamp_min_188, 6);  clamp_min_188 = None
        mul_710: "f32[8, 1344, 8, 8]" = torch.ops.aten.mul.Tensor(add_581, clamp_max_188);  add_581 = clamp_max_188 = None
        div_188: "f32[8, 1344, 8, 8]" = torch.ops.aten.div.Tensor(mul_710, 6);  mul_710 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_37: "f32[8, 1344, 1, 1]" = torch.ops.aten.mean.dim(div_188, [-1, -2], True);  div_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:261 in forward_head, code: x = self.conv_head(x)
        convolution_247: "f32[8, 1984, 1, 1]" = torch.ops.aten.convolution.default(mean_37, arg508_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_37 = arg508_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:263 in forward_head, code: x = self.act2(x)
        add_583: "f32[8, 1984, 1, 1]" = torch.ops.aten.add.Tensor(convolution_247, 3)
        clamp_min_189: "f32[8, 1984, 1, 1]" = torch.ops.aten.clamp_min.default(add_583, 0);  add_583 = None
        clamp_max_189: "f32[8, 1984, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_189, 6);  clamp_min_189 = None
        mul_711: "f32[8, 1984, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_247, clamp_max_189);  convolution_247 = clamp_max_189 = None
        div_189: "f32[8, 1984, 1, 1]" = torch.ops.aten.div.Tensor(mul_711, 6);  mul_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        view_3: "f32[8, 1984]" = torch.ops.aten.reshape.default(div_189, [8, 1984]);  div_189 = None
        permute_1: "f32[1984, 1000]" = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg510_1, view_3, permute_1);  arg510_1 = view_3 = permute_1 = None
        return (addmm_1,)
        