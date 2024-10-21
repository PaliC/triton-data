class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[16]", arg3_1: "f32[16]", arg4_1: "f32[16]", arg5_1: "f32[16]", arg6_1: "f32[8, 16, 1, 1]", arg7_1: "f32[8]", arg8_1: "f32[8]", arg9_1: "f32[8]", arg10_1: "f32[8]", arg11_1: "f32[8, 1, 3, 3]", arg12_1: "f32[8]", arg13_1: "f32[8]", arg14_1: "f32[8]", arg15_1: "f32[8]", arg16_1: "f32[8, 16, 1, 1]", arg17_1: "f32[8]", arg18_1: "f32[8]", arg19_1: "f32[8]", arg20_1: "f32[8]", arg21_1: "f32[8, 1, 3, 3]", arg22_1: "f32[8]", arg23_1: "f32[8]", arg24_1: "f32[8]", arg25_1: "f32[8]", arg26_1: "f32[24, 16, 1, 1]", arg27_1: "f32[24]", arg28_1: "f32[24]", arg29_1: "f32[24]", arg30_1: "f32[24]", arg31_1: "f32[24, 1, 3, 3]", arg32_1: "f32[24]", arg33_1: "f32[24]", arg34_1: "f32[24]", arg35_1: "f32[24]", arg36_1: "f32[48, 1, 3, 3]", arg37_1: "f32[48]", arg38_1: "f32[48]", arg39_1: "f32[48]", arg40_1: "f32[48]", arg41_1: "f32[12, 48, 1, 1]", arg42_1: "f32[12]", arg43_1: "f32[12]", arg44_1: "f32[12]", arg45_1: "f32[12]", arg46_1: "f32[12, 1, 3, 3]", arg47_1: "f32[12]", arg48_1: "f32[12]", arg49_1: "f32[12]", arg50_1: "f32[12]", arg51_1: "f32[16, 1, 3, 3]", arg52_1: "f32[16]", arg53_1: "f32[16]", arg54_1: "f32[16]", arg55_1: "f32[16]", arg56_1: "f32[24, 16, 1, 1]", arg57_1: "f32[24]", arg58_1: "f32[24]", arg59_1: "f32[24]", arg60_1: "f32[24]", arg61_1: "f32[36, 24, 1, 1]", arg62_1: "f32[36]", arg63_1: "f32[36]", arg64_1: "f32[36]", arg65_1: "f32[36]", arg66_1: "f32[36, 1, 3, 3]", arg67_1: "f32[36]", arg68_1: "f32[36]", arg69_1: "f32[36]", arg70_1: "f32[36]", arg71_1: "f32[12, 72, 1, 1]", arg72_1: "f32[12]", arg73_1: "f32[12]", arg74_1: "f32[12]", arg75_1: "f32[12]", arg76_1: "f32[12, 1, 3, 3]", arg77_1: "f32[12]", arg78_1: "f32[12]", arg79_1: "f32[12]", arg80_1: "f32[12]", arg81_1: "f32[36, 24, 1, 1]", arg82_1: "f32[36]", arg83_1: "f32[36]", arg84_1: "f32[36]", arg85_1: "f32[36]", arg86_1: "f32[36, 1, 3, 3]", arg87_1: "f32[36]", arg88_1: "f32[36]", arg89_1: "f32[36]", arg90_1: "f32[36]", arg91_1: "f32[72, 1, 5, 5]", arg92_1: "f32[72]", arg93_1: "f32[72]", arg94_1: "f32[72]", arg95_1: "f32[72]", arg96_1: "f32[20, 72, 1, 1]", arg97_1: "f32[20]", arg98_1: "f32[72, 20, 1, 1]", arg99_1: "f32[72]", arg100_1: "f32[20, 72, 1, 1]", arg101_1: "f32[20]", arg102_1: "f32[20]", arg103_1: "f32[20]", arg104_1: "f32[20]", arg105_1: "f32[20, 1, 3, 3]", arg106_1: "f32[20]", arg107_1: "f32[20]", arg108_1: "f32[20]", arg109_1: "f32[20]", arg110_1: "f32[24, 1, 5, 5]", arg111_1: "f32[24]", arg112_1: "f32[24]", arg113_1: "f32[24]", arg114_1: "f32[24]", arg115_1: "f32[40, 24, 1, 1]", arg116_1: "f32[40]", arg117_1: "f32[40]", arg118_1: "f32[40]", arg119_1: "f32[40]", arg120_1: "f32[60, 40, 1, 1]", arg121_1: "f32[60]", arg122_1: "f32[60]", arg123_1: "f32[60]", arg124_1: "f32[60]", arg125_1: "f32[60, 1, 3, 3]", arg126_1: "f32[60]", arg127_1: "f32[60]", arg128_1: "f32[60]", arg129_1: "f32[60]", arg130_1: "f32[32, 120, 1, 1]", arg131_1: "f32[32]", arg132_1: "f32[120, 32, 1, 1]", arg133_1: "f32[120]", arg134_1: "f32[20, 120, 1, 1]", arg135_1: "f32[20]", arg136_1: "f32[20]", arg137_1: "f32[20]", arg138_1: "f32[20]", arg139_1: "f32[20, 1, 3, 3]", arg140_1: "f32[20]", arg141_1: "f32[20]", arg142_1: "f32[20]", arg143_1: "f32[20]", arg144_1: "f32[120, 40, 1, 1]", arg145_1: "f32[120]", arg146_1: "f32[120]", arg147_1: "f32[120]", arg148_1: "f32[120]", arg149_1: "f32[120, 1, 3, 3]", arg150_1: "f32[120]", arg151_1: "f32[120]", arg152_1: "f32[120]", arg153_1: "f32[120]", arg154_1: "f32[240, 1, 3, 3]", arg155_1: "f32[240]", arg156_1: "f32[240]", arg157_1: "f32[240]", arg158_1: "f32[240]", arg159_1: "f32[40, 240, 1, 1]", arg160_1: "f32[40]", arg161_1: "f32[40]", arg162_1: "f32[40]", arg163_1: "f32[40]", arg164_1: "f32[40, 1, 3, 3]", arg165_1: "f32[40]", arg166_1: "f32[40]", arg167_1: "f32[40]", arg168_1: "f32[40]", arg169_1: "f32[40, 1, 3, 3]", arg170_1: "f32[40]", arg171_1: "f32[40]", arg172_1: "f32[40]", arg173_1: "f32[40]", arg174_1: "f32[80, 40, 1, 1]", arg175_1: "f32[80]", arg176_1: "f32[80]", arg177_1: "f32[80]", arg178_1: "f32[80]", arg179_1: "f32[100, 80, 1, 1]", arg180_1: "f32[100]", arg181_1: "f32[100]", arg182_1: "f32[100]", arg183_1: "f32[100]", arg184_1: "f32[100, 1, 3, 3]", arg185_1: "f32[100]", arg186_1: "f32[100]", arg187_1: "f32[100]", arg188_1: "f32[100]", arg189_1: "f32[40, 200, 1, 1]", arg190_1: "f32[40]", arg191_1: "f32[40]", arg192_1: "f32[40]", arg193_1: "f32[40]", arg194_1: "f32[40, 1, 3, 3]", arg195_1: "f32[40]", arg196_1: "f32[40]", arg197_1: "f32[40]", arg198_1: "f32[40]", arg199_1: "f32[92, 80, 1, 1]", arg200_1: "f32[92]", arg201_1: "f32[92]", arg202_1: "f32[92]", arg203_1: "f32[92]", arg204_1: "f32[92, 1, 3, 3]", arg205_1: "f32[92]", arg206_1: "f32[92]", arg207_1: "f32[92]", arg208_1: "f32[92]", arg209_1: "f32[40, 184, 1, 1]", arg210_1: "f32[40]", arg211_1: "f32[40]", arg212_1: "f32[40]", arg213_1: "f32[40]", arg214_1: "f32[40, 1, 3, 3]", arg215_1: "f32[40]", arg216_1: "f32[40]", arg217_1: "f32[40]", arg218_1: "f32[40]", arg219_1: "f32[92, 80, 1, 1]", arg220_1: "f32[92]", arg221_1: "f32[92]", arg222_1: "f32[92]", arg223_1: "f32[92]", arg224_1: "f32[92, 1, 3, 3]", arg225_1: "f32[92]", arg226_1: "f32[92]", arg227_1: "f32[92]", arg228_1: "f32[92]", arg229_1: "f32[40, 184, 1, 1]", arg230_1: "f32[40]", arg231_1: "f32[40]", arg232_1: "f32[40]", arg233_1: "f32[40]", arg234_1: "f32[40, 1, 3, 3]", arg235_1: "f32[40]", arg236_1: "f32[40]", arg237_1: "f32[40]", arg238_1: "f32[40]", arg239_1: "f32[240, 80, 1, 1]", arg240_1: "f32[240]", arg241_1: "f32[240]", arg242_1: "f32[240]", arg243_1: "f32[240]", arg244_1: "f32[240, 1, 3, 3]", arg245_1: "f32[240]", arg246_1: "f32[240]", arg247_1: "f32[240]", arg248_1: "f32[240]", arg249_1: "f32[120, 480, 1, 1]", arg250_1: "f32[120]", arg251_1: "f32[480, 120, 1, 1]", arg252_1: "f32[480]", arg253_1: "f32[56, 480, 1, 1]", arg254_1: "f32[56]", arg255_1: "f32[56]", arg256_1: "f32[56]", arg257_1: "f32[56]", arg258_1: "f32[56, 1, 3, 3]", arg259_1: "f32[56]", arg260_1: "f32[56]", arg261_1: "f32[56]", arg262_1: "f32[56]", arg263_1: "f32[80, 1, 3, 3]", arg264_1: "f32[80]", arg265_1: "f32[80]", arg266_1: "f32[80]", arg267_1: "f32[80]", arg268_1: "f32[112, 80, 1, 1]", arg269_1: "f32[112]", arg270_1: "f32[112]", arg271_1: "f32[112]", arg272_1: "f32[112]", arg273_1: "f32[336, 112, 1, 1]", arg274_1: "f32[336]", arg275_1: "f32[336]", arg276_1: "f32[336]", arg277_1: "f32[336]", arg278_1: "f32[336, 1, 3, 3]", arg279_1: "f32[336]", arg280_1: "f32[336]", arg281_1: "f32[336]", arg282_1: "f32[336]", arg283_1: "f32[168, 672, 1, 1]", arg284_1: "f32[168]", arg285_1: "f32[672, 168, 1, 1]", arg286_1: "f32[672]", arg287_1: "f32[56, 672, 1, 1]", arg288_1: "f32[56]", arg289_1: "f32[56]", arg290_1: "f32[56]", arg291_1: "f32[56]", arg292_1: "f32[56, 1, 3, 3]", arg293_1: "f32[56]", arg294_1: "f32[56]", arg295_1: "f32[56]", arg296_1: "f32[56]", arg297_1: "f32[336, 112, 1, 1]", arg298_1: "f32[336]", arg299_1: "f32[336]", arg300_1: "f32[336]", arg301_1: "f32[336]", arg302_1: "f32[336, 1, 3, 3]", arg303_1: "f32[336]", arg304_1: "f32[336]", arg305_1: "f32[336]", arg306_1: "f32[336]", arg307_1: "f32[672, 1, 5, 5]", arg308_1: "f32[672]", arg309_1: "f32[672]", arg310_1: "f32[672]", arg311_1: "f32[672]", arg312_1: "f32[168, 672, 1, 1]", arg313_1: "f32[168]", arg314_1: "f32[672, 168, 1, 1]", arg315_1: "f32[672]", arg316_1: "f32[80, 672, 1, 1]", arg317_1: "f32[80]", arg318_1: "f32[80]", arg319_1: "f32[80]", arg320_1: "f32[80]", arg321_1: "f32[80, 1, 3, 3]", arg322_1: "f32[80]", arg323_1: "f32[80]", arg324_1: "f32[80]", arg325_1: "f32[80]", arg326_1: "f32[112, 1, 5, 5]", arg327_1: "f32[112]", arg328_1: "f32[112]", arg329_1: "f32[112]", arg330_1: "f32[112]", arg331_1: "f32[160, 112, 1, 1]", arg332_1: "f32[160]", arg333_1: "f32[160]", arg334_1: "f32[160]", arg335_1: "f32[160]", arg336_1: "f32[480, 160, 1, 1]", arg337_1: "f32[480]", arg338_1: "f32[480]", arg339_1: "f32[480]", arg340_1: "f32[480]", arg341_1: "f32[480, 1, 3, 3]", arg342_1: "f32[480]", arg343_1: "f32[480]", arg344_1: "f32[480]", arg345_1: "f32[480]", arg346_1: "f32[80, 960, 1, 1]", arg347_1: "f32[80]", arg348_1: "f32[80]", arg349_1: "f32[80]", arg350_1: "f32[80]", arg351_1: "f32[80, 1, 3, 3]", arg352_1: "f32[80]", arg353_1: "f32[80]", arg354_1: "f32[80]", arg355_1: "f32[80]", arg356_1: "f32[480, 160, 1, 1]", arg357_1: "f32[480]", arg358_1: "f32[480]", arg359_1: "f32[480]", arg360_1: "f32[480]", arg361_1: "f32[480, 1, 3, 3]", arg362_1: "f32[480]", arg363_1: "f32[480]", arg364_1: "f32[480]", arg365_1: "f32[480]", arg366_1: "f32[240, 960, 1, 1]", arg367_1: "f32[240]", arg368_1: "f32[960, 240, 1, 1]", arg369_1: "f32[960]", arg370_1: "f32[80, 960, 1, 1]", arg371_1: "f32[80]", arg372_1: "f32[80]", arg373_1: "f32[80]", arg374_1: "f32[80]", arg375_1: "f32[80, 1, 3, 3]", arg376_1: "f32[80]", arg377_1: "f32[80]", arg378_1: "f32[80]", arg379_1: "f32[80]", arg380_1: "f32[480, 160, 1, 1]", arg381_1: "f32[480]", arg382_1: "f32[480]", arg383_1: "f32[480]", arg384_1: "f32[480]", arg385_1: "f32[480, 1, 3, 3]", arg386_1: "f32[480]", arg387_1: "f32[480]", arg388_1: "f32[480]", arg389_1: "f32[480]", arg390_1: "f32[80, 960, 1, 1]", arg391_1: "f32[80]", arg392_1: "f32[80]", arg393_1: "f32[80]", arg394_1: "f32[80]", arg395_1: "f32[80, 1, 3, 3]", arg396_1: "f32[80]", arg397_1: "f32[80]", arg398_1: "f32[80]", arg399_1: "f32[80]", arg400_1: "f32[480, 160, 1, 1]", arg401_1: "f32[480]", arg402_1: "f32[480]", arg403_1: "f32[480]", arg404_1: "f32[480]", arg405_1: "f32[480, 1, 3, 3]", arg406_1: "f32[480]", arg407_1: "f32[480]", arg408_1: "f32[480]", arg409_1: "f32[480]", arg410_1: "f32[240, 960, 1, 1]", arg411_1: "f32[240]", arg412_1: "f32[960, 240, 1, 1]", arg413_1: "f32[960]", arg414_1: "f32[80, 960, 1, 1]", arg415_1: "f32[80]", arg416_1: "f32[80]", arg417_1: "f32[80]", arg418_1: "f32[80]", arg419_1: "f32[80, 1, 3, 3]", arg420_1: "f32[80]", arg421_1: "f32[80]", arg422_1: "f32[80]", arg423_1: "f32[80]", arg424_1: "f32[960, 160, 1, 1]", arg425_1: "f32[960]", arg426_1: "f32[960]", arg427_1: "f32[960]", arg428_1: "f32[960]", arg429_1: "f32[1280, 960, 1, 1]", arg430_1: "f32[1280]", arg431_1: "f32[1000, 1280]", arg432_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:284 in forward_features, code: x = self.conv_stem(x)
        convolution_95: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:285 in forward_features, code: x = self.bn1(x)
        add_183: "f32[16]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_80: "f32[16]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_80: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_247: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_641: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
        unsqueeze_643: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_641);  convolution_95 = unsqueeze_641 = None
        mul_248: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_645: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_249: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_645);  mul_248 = unsqueeze_645 = None
        unsqueeze_646: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_647: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_184: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_249, unsqueeze_647);  mul_249 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:286 in forward_features, code: x = self.act1(x)
        relu_42: "f32[8, 16, 112, 112]" = torch.ops.aten.relu.default(add_184);  add_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_96: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(relu_42, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg6_1 = None
        add_185: "f32[8]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_81: "f32[8]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_81: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_250: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_649: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_651: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_649);  convolution_96 = unsqueeze_649 = None
        mul_251: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_653: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_252: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_653);  mul_251 = unsqueeze_653 = None
        unsqueeze_654: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_655: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_186: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_655);  mul_252 = unsqueeze_655 = None
        relu_43: "f32[8, 8, 112, 112]" = torch.ops.aten.relu.default(add_186);  add_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_97: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(relu_43, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  arg11_1 = None
        add_187: "f32[8]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_82: "f32[8]" = torch.ops.aten.sqrt.default(add_187);  add_187 = None
        reciprocal_82: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_253: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_657: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_659: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_657);  convolution_97 = unsqueeze_657 = None
        mul_254: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_661: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_255: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_661);  mul_254 = unsqueeze_661 = None
        unsqueeze_662: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_663: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_188: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_663);  mul_255 = unsqueeze_663 = None
        relu_44: "f32[8, 8, 112, 112]" = torch.ops.aten.relu.default(add_188);  add_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_32: "f32[8, 16, 112, 112]" = torch.ops.aten.cat.default([relu_43, relu_44], 1);  relu_43 = relu_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_98: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(cat_32, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_32 = arg16_1 = None
        add_189: "f32[8]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_83: "f32[8]" = torch.ops.aten.sqrt.default(add_189);  add_189 = None
        reciprocal_83: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_256: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_665: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_256, -1);  mul_256 = None
        unsqueeze_667: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_665);  convolution_98 = unsqueeze_665 = None
        mul_257: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_669: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_258: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_257, unsqueeze_669);  mul_257 = unsqueeze_669 = None
        unsqueeze_670: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_671: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_190: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_671);  mul_258 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_99: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(add_190, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  arg21_1 = None
        add_191: "f32[8]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_84: "f32[8]" = torch.ops.aten.sqrt.default(add_191);  add_191 = None
        reciprocal_84: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_259: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_673: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_259, -1);  mul_259 = None
        unsqueeze_675: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_673);  convolution_99 = unsqueeze_673 = None
        mul_260: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_677: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_261: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_260, unsqueeze_677);  mul_260 = unsqueeze_677 = None
        unsqueeze_678: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_679: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_192: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_679);  mul_261 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_33: "f32[8, 16, 112, 112]" = torch.ops.aten.cat.default([add_190, add_192], 1);  add_190 = add_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_193: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(cat_33, relu_42);  cat_33 = relu_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_100: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(add_193, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg26_1 = None
        add_194: "f32[24]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_85: "f32[24]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_85: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_262: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_681: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_262, -1);  mul_262 = None
        unsqueeze_683: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_681);  convolution_100 = unsqueeze_681 = None
        mul_263: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_685: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_264: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_685);  mul_263 = unsqueeze_685 = None
        unsqueeze_686: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_687: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_195: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_264, unsqueeze_687);  mul_264 = unsqueeze_687 = None
        relu_45: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_195);  add_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_101: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(relu_45, arg31_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 24);  arg31_1 = None
        add_196: "f32[24]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_86: "f32[24]" = torch.ops.aten.sqrt.default(add_196);  add_196 = None
        reciprocal_86: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_265: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_689: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_265, -1);  mul_265 = None
        unsqueeze_691: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_689);  convolution_101 = unsqueeze_689 = None
        mul_266: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_693: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_267: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_693);  mul_266 = unsqueeze_693 = None
        unsqueeze_694: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_695: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_197: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_267, unsqueeze_695);  mul_267 = unsqueeze_695 = None
        relu_46: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_197);  add_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_34: "f32[8, 48, 112, 112]" = torch.ops.aten.cat.default([relu_45, relu_46], 1);  relu_45 = relu_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:173 in forward, code: x = self.conv_dw(x)
        convolution_102: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(cat_34, arg36_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 48);  cat_34 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:174 in forward, code: x = self.bn_dw(x)
        add_198: "f32[48]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_87: "f32[48]" = torch.ops.aten.sqrt.default(add_198);  add_198 = None
        reciprocal_87: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_268: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_697: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_268, -1);  mul_268 = None
        unsqueeze_699: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_697);  convolution_102 = unsqueeze_697 = None
        mul_269: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_701: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_270: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_701);  mul_269 = unsqueeze_701 = None
        unsqueeze_702: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_703: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_199: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_703);  mul_270 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_103: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_199, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_199 = arg41_1 = None
        add_200: "f32[12]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_88: "f32[12]" = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_88: "f32[12]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_271: "f32[12]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_705: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(mul_271, -1);  mul_271 = None
        unsqueeze_707: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_705);  convolution_103 = unsqueeze_705 = None
        mul_272: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_709: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_273: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_709);  mul_272 = unsqueeze_709 = None
        unsqueeze_710: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_711: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_201: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_273, unsqueeze_711);  mul_273 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_104: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_201, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 12);  arg46_1 = None
        add_202: "f32[12]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_89: "f32[12]" = torch.ops.aten.sqrt.default(add_202);  add_202 = None
        reciprocal_89: "f32[12]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_274: "f32[12]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_713: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(mul_274, -1);  mul_274 = None
        unsqueeze_715: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_713);  convolution_104 = unsqueeze_713 = None
        mul_275: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_717: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_276: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_717);  mul_275 = unsqueeze_717 = None
        unsqueeze_718: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_719: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_203: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_719);  mul_276 = unsqueeze_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_35: "f32[8, 24, 56, 56]" = torch.ops.aten.cat.default([add_201, add_203], 1);  add_201 = add_203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        convolution_105: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(add_193, arg51_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  add_193 = arg51_1 = None
        add_204: "f32[16]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_90: "f32[16]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_90: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_277: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_721: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_723: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_721);  convolution_105 = unsqueeze_721 = None
        mul_278: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_725: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_279: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_725);  mul_278 = unsqueeze_725 = None
        unsqueeze_726: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_727: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_205: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_727);  mul_279 = unsqueeze_727 = None
        convolution_106: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(add_205, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_205 = arg56_1 = None
        add_206: "f32[24]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_91: "f32[24]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_91: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_280: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_729: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_280, -1);  mul_280 = None
        unsqueeze_731: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_729);  convolution_106 = unsqueeze_729 = None
        mul_281: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_733: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_282: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_281, unsqueeze_733);  mul_281 = unsqueeze_733 = None
        unsqueeze_734: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_735: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_207: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_282, unsqueeze_735);  mul_282 = unsqueeze_735 = None
        add_208: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(cat_35, add_207);  cat_35 = add_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_107: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(add_208, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg61_1 = None
        add_209: "f32[36]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_92: "f32[36]" = torch.ops.aten.sqrt.default(add_209);  add_209 = None
        reciprocal_92: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_283: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_737: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_739: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_737);  convolution_107 = unsqueeze_737 = None
        mul_284: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_741: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_285: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_741);  mul_284 = unsqueeze_741 = None
        unsqueeze_742: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_743: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_210: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_285, unsqueeze_743);  mul_285 = unsqueeze_743 = None
        relu_47: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_210);  add_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_108: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(relu_47, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 36);  arg66_1 = None
        add_211: "f32[36]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_93: "f32[36]" = torch.ops.aten.sqrt.default(add_211);  add_211 = None
        reciprocal_93: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_286: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_745: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_286, -1);  mul_286 = None
        unsqueeze_747: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_745);  convolution_108 = unsqueeze_745 = None
        mul_287: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_749: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_288: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_749);  mul_287 = unsqueeze_749 = None
        unsqueeze_750: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_751: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_212: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_288, unsqueeze_751);  mul_288 = unsqueeze_751 = None
        relu_48: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_212);  add_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_36: "f32[8, 72, 56, 56]" = torch.ops.aten.cat.default([relu_47, relu_48], 1);  relu_47 = relu_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_109: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(cat_36, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_36 = arg71_1 = None
        add_213: "f32[12]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_94: "f32[12]" = torch.ops.aten.sqrt.default(add_213);  add_213 = None
        reciprocal_94: "f32[12]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_289: "f32[12]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_753: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(mul_289, -1);  mul_289 = None
        unsqueeze_755: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_753);  convolution_109 = unsqueeze_753 = None
        mul_290: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_757: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_291: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_290, unsqueeze_757);  mul_290 = unsqueeze_757 = None
        unsqueeze_758: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_759: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_214: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_291, unsqueeze_759);  mul_291 = unsqueeze_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_110: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_214, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 12);  arg76_1 = None
        add_215: "f32[12]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_95: "f32[12]" = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_95: "f32[12]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_292: "f32[12]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_761: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_763: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_761);  convolution_110 = unsqueeze_761 = None
        mul_293: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_765: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_294: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_765);  mul_293 = unsqueeze_765 = None
        unsqueeze_766: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_767: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_216: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_767);  mul_294 = unsqueeze_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_37: "f32[8, 24, 56, 56]" = torch.ops.aten.cat.default([add_214, add_216], 1);  add_214 = add_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_217: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(cat_37, add_208);  cat_37 = add_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_111: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(add_217, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg81_1 = None
        add_218: "f32[36]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_96: "f32[36]" = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_96: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_295: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_769: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_771: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_769);  convolution_111 = unsqueeze_769 = None
        mul_296: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_773: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_297: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_773);  mul_296 = unsqueeze_773 = None
        unsqueeze_774: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_775: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_219: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_297, unsqueeze_775);  mul_297 = unsqueeze_775 = None
        relu_49: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_219);  add_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_112: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(relu_49, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 36);  arg86_1 = None
        add_220: "f32[36]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_97: "f32[36]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_97: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_298: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_777: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_298, -1);  mul_298 = None
        unsqueeze_779: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_777);  convolution_112 = unsqueeze_777 = None
        mul_299: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_781: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_300: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_299, unsqueeze_781);  mul_299 = unsqueeze_781 = None
        unsqueeze_782: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_783: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_221: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_783);  mul_300 = unsqueeze_783 = None
        relu_50: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_221);  add_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_38: "f32[8, 72, 56, 56]" = torch.ops.aten.cat.default([relu_49, relu_50], 1);  relu_49 = relu_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:173 in forward, code: x = self.conv_dw(x)
        convolution_113: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(cat_38, arg91_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72);  cat_38 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:174 in forward, code: x = self.bn_dw(x)
        add_222: "f32[72]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_98: "f32[72]" = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_98: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_301: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_785: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_301, -1);  mul_301 = None
        unsqueeze_787: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_785);  convolution_113 = unsqueeze_785 = None
        mul_302: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_789: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_303: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_789);  mul_302 = unsqueeze_789 = None
        unsqueeze_790: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_791: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_223: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_303, unsqueeze_791);  mul_303 = unsqueeze_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "f32[8, 72, 1, 1]" = torch.ops.aten.mean.dim(add_223, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_114: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_8, arg96_1, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg96_1 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_51: "f32[8, 20, 1, 1]" = torch.ops.aten.relu.default(convolution_114);  convolution_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_115: "f32[8, 72, 1, 1]" = torch.ops.aten.convolution.default(relu_51, arg98_1, arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_51 = arg98_1 = arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_224: "f32[8, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_115, 3);  convolution_115 = None
        clamp_min_7: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_224, 0);  add_224 = None
        clamp_max_7: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
        div_7: "f32[8, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_7, 6);  clamp_max_7 = None
        mul_304: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(add_223, div_7);  add_223 = div_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_116: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(mul_304, arg100_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_304 = arg100_1 = None
        add_225: "f32[20]" = torch.ops.aten.add.Tensor(arg102_1, 1e-05);  arg102_1 = None
        sqrt_99: "f32[20]" = torch.ops.aten.sqrt.default(add_225);  add_225 = None
        reciprocal_99: "f32[20]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_305: "f32[20]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_793: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(mul_305, -1);  mul_305 = None
        unsqueeze_795: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_793);  convolution_116 = unsqueeze_793 = None
        mul_306: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_797: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_307: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_306, unsqueeze_797);  mul_306 = unsqueeze_797 = None
        unsqueeze_798: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_799: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_226: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_799);  mul_307 = unsqueeze_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_117: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(add_226, arg105_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 20);  arg105_1 = None
        add_227: "f32[20]" = torch.ops.aten.add.Tensor(arg107_1, 1e-05);  arg107_1 = None
        sqrt_100: "f32[20]" = torch.ops.aten.sqrt.default(add_227);  add_227 = None
        reciprocal_100: "f32[20]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_308: "f32[20]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_801: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_803: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_801);  convolution_117 = unsqueeze_801 = None
        mul_309: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_805: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_310: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_805);  mul_309 = unsqueeze_805 = None
        unsqueeze_806: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_807: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_228: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_807);  mul_310 = unsqueeze_807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_39: "f32[8, 40, 28, 28]" = torch.ops.aten.cat.default([add_226, add_228], 1);  add_226 = add_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        convolution_118: "f32[8, 24, 28, 28]" = torch.ops.aten.convolution.default(add_217, arg110_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 24);  add_217 = arg110_1 = None
        add_229: "f32[24]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_101: "f32[24]" = torch.ops.aten.sqrt.default(add_229);  add_229 = None
        reciprocal_101: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_311: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_809: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_311, -1);  mul_311 = None
        unsqueeze_811: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101: "f32[8, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_809);  convolution_118 = unsqueeze_809 = None
        mul_312: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_813: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_313: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(mul_312, unsqueeze_813);  mul_312 = unsqueeze_813 = None
        unsqueeze_814: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_815: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_230: "f32[8, 24, 28, 28]" = torch.ops.aten.add.Tensor(mul_313, unsqueeze_815);  mul_313 = unsqueeze_815 = None
        convolution_119: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(add_230, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_230 = arg115_1 = None
        add_231: "f32[40]" = torch.ops.aten.add.Tensor(arg117_1, 1e-05);  arg117_1 = None
        sqrt_102: "f32[40]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_102: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_314: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_817: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_314, -1);  mul_314 = None
        unsqueeze_819: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_817);  convolution_119 = unsqueeze_817 = None
        mul_315: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_821: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_316: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_821);  mul_315 = unsqueeze_821 = None
        unsqueeze_822: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_823: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_232: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_316, unsqueeze_823);  mul_316 = unsqueeze_823 = None
        add_233: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(cat_39, add_232);  cat_39 = add_232 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_120: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(add_233, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg120_1 = None
        add_234: "f32[60]" = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_103: "f32[60]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_103: "f32[60]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_317: "f32[60]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_825: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(mul_317, -1);  mul_317 = None
        unsqueeze_827: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_825);  convolution_120 = unsqueeze_825 = None
        mul_318: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_829: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_319: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_829);  mul_318 = unsqueeze_829 = None
        unsqueeze_830: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_831: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_235: "f32[8, 60, 28, 28]" = torch.ops.aten.add.Tensor(mul_319, unsqueeze_831);  mul_319 = unsqueeze_831 = None
        relu_52: "f32[8, 60, 28, 28]" = torch.ops.aten.relu.default(add_235);  add_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_121: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(relu_52, arg125_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 60);  arg125_1 = None
        add_236: "f32[60]" = torch.ops.aten.add.Tensor(arg127_1, 1e-05);  arg127_1 = None
        sqrt_104: "f32[60]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_104: "f32[60]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_320: "f32[60]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_833: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_835: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_833);  convolution_121 = unsqueeze_833 = None
        mul_321: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_837: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_322: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_837);  mul_321 = unsqueeze_837 = None
        unsqueeze_838: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_839: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_237: "f32[8, 60, 28, 28]" = torch.ops.aten.add.Tensor(mul_322, unsqueeze_839);  mul_322 = unsqueeze_839 = None
        relu_53: "f32[8, 60, 28, 28]" = torch.ops.aten.relu.default(add_237);  add_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_40: "f32[8, 120, 28, 28]" = torch.ops.aten.cat.default([relu_52, relu_53], 1);  relu_52 = relu_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(cat_40, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_122: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_9, arg130_1, arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg130_1 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_54: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_122);  convolution_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_123: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_54, arg132_1, arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_54 = arg132_1 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_238: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_123, 3);  convolution_123 = None
        clamp_min_8: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_238, 0);  add_238 = None
        clamp_max_8: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
        div_8: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_8, 6);  clamp_max_8 = None
        mul_323: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(cat_40, div_8);  cat_40 = div_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_124: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(mul_323, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_323 = arg134_1 = None
        add_239: "f32[20]" = torch.ops.aten.add.Tensor(arg136_1, 1e-05);  arg136_1 = None
        sqrt_105: "f32[20]" = torch.ops.aten.sqrt.default(add_239);  add_239 = None
        reciprocal_105: "f32[20]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_324: "f32[20]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_841: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
        unsqueeze_843: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_841);  convolution_124 = unsqueeze_841 = None
        mul_325: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_845: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_326: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_845);  mul_325 = unsqueeze_845 = None
        unsqueeze_846: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_847: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_240: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_847);  mul_326 = unsqueeze_847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_125: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(add_240, arg139_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 20);  arg139_1 = None
        add_241: "f32[20]" = torch.ops.aten.add.Tensor(arg141_1, 1e-05);  arg141_1 = None
        sqrt_106: "f32[20]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_106: "f32[20]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_327: "f32[20]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_849: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_851: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_849);  convolution_125 = unsqueeze_849 = None
        mul_328: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_853: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_329: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_853);  mul_328 = unsqueeze_853 = None
        unsqueeze_854: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_855: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_242: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_855);  mul_329 = unsqueeze_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_41: "f32[8, 40, 28, 28]" = torch.ops.aten.cat.default([add_240, add_242], 1);  add_240 = add_242 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_243: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(cat_41, add_233);  cat_41 = add_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_126: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_243, arg144_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg144_1 = None
        add_244: "f32[120]" = torch.ops.aten.add.Tensor(arg146_1, 1e-05);  arg146_1 = None
        sqrt_107: "f32[120]" = torch.ops.aten.sqrt.default(add_244);  add_244 = None
        reciprocal_107: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_330: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_857: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_859: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_857);  convolution_126 = unsqueeze_857 = None
        mul_331: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_861: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_332: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_861);  mul_331 = unsqueeze_861 = None
        unsqueeze_862: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_863: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_245: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_863);  mul_332 = unsqueeze_863 = None
        relu_55: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_245);  add_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_127: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_55, arg149_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  arg149_1 = None
        add_246: "f32[120]" = torch.ops.aten.add.Tensor(arg151_1, 1e-05);  arg151_1 = None
        sqrt_108: "f32[120]" = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_108: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_333: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_865: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_867: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_865);  convolution_127 = unsqueeze_865 = None
        mul_334: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_869: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_335: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_869);  mul_334 = unsqueeze_869 = None
        unsqueeze_870: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_871: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_247: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_871);  mul_335 = unsqueeze_871 = None
        relu_56: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_247);  add_247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_42: "f32[8, 240, 28, 28]" = torch.ops.aten.cat.default([relu_55, relu_56], 1);  relu_55 = relu_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:173 in forward, code: x = self.conv_dw(x)
        convolution_128: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(cat_42, arg154_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  cat_42 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:174 in forward, code: x = self.bn_dw(x)
        add_248: "f32[240]" = torch.ops.aten.add.Tensor(arg156_1, 1e-05);  arg156_1 = None
        sqrt_109: "f32[240]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_109: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_336: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_873: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_875: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_873);  convolution_128 = unsqueeze_873 = None
        mul_337: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_877: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_338: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_877);  mul_337 = unsqueeze_877 = None
        unsqueeze_878: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_879: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_249: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_879);  mul_338 = unsqueeze_879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_129: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_249, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_249 = arg159_1 = None
        add_250: "f32[40]" = torch.ops.aten.add.Tensor(arg161_1, 1e-05);  arg161_1 = None
        sqrt_110: "f32[40]" = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_110: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_339: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_881: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_883: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_881);  convolution_129 = unsqueeze_881 = None
        mul_340: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_885: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_341: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_885);  mul_340 = unsqueeze_885 = None
        unsqueeze_886: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_887: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_251: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_887);  mul_341 = unsqueeze_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_130: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_251, arg164_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg164_1 = None
        add_252: "f32[40]" = torch.ops.aten.add.Tensor(arg166_1, 1e-05);  arg166_1 = None
        sqrt_111: "f32[40]" = torch.ops.aten.sqrt.default(add_252);  add_252 = None
        reciprocal_111: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_342: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_889: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_891: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_889);  convolution_130 = unsqueeze_889 = None
        mul_343: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_893: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_344: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_893);  mul_343 = unsqueeze_893 = None
        unsqueeze_894: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_895: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_253: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_895);  mul_344 = unsqueeze_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_43: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_251, add_253], 1);  add_251 = add_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        convolution_131: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_243, arg169_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 40);  add_243 = arg169_1 = None
        add_254: "f32[40]" = torch.ops.aten.add.Tensor(arg171_1, 1e-05);  arg171_1 = None
        sqrt_112: "f32[40]" = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_112: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_345: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_897: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
        unsqueeze_899: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_897);  convolution_131 = unsqueeze_897 = None
        mul_346: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_901: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_347: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_901);  mul_346 = unsqueeze_901 = None
        unsqueeze_902: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_903: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_255: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_347, unsqueeze_903);  mul_347 = unsqueeze_903 = None
        convolution_132: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(add_255, arg174_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_255 = arg174_1 = None
        add_256: "f32[80]" = torch.ops.aten.add.Tensor(arg176_1, 1e-05);  arg176_1 = None
        sqrt_113: "f32[80]" = torch.ops.aten.sqrt.default(add_256);  add_256 = None
        reciprocal_113: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_348: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_905: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_907: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_905);  convolution_132 = unsqueeze_905 = None
        mul_349: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_909: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_350: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_909);  mul_349 = unsqueeze_909 = None
        unsqueeze_910: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_911: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_257: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_911);  mul_350 = unsqueeze_911 = None
        add_258: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_43, add_257);  cat_43 = add_257 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_133: "f32[8, 100, 14, 14]" = torch.ops.aten.convolution.default(add_258, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg179_1 = None
        add_259: "f32[100]" = torch.ops.aten.add.Tensor(arg181_1, 1e-05);  arg181_1 = None
        sqrt_114: "f32[100]" = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_114: "f32[100]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_351: "f32[100]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_913: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_915: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_913);  convolution_133 = unsqueeze_913 = None
        mul_352: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_917: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_353: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_917);  mul_352 = unsqueeze_917 = None
        unsqueeze_918: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_919: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_260: "f32[8, 100, 14, 14]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_919);  mul_353 = unsqueeze_919 = None
        relu_57: "f32[8, 100, 14, 14]" = torch.ops.aten.relu.default(add_260);  add_260 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_134: "f32[8, 100, 14, 14]" = torch.ops.aten.convolution.default(relu_57, arg184_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 100);  arg184_1 = None
        add_261: "f32[100]" = torch.ops.aten.add.Tensor(arg186_1, 1e-05);  arg186_1 = None
        sqrt_115: "f32[100]" = torch.ops.aten.sqrt.default(add_261);  add_261 = None
        reciprocal_115: "f32[100]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_354: "f32[100]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_921: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
        unsqueeze_923: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_921);  convolution_134 = unsqueeze_921 = None
        mul_355: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_925: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_356: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_925);  mul_355 = unsqueeze_925 = None
        unsqueeze_926: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_927: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_262: "f32[8, 100, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_927);  mul_356 = unsqueeze_927 = None
        relu_58: "f32[8, 100, 14, 14]" = torch.ops.aten.relu.default(add_262);  add_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_44: "f32[8, 200, 14, 14]" = torch.ops.aten.cat.default([relu_57, relu_58], 1);  relu_57 = relu_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_135: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(cat_44, arg189_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_44 = arg189_1 = None
        add_263: "f32[40]" = torch.ops.aten.add.Tensor(arg191_1, 1e-05);  arg191_1 = None
        sqrt_116: "f32[40]" = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_116: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_357: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_929: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_931: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_929);  convolution_135 = unsqueeze_929 = None
        mul_358: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_933: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_359: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_933);  mul_358 = unsqueeze_933 = None
        unsqueeze_934: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_935: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_264: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_935);  mul_359 = unsqueeze_935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_136: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_264, arg194_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg194_1 = None
        add_265: "f32[40]" = torch.ops.aten.add.Tensor(arg196_1, 1e-05);  arg196_1 = None
        sqrt_117: "f32[40]" = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_117: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_360: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_937: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_939: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_937);  convolution_136 = unsqueeze_937 = None
        mul_361: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_941: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_362: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_941);  mul_361 = unsqueeze_941 = None
        unsqueeze_942: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_943: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_266: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_943);  mul_362 = unsqueeze_943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_45: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_264, add_266], 1);  add_264 = add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_267: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_45, add_258);  cat_45 = add_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_137: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(add_267, arg199_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg199_1 = None
        add_268: "f32[92]" = torch.ops.aten.add.Tensor(arg201_1, 1e-05);  arg201_1 = None
        sqrt_118: "f32[92]" = torch.ops.aten.sqrt.default(add_268);  add_268 = None
        reciprocal_118: "f32[92]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_363: "f32[92]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_945: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
        unsqueeze_947: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_945);  convolution_137 = unsqueeze_945 = None
        mul_364: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_949: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_365: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_949);  mul_364 = unsqueeze_949 = None
        unsqueeze_950: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_951: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_269: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_365, unsqueeze_951);  mul_365 = unsqueeze_951 = None
        relu_59: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_269);  add_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_138: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(relu_59, arg204_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 92);  arg204_1 = None
        add_270: "f32[92]" = torch.ops.aten.add.Tensor(arg206_1, 1e-05);  arg206_1 = None
        sqrt_119: "f32[92]" = torch.ops.aten.sqrt.default(add_270);  add_270 = None
        reciprocal_119: "f32[92]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_366: "f32[92]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_953: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_955: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_953);  convolution_138 = unsqueeze_953 = None
        mul_367: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_957: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_368: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_957);  mul_367 = unsqueeze_957 = None
        unsqueeze_958: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_959: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_271: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_368, unsqueeze_959);  mul_368 = unsqueeze_959 = None
        relu_60: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_271);  add_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_46: "f32[8, 184, 14, 14]" = torch.ops.aten.cat.default([relu_59, relu_60], 1);  relu_59 = relu_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_139: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(cat_46, arg209_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_46 = arg209_1 = None
        add_272: "f32[40]" = torch.ops.aten.add.Tensor(arg211_1, 1e-05);  arg211_1 = None
        sqrt_120: "f32[40]" = torch.ops.aten.sqrt.default(add_272);  add_272 = None
        reciprocal_120: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_369: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_961: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_963: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_961);  convolution_139 = unsqueeze_961 = None
        mul_370: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_965: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_371: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_965);  mul_370 = unsqueeze_965 = None
        unsqueeze_966: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_967: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_273: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_967);  mul_371 = unsqueeze_967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_140: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_273, arg214_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg214_1 = None
        add_274: "f32[40]" = torch.ops.aten.add.Tensor(arg216_1, 1e-05);  arg216_1 = None
        sqrt_121: "f32[40]" = torch.ops.aten.sqrt.default(add_274);  add_274 = None
        reciprocal_121: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_372: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_969: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_971: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_969);  convolution_140 = unsqueeze_969 = None
        mul_373: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_973: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_374: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_973);  mul_373 = unsqueeze_973 = None
        unsqueeze_974: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_975: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_275: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_975);  mul_374 = unsqueeze_975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_47: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_273, add_275], 1);  add_273 = add_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_276: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_47, add_267);  cat_47 = add_267 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_141: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(add_276, arg219_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg219_1 = None
        add_277: "f32[92]" = torch.ops.aten.add.Tensor(arg221_1, 1e-05);  arg221_1 = None
        sqrt_122: "f32[92]" = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_122: "f32[92]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_375: "f32[92]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_977: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
        unsqueeze_979: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_977);  convolution_141 = unsqueeze_977 = None
        mul_376: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_981: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_377: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_981);  mul_376 = unsqueeze_981 = None
        unsqueeze_982: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_983: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_278: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_983);  mul_377 = unsqueeze_983 = None
        relu_61: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_278);  add_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_142: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(relu_61, arg224_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 92);  arg224_1 = None
        add_279: "f32[92]" = torch.ops.aten.add.Tensor(arg226_1, 1e-05);  arg226_1 = None
        sqrt_123: "f32[92]" = torch.ops.aten.sqrt.default(add_279);  add_279 = None
        reciprocal_123: "f32[92]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_378: "f32[92]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_985: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_987: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_985);  convolution_142 = unsqueeze_985 = None
        mul_379: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_989: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_380: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_989);  mul_379 = unsqueeze_989 = None
        unsqueeze_990: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_991: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_280: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_991);  mul_380 = unsqueeze_991 = None
        relu_62: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_280);  add_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_48: "f32[8, 184, 14, 14]" = torch.ops.aten.cat.default([relu_61, relu_62], 1);  relu_61 = relu_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_143: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(cat_48, arg229_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_48 = arg229_1 = None
        add_281: "f32[40]" = torch.ops.aten.add.Tensor(arg231_1, 1e-05);  arg231_1 = None
        sqrt_124: "f32[40]" = torch.ops.aten.sqrt.default(add_281);  add_281 = None
        reciprocal_124: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_381: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_992: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_993: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        unsqueeze_994: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_995: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        sub_124: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_993);  convolution_143 = unsqueeze_993 = None
        mul_382: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_997: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_383: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_997);  mul_382 = unsqueeze_997 = None
        unsqueeze_998: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_999: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_282: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_383, unsqueeze_999);  mul_383 = unsqueeze_999 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_144: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_282, arg234_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40);  arg234_1 = None
        add_283: "f32[40]" = torch.ops.aten.add.Tensor(arg236_1, 1e-05);  arg236_1 = None
        sqrt_125: "f32[40]" = torch.ops.aten.sqrt.default(add_283);  add_283 = None
        reciprocal_125: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_384: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1000: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_1001: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        unsqueeze_1002: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
        unsqueeze_1003: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        sub_125: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_1001);  convolution_144 = unsqueeze_1001 = None
        mul_385: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_1005: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_386: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_1005);  mul_385 = unsqueeze_1005 = None
        unsqueeze_1006: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_1007: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_284: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_386, unsqueeze_1007);  mul_386 = unsqueeze_1007 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_49: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_282, add_284], 1);  add_282 = add_284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_285: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_49, add_276);  cat_49 = add_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_145: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(add_285, arg239_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg239_1 = None
        add_286: "f32[240]" = torch.ops.aten.add.Tensor(arg241_1, 1e-05);  arg241_1 = None
        sqrt_126: "f32[240]" = torch.ops.aten.sqrt.default(add_286);  add_286 = None
        reciprocal_126: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_387: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1008: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_1009: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        unsqueeze_1010: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
        unsqueeze_1011: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        sub_126: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_1009);  convolution_145 = unsqueeze_1009 = None
        mul_388: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_1013: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_389: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_1013);  mul_388 = unsqueeze_1013 = None
        unsqueeze_1014: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_1015: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_287: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_1015);  mul_389 = unsqueeze_1015 = None
        relu_63: "f32[8, 240, 14, 14]" = torch.ops.aten.relu.default(add_287);  add_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_146: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(relu_63, arg244_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 240);  arg244_1 = None
        add_288: "f32[240]" = torch.ops.aten.add.Tensor(arg246_1, 1e-05);  arg246_1 = None
        sqrt_127: "f32[240]" = torch.ops.aten.sqrt.default(add_288);  add_288 = None
        reciprocal_127: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_390: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1016: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_1017: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        unsqueeze_1018: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
        unsqueeze_1019: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        sub_127: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_1017);  convolution_146 = unsqueeze_1017 = None
        mul_391: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_1021: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_392: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_1021);  mul_391 = unsqueeze_1021 = None
        unsqueeze_1022: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_1023: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_289: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_392, unsqueeze_1023);  mul_392 = unsqueeze_1023 = None
        relu_64: "f32[8, 240, 14, 14]" = torch.ops.aten.relu.default(add_289);  add_289 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_50: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([relu_63, relu_64], 1);  relu_63 = relu_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(cat_50, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_147: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(mean_10, arg249_1, arg250_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg249_1 = arg250_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_65: "f32[8, 120, 1, 1]" = torch.ops.aten.relu.default(convolution_147);  convolution_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_148: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(relu_65, arg251_1, arg252_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_65 = arg251_1 = arg252_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_290: "f32[8, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_148, 3);  convolution_148 = None
        clamp_min_9: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_290, 0);  add_290 = None
        clamp_max_9: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
        div_9: "f32[8, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_9, 6);  clamp_max_9 = None
        mul_393: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_50, div_9);  cat_50 = div_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_149: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(mul_393, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_393 = arg253_1 = None
        add_291: "f32[56]" = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_128: "f32[56]" = torch.ops.aten.sqrt.default(add_291);  add_291 = None
        reciprocal_128: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_394: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1024: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_1025: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        unsqueeze_1026: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_394, -1);  mul_394 = None
        unsqueeze_1027: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        sub_128: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_1025);  convolution_149 = unsqueeze_1025 = None
        mul_395: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_1029: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_396: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_395, unsqueeze_1029);  mul_395 = unsqueeze_1029 = None
        unsqueeze_1030: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_1031: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_292: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_396, unsqueeze_1031);  mul_396 = unsqueeze_1031 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_150: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_292, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 56);  arg258_1 = None
        add_293: "f32[56]" = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
        sqrt_129: "f32[56]" = torch.ops.aten.sqrt.default(add_293);  add_293 = None
        reciprocal_129: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_397: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1032: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_1033: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        unsqueeze_1034: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_397, -1);  mul_397 = None
        unsqueeze_1035: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        sub_129: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_1033);  convolution_150 = unsqueeze_1033 = None
        mul_398: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_1037: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_399: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_398, unsqueeze_1037);  mul_398 = unsqueeze_1037 = None
        unsqueeze_1038: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_1039: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_294: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_399, unsqueeze_1039);  mul_399 = unsqueeze_1039 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_51: "f32[8, 112, 14, 14]" = torch.ops.aten.cat.default([add_292, add_294], 1);  add_292 = add_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        convolution_151: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(add_285, arg263_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  add_285 = arg263_1 = None
        add_295: "f32[80]" = torch.ops.aten.add.Tensor(arg265_1, 1e-05);  arg265_1 = None
        sqrt_130: "f32[80]" = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_130: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_400: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1040: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_1041: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        unsqueeze_1042: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_400, -1);  mul_400 = None
        unsqueeze_1043: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        sub_130: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_1041);  convolution_151 = unsqueeze_1041 = None
        mul_401: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_1045: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_402: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_401, unsqueeze_1045);  mul_401 = unsqueeze_1045 = None
        unsqueeze_1046: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_1047: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_296: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_402, unsqueeze_1047);  mul_402 = unsqueeze_1047 = None
        convolution_152: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(add_296, arg268_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_296 = arg268_1 = None
        add_297: "f32[112]" = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
        sqrt_131: "f32[112]" = torch.ops.aten.sqrt.default(add_297);  add_297 = None
        reciprocal_131: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_403: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1048: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_1049: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        unsqueeze_1050: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_403, -1);  mul_403 = None
        unsqueeze_1051: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        sub_131: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_1049);  convolution_152 = unsqueeze_1049 = None
        mul_404: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_1053: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_405: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_404, unsqueeze_1053);  mul_404 = unsqueeze_1053 = None
        unsqueeze_1054: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_1055: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_298: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_1055);  mul_405 = unsqueeze_1055 = None
        add_299: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(cat_51, add_298);  cat_51 = add_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_153: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(add_299, arg273_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg273_1 = None
        add_300: "f32[336]" = torch.ops.aten.add.Tensor(arg275_1, 1e-05);  arg275_1 = None
        sqrt_132: "f32[336]" = torch.ops.aten.sqrt.default(add_300);  add_300 = None
        reciprocal_132: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_406: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1056: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_1057: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        unsqueeze_1058: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_406, -1);  mul_406 = None
        unsqueeze_1059: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        sub_132: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_153, unsqueeze_1057);  convolution_153 = unsqueeze_1057 = None
        mul_407: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_1061: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_408: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_407, unsqueeze_1061);  mul_407 = unsqueeze_1061 = None
        unsqueeze_1062: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_1063: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_301: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_408, unsqueeze_1063);  mul_408 = unsqueeze_1063 = None
        relu_66: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_301);  add_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_154: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(relu_66, arg278_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 336);  arg278_1 = None
        add_302: "f32[336]" = torch.ops.aten.add.Tensor(arg280_1, 1e-05);  arg280_1 = None
        sqrt_133: "f32[336]" = torch.ops.aten.sqrt.default(add_302);  add_302 = None
        reciprocal_133: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_409: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1064: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_1065: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        unsqueeze_1066: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_409, -1);  mul_409 = None
        unsqueeze_1067: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        sub_133: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_1065);  convolution_154 = unsqueeze_1065 = None
        mul_410: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
        unsqueeze_1069: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_411: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_410, unsqueeze_1069);  mul_410 = unsqueeze_1069 = None
        unsqueeze_1070: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_1071: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_303: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_411, unsqueeze_1071);  mul_411 = unsqueeze_1071 = None
        relu_67: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_303);  add_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_52: "f32[8, 672, 14, 14]" = torch.ops.aten.cat.default([relu_66, relu_67], 1);  relu_66 = relu_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(cat_52, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_155: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_11, arg283_1, arg284_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg283_1 = arg284_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_68: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_155);  convolution_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_156: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_68, arg285_1, arg286_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_68 = arg285_1 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_304: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_156, 3);  convolution_156 = None
        clamp_min_10: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_304, 0);  add_304 = None
        clamp_max_10: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
        div_10: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_10, 6);  clamp_max_10 = None
        mul_412: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(cat_52, div_10);  cat_52 = div_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_157: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(mul_412, arg287_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_412 = arg287_1 = None
        add_305: "f32[56]" = torch.ops.aten.add.Tensor(arg289_1, 1e-05);  arg289_1 = None
        sqrt_134: "f32[56]" = torch.ops.aten.sqrt.default(add_305);  add_305 = None
        reciprocal_134: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
        mul_413: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
        unsqueeze_1072: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_1073: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
        unsqueeze_1074: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_413, -1);  mul_413 = None
        unsqueeze_1075: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
        sub_134: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_1073);  convolution_157 = unsqueeze_1073 = None
        mul_414: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
        unsqueeze_1076: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_1077: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
        mul_415: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_414, unsqueeze_1077);  mul_414 = unsqueeze_1077 = None
        unsqueeze_1078: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_1079: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
        add_306: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_415, unsqueeze_1079);  mul_415 = unsqueeze_1079 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_158: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_306, arg292_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 56);  arg292_1 = None
        add_307: "f32[56]" = torch.ops.aten.add.Tensor(arg294_1, 1e-05);  arg294_1 = None
        sqrt_135: "f32[56]" = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_135: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
        mul_416: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
        unsqueeze_1080: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg293_1, -1);  arg293_1 = None
        unsqueeze_1081: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
        unsqueeze_1082: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_416, -1);  mul_416 = None
        unsqueeze_1083: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
        sub_135: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_1081);  convolution_158 = unsqueeze_1081 = None
        mul_417: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
        unsqueeze_1084: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_1085: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
        mul_418: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_417, unsqueeze_1085);  mul_417 = unsqueeze_1085 = None
        unsqueeze_1086: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_1087: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
        add_308: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_418, unsqueeze_1087);  mul_418 = unsqueeze_1087 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_53: "f32[8, 112, 14, 14]" = torch.ops.aten.cat.default([add_306, add_308], 1);  add_306 = add_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_309: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(cat_53, add_299);  cat_53 = add_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_159: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(add_309, arg297_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg297_1 = None
        add_310: "f32[336]" = torch.ops.aten.add.Tensor(arg299_1, 1e-05);  arg299_1 = None
        sqrt_136: "f32[336]" = torch.ops.aten.sqrt.default(add_310);  add_310 = None
        reciprocal_136: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
        mul_419: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
        unsqueeze_1088: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_1089: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
        unsqueeze_1090: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_419, -1);  mul_419 = None
        unsqueeze_1091: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
        sub_136: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_159, unsqueeze_1089);  convolution_159 = unsqueeze_1089 = None
        mul_420: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
        unsqueeze_1092: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_1093: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
        mul_421: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_1093);  mul_420 = unsqueeze_1093 = None
        unsqueeze_1094: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
        unsqueeze_1095: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
        add_311: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_421, unsqueeze_1095);  mul_421 = unsqueeze_1095 = None
        relu_69: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_311);  add_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_160: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(relu_69, arg302_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 336);  arg302_1 = None
        add_312: "f32[336]" = torch.ops.aten.add.Tensor(arg304_1, 1e-05);  arg304_1 = None
        sqrt_137: "f32[336]" = torch.ops.aten.sqrt.default(add_312);  add_312 = None
        reciprocal_137: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
        mul_422: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
        unsqueeze_1096: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_1097: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
        unsqueeze_1098: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_422, -1);  mul_422 = None
        unsqueeze_1099: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
        sub_137: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_1097);  convolution_160 = unsqueeze_1097 = None
        mul_423: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
        unsqueeze_1100: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_1101: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
        mul_424: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_423, unsqueeze_1101);  mul_423 = unsqueeze_1101 = None
        unsqueeze_1102: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_1103: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
        add_313: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_424, unsqueeze_1103);  mul_424 = unsqueeze_1103 = None
        relu_70: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_313);  add_313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_54: "f32[8, 672, 14, 14]" = torch.ops.aten.cat.default([relu_69, relu_70], 1);  relu_69 = relu_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:173 in forward, code: x = self.conv_dw(x)
        convolution_161: "f32[8, 672, 7, 7]" = torch.ops.aten.convolution.default(cat_54, arg307_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672);  cat_54 = arg307_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:174 in forward, code: x = self.bn_dw(x)
        add_314: "f32[672]" = torch.ops.aten.add.Tensor(arg309_1, 1e-05);  arg309_1 = None
        sqrt_138: "f32[672]" = torch.ops.aten.sqrt.default(add_314);  add_314 = None
        reciprocal_138: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
        mul_425: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
        unsqueeze_1104: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_1105: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
        unsqueeze_1106: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_425, -1);  mul_425 = None
        unsqueeze_1107: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
        sub_138: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_1105);  convolution_161 = unsqueeze_1105 = None
        mul_426: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
        unsqueeze_1108: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1109: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
        mul_427: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_1109);  mul_426 = unsqueeze_1109 = None
        unsqueeze_1110: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_1111: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
        add_315: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_427, unsqueeze_1111);  mul_427 = unsqueeze_1111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_12: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(add_315, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_162: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_12, arg312_1, arg313_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_12 = arg312_1 = arg313_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_71: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_162);  convolution_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_163: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_71, arg314_1, arg315_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_71 = arg314_1 = arg315_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_316: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_163, 3);  convolution_163 = None
        clamp_min_11: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_316, 0);  add_316 = None
        clamp_max_11: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
        div_11: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_11, 6);  clamp_max_11 = None
        mul_428: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_315, div_11);  add_315 = div_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_164: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_428, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_428 = arg316_1 = None
        add_317: "f32[80]" = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_139: "f32[80]" = torch.ops.aten.sqrt.default(add_317);  add_317 = None
        reciprocal_139: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_429: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1112: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1113: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        unsqueeze_1114: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
        unsqueeze_1115: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        sub_139: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_1113);  convolution_164 = unsqueeze_1113 = None
        mul_430: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
        unsqueeze_1116: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1117: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_431: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_1117);  mul_430 = unsqueeze_1117 = None
        unsqueeze_1118: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_1119: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_318: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_431, unsqueeze_1119);  mul_431 = unsqueeze_1119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_165: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_318, arg321_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg321_1 = None
        add_319: "f32[80]" = torch.ops.aten.add.Tensor(arg323_1, 1e-05);  arg323_1 = None
        sqrt_140: "f32[80]" = torch.ops.aten.sqrt.default(add_319);  add_319 = None
        reciprocal_140: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_432: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1120: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1121: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        unsqueeze_1122: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
        unsqueeze_1123: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        sub_140: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_1121);  convolution_165 = unsqueeze_1121 = None
        mul_433: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
        unsqueeze_1124: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_1125: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_434: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_1125);  mul_433 = unsqueeze_1125 = None
        unsqueeze_1126: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_1127: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_320: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_434, unsqueeze_1127);  mul_434 = unsqueeze_1127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_55: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_318, add_320], 1);  add_318 = add_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        convolution_166: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(add_309, arg326_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 112);  add_309 = arg326_1 = None
        add_321: "f32[112]" = torch.ops.aten.add.Tensor(arg328_1, 1e-05);  arg328_1 = None
        sqrt_141: "f32[112]" = torch.ops.aten.sqrt.default(add_321);  add_321 = None
        reciprocal_141: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_435: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1128: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_1129: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        unsqueeze_1130: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
        unsqueeze_1131: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        sub_141: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_1129);  convolution_166 = unsqueeze_1129 = None
        mul_436: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
        unsqueeze_1132: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1133: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_437: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_1133);  mul_436 = unsqueeze_1133 = None
        unsqueeze_1134: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_1135: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_322: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_1135);  mul_437 = unsqueeze_1135 = None
        convolution_167: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(add_322, arg331_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_322 = arg331_1 = None
        add_323: "f32[160]" = torch.ops.aten.add.Tensor(arg333_1, 1e-05);  arg333_1 = None
        sqrt_142: "f32[160]" = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_142: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_438: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1136: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_1137: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        unsqueeze_1138: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
        unsqueeze_1139: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        sub_142: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_1137);  convolution_167 = unsqueeze_1137 = None
        mul_439: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
        unsqueeze_1140: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1141: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_440: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_1141);  mul_439 = unsqueeze_1141 = None
        unsqueeze_1142: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_1143: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_324: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_1143);  mul_440 = unsqueeze_1143 = None
        add_325: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_55, add_324);  cat_55 = add_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_168: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(add_325, arg336_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg336_1 = None
        add_326: "f32[480]" = torch.ops.aten.add.Tensor(arg338_1, 1e-05);  arg338_1 = None
        sqrt_143: "f32[480]" = torch.ops.aten.sqrt.default(add_326);  add_326 = None
        reciprocal_143: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_441: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1144: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_1145: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        unsqueeze_1146: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
        unsqueeze_1147: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        sub_143: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_1145);  convolution_168 = unsqueeze_1145 = None
        mul_442: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
        unsqueeze_1148: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_1149: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_443: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_1149);  mul_442 = unsqueeze_1149 = None
        unsqueeze_1150: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_1151: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_327: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_443, unsqueeze_1151);  mul_443 = unsqueeze_1151 = None
        relu_72: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_327);  add_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_169: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_72, arg341_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg341_1 = None
        add_328: "f32[480]" = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_144: "f32[480]" = torch.ops.aten.sqrt.default(add_328);  add_328 = None
        reciprocal_144: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_444: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1152: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_1153: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        unsqueeze_1154: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_1155: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        sub_144: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_169, unsqueeze_1153);  convolution_169 = unsqueeze_1153 = None
        mul_445: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
        unsqueeze_1156: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_1157: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_446: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_1157);  mul_445 = unsqueeze_1157 = None
        unsqueeze_1158: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_1159: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_329: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_1159);  mul_446 = unsqueeze_1159 = None
        relu_73: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_329);  add_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_56: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_72, relu_73], 1);  relu_72 = relu_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_170: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(cat_56, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_56 = arg346_1 = None
        add_330: "f32[80]" = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_145: "f32[80]" = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_145: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_447: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1160: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_1161: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        unsqueeze_1162: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_1163: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        sub_145: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_1161);  convolution_170 = unsqueeze_1161 = None
        mul_448: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
        unsqueeze_1164: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_1165: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_449: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_1165);  mul_448 = unsqueeze_1165 = None
        unsqueeze_1166: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_1167: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_331: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_449, unsqueeze_1167);  mul_449 = unsqueeze_1167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_171: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_331, arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg351_1 = None
        add_332: "f32[80]" = torch.ops.aten.add.Tensor(arg353_1, 1e-05);  arg353_1 = None
        sqrt_146: "f32[80]" = torch.ops.aten.sqrt.default(add_332);  add_332 = None
        reciprocal_146: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_450: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1168: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_1169: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        unsqueeze_1170: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1171: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        sub_146: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_1169);  convolution_171 = unsqueeze_1169 = None
        mul_451: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
        unsqueeze_1172: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_1173: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_452: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1173);  mul_451 = unsqueeze_1173 = None
        unsqueeze_1174: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_1175: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_333: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1175);  mul_452 = unsqueeze_1175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_57: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_331, add_333], 1);  add_331 = add_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_334: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_57, add_325);  cat_57 = add_325 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_172: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(add_334, arg356_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg356_1 = None
        add_335: "f32[480]" = torch.ops.aten.add.Tensor(arg358_1, 1e-05);  arg358_1 = None
        sqrt_147: "f32[480]" = torch.ops.aten.sqrt.default(add_335);  add_335 = None
        reciprocal_147: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_453: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1176: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_1177: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        unsqueeze_1178: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1179: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        sub_147: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_1177);  convolution_172 = unsqueeze_1177 = None
        mul_454: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
        unsqueeze_1180: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_1181: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_455: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1181);  mul_454 = unsqueeze_1181 = None
        unsqueeze_1182: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_1183: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_336: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1183);  mul_455 = unsqueeze_1183 = None
        relu_74: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_336);  add_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_173: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_74, arg361_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg361_1 = None
        add_337: "f32[480]" = torch.ops.aten.add.Tensor(arg363_1, 1e-05);  arg363_1 = None
        sqrt_148: "f32[480]" = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_148: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_456: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1184: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg362_1, -1);  arg362_1 = None
        unsqueeze_1185: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        unsqueeze_1186: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1187: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        sub_148: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_173, unsqueeze_1185);  convolution_173 = unsqueeze_1185 = None
        mul_457: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
        unsqueeze_1188: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_1189: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_458: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1189);  mul_457 = unsqueeze_1189 = None
        unsqueeze_1190: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_1191: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_338: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1191);  mul_458 = unsqueeze_1191 = None
        relu_75: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_338);  add_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_58: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_74, relu_75], 1);  relu_74 = relu_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_13: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(cat_58, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_174: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_13, arg366_1, arg367_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg366_1 = arg367_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_76: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_174);  convolution_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_175: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_76, arg368_1, arg369_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_76 = arg368_1 = arg369_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_339: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_175, 3);  convolution_175 = None
        clamp_min_12: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_339, 0);  add_339 = None
        clamp_max_12: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
        div_12: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_12, 6);  clamp_max_12 = None
        mul_459: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(cat_58, div_12);  cat_58 = div_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_176: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_459, arg370_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_459 = arg370_1 = None
        add_340: "f32[80]" = torch.ops.aten.add.Tensor(arg372_1, 1e-05);  arg372_1 = None
        sqrt_149: "f32[80]" = torch.ops.aten.sqrt.default(add_340);  add_340 = None
        reciprocal_149: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_460: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1192: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg371_1, -1);  arg371_1 = None
        unsqueeze_1193: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        unsqueeze_1194: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_460, -1);  mul_460 = None
        unsqueeze_1195: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        sub_149: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_1193);  convolution_176 = unsqueeze_1193 = None
        mul_461: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_1197: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_462: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_461, unsqueeze_1197);  mul_461 = unsqueeze_1197 = None
        unsqueeze_1198: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_1199: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_341: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_1199);  mul_462 = unsqueeze_1199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_177: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_341, arg375_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg375_1 = None
        add_342: "f32[80]" = torch.ops.aten.add.Tensor(arg377_1, 1e-05);  arg377_1 = None
        sqrt_150: "f32[80]" = torch.ops.aten.sqrt.default(add_342);  add_342 = None
        reciprocal_150: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_463: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1200: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg376_1, -1);  arg376_1 = None
        unsqueeze_1201: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        unsqueeze_1202: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_463, -1);  mul_463 = None
        unsqueeze_1203: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        sub_150: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_1201);  convolution_177 = unsqueeze_1201 = None
        mul_464: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_1205: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_465: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_464, unsqueeze_1205);  mul_464 = unsqueeze_1205 = None
        unsqueeze_1206: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_1207: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_343: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_465, unsqueeze_1207);  mul_465 = unsqueeze_1207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_59: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_341, add_343], 1);  add_341 = add_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_344: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_59, add_334);  cat_59 = add_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_178: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(add_344, arg380_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg380_1 = None
        add_345: "f32[480]" = torch.ops.aten.add.Tensor(arg382_1, 1e-05);  arg382_1 = None
        sqrt_151: "f32[480]" = torch.ops.aten.sqrt.default(add_345);  add_345 = None
        reciprocal_151: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_466: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1208: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg381_1, -1);  arg381_1 = None
        unsqueeze_1209: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        unsqueeze_1210: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_466, -1);  mul_466 = None
        unsqueeze_1211: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        sub_151: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_1209);  convolution_178 = unsqueeze_1209 = None
        mul_467: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg383_1, -1);  arg383_1 = None
        unsqueeze_1213: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_468: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_467, unsqueeze_1213);  mul_467 = unsqueeze_1213 = None
        unsqueeze_1214: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_1215: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_346: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_1215);  mul_468 = unsqueeze_1215 = None
        relu_77: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_346);  add_346 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_179: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_77, arg385_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg385_1 = None
        add_347: "f32[480]" = torch.ops.aten.add.Tensor(arg387_1, 1e-05);  arg387_1 = None
        sqrt_152: "f32[480]" = torch.ops.aten.sqrt.default(add_347);  add_347 = None
        reciprocal_152: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_469: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1216: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
        unsqueeze_1217: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        unsqueeze_1218: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_469, -1);  mul_469 = None
        unsqueeze_1219: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        sub_152: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_179, unsqueeze_1217);  convolution_179 = unsqueeze_1217 = None
        mul_470: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_1221: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_471: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_1221);  mul_470 = unsqueeze_1221 = None
        unsqueeze_1222: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_1223: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_348: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_471, unsqueeze_1223);  mul_471 = unsqueeze_1223 = None
        relu_78: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_348);  add_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_60: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_77, relu_78], 1);  relu_77 = relu_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_180: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(cat_60, arg390_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_60 = arg390_1 = None
        add_349: "f32[80]" = torch.ops.aten.add.Tensor(arg392_1, 1e-05);  arg392_1 = None
        sqrt_153: "f32[80]" = torch.ops.aten.sqrt.default(add_349);  add_349 = None
        reciprocal_153: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_472: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1224: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_1225: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        unsqueeze_1226: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_472, -1);  mul_472 = None
        unsqueeze_1227: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        sub_153: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_1225);  convolution_180 = unsqueeze_1225 = None
        mul_473: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg393_1, -1);  arg393_1 = None
        unsqueeze_1229: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_474: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_1229);  mul_473 = unsqueeze_1229 = None
        unsqueeze_1230: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_1231: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_350: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_474, unsqueeze_1231);  mul_474 = unsqueeze_1231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_181: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_350, arg395_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg395_1 = None
        add_351: "f32[80]" = torch.ops.aten.add.Tensor(arg397_1, 1e-05);  arg397_1 = None
        sqrt_154: "f32[80]" = torch.ops.aten.sqrt.default(add_351);  add_351 = None
        reciprocal_154: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_475: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1232: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg396_1, -1);  arg396_1 = None
        unsqueeze_1233: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        unsqueeze_1234: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_475, -1);  mul_475 = None
        unsqueeze_1235: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        sub_154: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_1233);  convolution_181 = unsqueeze_1233 = None
        mul_476: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_1237: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_477: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_1237);  mul_476 = unsqueeze_1237 = None
        unsqueeze_1238: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_1239: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_352: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_477, unsqueeze_1239);  mul_477 = unsqueeze_1239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_61: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_350, add_352], 1);  add_350 = add_352 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_353: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_61, add_344);  cat_61 = add_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_182: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(add_353, arg400_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg400_1 = None
        add_354: "f32[480]" = torch.ops.aten.add.Tensor(arg402_1, 1e-05);  arg402_1 = None
        sqrt_155: "f32[480]" = torch.ops.aten.sqrt.default(add_354);  add_354 = None
        reciprocal_155: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_155);  sqrt_155 = None
        mul_478: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_155, 1);  reciprocal_155 = None
        unsqueeze_1240: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg401_1, -1);  arg401_1 = None
        unsqueeze_1241: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, -1);  unsqueeze_1240 = None
        unsqueeze_1242: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_478, -1);  mul_478 = None
        unsqueeze_1243: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, -1);  unsqueeze_1242 = None
        sub_155: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_1241);  convolution_182 = unsqueeze_1241 = None
        mul_479: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_1243);  sub_155 = unsqueeze_1243 = None
        unsqueeze_1244: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_1245: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, -1);  unsqueeze_1244 = None
        mul_480: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_479, unsqueeze_1245);  mul_479 = unsqueeze_1245 = None
        unsqueeze_1246: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_1247: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, -1);  unsqueeze_1246 = None
        add_355: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_480, unsqueeze_1247);  mul_480 = unsqueeze_1247 = None
        relu_79: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_355);  add_355 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_183: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_79, arg405_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  arg405_1 = None
        add_356: "f32[480]" = torch.ops.aten.add.Tensor(arg407_1, 1e-05);  arg407_1 = None
        sqrt_156: "f32[480]" = torch.ops.aten.sqrt.default(add_356);  add_356 = None
        reciprocal_156: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_156);  sqrt_156 = None
        mul_481: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_156, 1);  reciprocal_156 = None
        unsqueeze_1248: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg406_1, -1);  arg406_1 = None
        unsqueeze_1249: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, -1);  unsqueeze_1248 = None
        unsqueeze_1250: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_481, -1);  mul_481 = None
        unsqueeze_1251: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, -1);  unsqueeze_1250 = None
        sub_156: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_183, unsqueeze_1249);  convolution_183 = unsqueeze_1249 = None
        mul_482: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_1251);  sub_156 = unsqueeze_1251 = None
        unsqueeze_1252: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg408_1, -1);  arg408_1 = None
        unsqueeze_1253: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, -1);  unsqueeze_1252 = None
        mul_483: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_482, unsqueeze_1253);  mul_482 = unsqueeze_1253 = None
        unsqueeze_1254: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_1255: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, -1);  unsqueeze_1254 = None
        add_357: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_483, unsqueeze_1255);  mul_483 = unsqueeze_1255 = None
        relu_80: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_357);  add_357 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_62: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_79, relu_80], 1);  relu_79 = relu_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(cat_62, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_184: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_14, arg410_1, arg411_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg410_1 = arg411_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_81: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_184);  convolution_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_185: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_81, arg412_1, arg413_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_81 = arg412_1 = arg413_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_358: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_185, 3);  convolution_185 = None
        clamp_min_13: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_358, 0);  add_358 = None
        clamp_max_13: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
        div_13: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_13, 6);  clamp_max_13 = None
        mul_484: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(cat_62, div_13);  cat_62 = div_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:61 in forward, code: x1 = self.primary_conv(x)
        convolution_186: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_484, arg414_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_484 = arg414_1 = None
        add_359: "f32[80]" = torch.ops.aten.add.Tensor(arg416_1, 1e-05);  arg416_1 = None
        sqrt_157: "f32[80]" = torch.ops.aten.sqrt.default(add_359);  add_359 = None
        reciprocal_157: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_157);  sqrt_157 = None
        mul_485: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_157, 1);  reciprocal_157 = None
        unsqueeze_1256: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
        unsqueeze_1257: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, -1);  unsqueeze_1256 = None
        unsqueeze_1258: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_485, -1);  mul_485 = None
        unsqueeze_1259: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, -1);  unsqueeze_1258 = None
        sub_157: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_1257);  convolution_186 = unsqueeze_1257 = None
        mul_486: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_1259);  sub_157 = unsqueeze_1259 = None
        unsqueeze_1260: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg417_1, -1);  arg417_1 = None
        unsqueeze_1261: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, -1);  unsqueeze_1260 = None
        mul_487: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_1261);  mul_486 = unsqueeze_1261 = None
        unsqueeze_1262: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_1263: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, -1);  unsqueeze_1262 = None
        add_360: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_487, unsqueeze_1263);  mul_487 = unsqueeze_1263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:62 in forward, code: x2 = self.cheap_operation(x1)
        convolution_187: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_360, arg419_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  arg419_1 = None
        add_361: "f32[80]" = torch.ops.aten.add.Tensor(arg421_1, 1e-05);  arg421_1 = None
        sqrt_158: "f32[80]" = torch.ops.aten.sqrt.default(add_361);  add_361 = None
        reciprocal_158: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_158);  sqrt_158 = None
        mul_488: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_158, 1);  reciprocal_158 = None
        unsqueeze_1264: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_1265: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, -1);  unsqueeze_1264 = None
        unsqueeze_1266: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_488, -1);  mul_488 = None
        unsqueeze_1267: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, -1);  unsqueeze_1266 = None
        sub_158: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1265);  convolution_187 = unsqueeze_1265 = None
        mul_489: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_1267);  sub_158 = unsqueeze_1267 = None
        unsqueeze_1268: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
        unsqueeze_1269: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, -1);  unsqueeze_1268 = None
        mul_490: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_489, unsqueeze_1269);  mul_489 = unsqueeze_1269 = None
        unsqueeze_1270: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_1271: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, -1);  unsqueeze_1270 = None
        add_362: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_490, unsqueeze_1271);  mul_490 = unsqueeze_1271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:63 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_63: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_360, add_362], 1);  add_360 = add_362 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:183 in forward, code: x += self.shortcut(shortcut)
        add_363: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_63, add_353);  cat_63 = add_353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:111 in forward, code: x = self.conv(x)
        convolution_188: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(add_363, arg424_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_363 = arg424_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_364: "f32[960]" = torch.ops.aten.add.Tensor(arg426_1, 1e-05);  arg426_1 = None
        sqrt_159: "f32[960]" = torch.ops.aten.sqrt.default(add_364);  add_364 = None
        reciprocal_159: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_159);  sqrt_159 = None
        mul_491: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_159, 1);  reciprocal_159 = None
        unsqueeze_1272: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_1273: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, -1);  unsqueeze_1272 = None
        unsqueeze_1274: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_491, -1);  mul_491 = None
        unsqueeze_1275: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, -1);  unsqueeze_1274 = None
        sub_159: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_188, unsqueeze_1273);  convolution_188 = unsqueeze_1273 = None
        mul_492: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_1275);  sub_159 = unsqueeze_1275 = None
        unsqueeze_1276: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg427_1, -1);  arg427_1 = None
        unsqueeze_1277: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, -1);  unsqueeze_1276 = None
        mul_493: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_492, unsqueeze_1277);  mul_492 = unsqueeze_1277 = None
        unsqueeze_1278: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg428_1, -1);  arg428_1 = None
        unsqueeze_1279: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, -1);  unsqueeze_1278 = None
        add_365: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_493, unsqueeze_1279);  mul_493 = unsqueeze_1279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_82: "f32[8, 960, 7, 7]" = torch.ops.aten.relu.default(add_365);  add_365 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_15: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(relu_82, [-1, -2], True);  relu_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:295 in forward_head, code: x = self.conv_head(x)
        convolution_189: "f32[8, 1280, 1, 1]" = torch.ops.aten.convolution.default(mean_15, arg429_1, arg430_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg429_1 = arg430_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/ghostnet.py:296 in forward_head, code: x = self.act2(x)
        relu_83: "f32[8, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_189);  convolution_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_1: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        view_3: "f32[8, 1280]" = torch.ops.aten.view.default(relu_83, [8, 1280]);  relu_83 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg432_1, view_3, permute_1);  arg432_1 = view_3 = permute_1 = None
        return (addmm_1,)
        