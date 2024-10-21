class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[32, 1, 3, 3]", arg7_1: "f32[32]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[32, 32, 1, 1]", arg12_1: "f32[32]", arg13_1: "f32[32]", arg14_1: "f32[32]", arg15_1: "f32[32]", arg16_1: "f32[96, 16, 1, 1]", arg17_1: "f32[96, 16, 1, 1]", arg18_1: "f32[192]", arg19_1: "f32[192]", arg20_1: "f32[192]", arg21_1: "f32[192]", arg22_1: "f32[64, 1, 3, 3]", arg23_1: "f32[64, 1, 5, 5]", arg24_1: "f32[64, 1, 7, 7]", arg25_1: "f32[192]", arg26_1: "f32[192]", arg27_1: "f32[192]", arg28_1: "f32[192]", arg29_1: "f32[20, 96, 1, 1]", arg30_1: "f32[20, 96, 1, 1]", arg31_1: "f32[40]", arg32_1: "f32[40]", arg33_1: "f32[40]", arg34_1: "f32[40]", arg35_1: "f32[60, 20, 1, 1]", arg36_1: "f32[60, 20, 1, 1]", arg37_1: "f32[120]", arg38_1: "f32[120]", arg39_1: "f32[120]", arg40_1: "f32[120]", arg41_1: "f32[120, 1, 3, 3]", arg42_1: "f32[120]", arg43_1: "f32[120]", arg44_1: "f32[120]", arg45_1: "f32[120]", arg46_1: "f32[20, 60, 1, 1]", arg47_1: "f32[20, 60, 1, 1]", arg48_1: "f32[40]", arg49_1: "f32[40]", arg50_1: "f32[40]", arg51_1: "f32[40]", arg52_1: "f32[240, 40, 1, 1]", arg53_1: "f32[240]", arg54_1: "f32[240]", arg55_1: "f32[240]", arg56_1: "f32[240]", arg57_1: "f32[60, 1, 3, 3]", arg58_1: "f32[60, 1, 5, 5]", arg59_1: "f32[60, 1, 7, 7]", arg60_1: "f32[60, 1, 9, 9]", arg61_1: "f32[240]", arg62_1: "f32[240]", arg63_1: "f32[240]", arg64_1: "f32[240]", arg65_1: "f32[20, 240, 1, 1]", arg66_1: "f32[20]", arg67_1: "f32[240, 20, 1, 1]", arg68_1: "f32[240]", arg69_1: "f32[56, 240, 1, 1]", arg70_1: "f32[56]", arg71_1: "f32[56]", arg72_1: "f32[56]", arg73_1: "f32[56]", arg74_1: "f32[168, 28, 1, 1]", arg75_1: "f32[168, 28, 1, 1]", arg76_1: "f32[336]", arg77_1: "f32[336]", arg78_1: "f32[336]", arg79_1: "f32[336]", arg80_1: "f32[168, 1, 3, 3]", arg81_1: "f32[168, 1, 5, 5]", arg82_1: "f32[336]", arg83_1: "f32[336]", arg84_1: "f32[336]", arg85_1: "f32[336]", arg86_1: "f32[28, 336, 1, 1]", arg87_1: "f32[28]", arg88_1: "f32[336, 28, 1, 1]", arg89_1: "f32[336]", arg90_1: "f32[28, 168, 1, 1]", arg91_1: "f32[28, 168, 1, 1]", arg92_1: "f32[56]", arg93_1: "f32[56]", arg94_1: "f32[56]", arg95_1: "f32[56]", arg96_1: "f32[168, 28, 1, 1]", arg97_1: "f32[168, 28, 1, 1]", arg98_1: "f32[336]", arg99_1: "f32[336]", arg100_1: "f32[336]", arg101_1: "f32[336]", arg102_1: "f32[168, 1, 3, 3]", arg103_1: "f32[168, 1, 5, 5]", arg104_1: "f32[336]", arg105_1: "f32[336]", arg106_1: "f32[336]", arg107_1: "f32[336]", arg108_1: "f32[28, 336, 1, 1]", arg109_1: "f32[28]", arg110_1: "f32[336, 28, 1, 1]", arg111_1: "f32[336]", arg112_1: "f32[28, 168, 1, 1]", arg113_1: "f32[28, 168, 1, 1]", arg114_1: "f32[56]", arg115_1: "f32[56]", arg116_1: "f32[56]", arg117_1: "f32[56]", arg118_1: "f32[168, 28, 1, 1]", arg119_1: "f32[168, 28, 1, 1]", arg120_1: "f32[336]", arg121_1: "f32[336]", arg122_1: "f32[336]", arg123_1: "f32[336]", arg124_1: "f32[168, 1, 3, 3]", arg125_1: "f32[168, 1, 5, 5]", arg126_1: "f32[336]", arg127_1: "f32[336]", arg128_1: "f32[336]", arg129_1: "f32[336]", arg130_1: "f32[28, 336, 1, 1]", arg131_1: "f32[28]", arg132_1: "f32[336, 28, 1, 1]", arg133_1: "f32[336]", arg134_1: "f32[28, 168, 1, 1]", arg135_1: "f32[28, 168, 1, 1]", arg136_1: "f32[56]", arg137_1: "f32[56]", arg138_1: "f32[56]", arg139_1: "f32[56]", arg140_1: "f32[336, 56, 1, 1]", arg141_1: "f32[336]", arg142_1: "f32[336]", arg143_1: "f32[336]", arg144_1: "f32[336]", arg145_1: "f32[112, 1, 3, 3]", arg146_1: "f32[112, 1, 5, 5]", arg147_1: "f32[112, 1, 7, 7]", arg148_1: "f32[336]", arg149_1: "f32[336]", arg150_1: "f32[336]", arg151_1: "f32[336]", arg152_1: "f32[14, 336, 1, 1]", arg153_1: "f32[14]", arg154_1: "f32[336, 14, 1, 1]", arg155_1: "f32[336]", arg156_1: "f32[104, 336, 1, 1]", arg157_1: "f32[104]", arg158_1: "f32[104]", arg159_1: "f32[104]", arg160_1: "f32[104]", arg161_1: "f32[312, 52, 1, 1]", arg162_1: "f32[312, 52, 1, 1]", arg163_1: "f32[624]", arg164_1: "f32[624]", arg165_1: "f32[624]", arg166_1: "f32[624]", arg167_1: "f32[156, 1, 3, 3]", arg168_1: "f32[156, 1, 5, 5]", arg169_1: "f32[156, 1, 7, 7]", arg170_1: "f32[156, 1, 9, 9]", arg171_1: "f32[624]", arg172_1: "f32[624]", arg173_1: "f32[624]", arg174_1: "f32[624]", arg175_1: "f32[26, 624, 1, 1]", arg176_1: "f32[26]", arg177_1: "f32[624, 26, 1, 1]", arg178_1: "f32[624]", arg179_1: "f32[52, 312, 1, 1]", arg180_1: "f32[52, 312, 1, 1]", arg181_1: "f32[104]", arg182_1: "f32[104]", arg183_1: "f32[104]", arg184_1: "f32[104]", arg185_1: "f32[312, 52, 1, 1]", arg186_1: "f32[312, 52, 1, 1]", arg187_1: "f32[624]", arg188_1: "f32[624]", arg189_1: "f32[624]", arg190_1: "f32[624]", arg191_1: "f32[156, 1, 3, 3]", arg192_1: "f32[156, 1, 5, 5]", arg193_1: "f32[156, 1, 7, 7]", arg194_1: "f32[156, 1, 9, 9]", arg195_1: "f32[624]", arg196_1: "f32[624]", arg197_1: "f32[624]", arg198_1: "f32[624]", arg199_1: "f32[26, 624, 1, 1]", arg200_1: "f32[26]", arg201_1: "f32[624, 26, 1, 1]", arg202_1: "f32[624]", arg203_1: "f32[52, 312, 1, 1]", arg204_1: "f32[52, 312, 1, 1]", arg205_1: "f32[104]", arg206_1: "f32[104]", arg207_1: "f32[104]", arg208_1: "f32[104]", arg209_1: "f32[312, 52, 1, 1]", arg210_1: "f32[312, 52, 1, 1]", arg211_1: "f32[624]", arg212_1: "f32[624]", arg213_1: "f32[624]", arg214_1: "f32[624]", arg215_1: "f32[156, 1, 3, 3]", arg216_1: "f32[156, 1, 5, 5]", arg217_1: "f32[156, 1, 7, 7]", arg218_1: "f32[156, 1, 9, 9]", arg219_1: "f32[624]", arg220_1: "f32[624]", arg221_1: "f32[624]", arg222_1: "f32[624]", arg223_1: "f32[26, 624, 1, 1]", arg224_1: "f32[26]", arg225_1: "f32[624, 26, 1, 1]", arg226_1: "f32[624]", arg227_1: "f32[52, 312, 1, 1]", arg228_1: "f32[52, 312, 1, 1]", arg229_1: "f32[104]", arg230_1: "f32[104]", arg231_1: "f32[104]", arg232_1: "f32[104]", arg233_1: "f32[624, 104, 1, 1]", arg234_1: "f32[624]", arg235_1: "f32[624]", arg236_1: "f32[624]", arg237_1: "f32[624]", arg238_1: "f32[624, 1, 3, 3]", arg239_1: "f32[624]", arg240_1: "f32[624]", arg241_1: "f32[624]", arg242_1: "f32[624]", arg243_1: "f32[52, 624, 1, 1]", arg244_1: "f32[52]", arg245_1: "f32[624, 52, 1, 1]", arg246_1: "f32[624]", arg247_1: "f32[160, 624, 1, 1]", arg248_1: "f32[160]", arg249_1: "f32[160]", arg250_1: "f32[160]", arg251_1: "f32[160]", arg252_1: "f32[240, 80, 1, 1]", arg253_1: "f32[240, 80, 1, 1]", arg254_1: "f32[480]", arg255_1: "f32[480]", arg256_1: "f32[480]", arg257_1: "f32[480]", arg258_1: "f32[120, 1, 3, 3]", arg259_1: "f32[120, 1, 5, 5]", arg260_1: "f32[120, 1, 7, 7]", arg261_1: "f32[120, 1, 9, 9]", arg262_1: "f32[480]", arg263_1: "f32[480]", arg264_1: "f32[480]", arg265_1: "f32[480]", arg266_1: "f32[80, 480, 1, 1]", arg267_1: "f32[80]", arg268_1: "f32[480, 80, 1, 1]", arg269_1: "f32[480]", arg270_1: "f32[80, 240, 1, 1]", arg271_1: "f32[80, 240, 1, 1]", arg272_1: "f32[160]", arg273_1: "f32[160]", arg274_1: "f32[160]", arg275_1: "f32[160]", arg276_1: "f32[240, 80, 1, 1]", arg277_1: "f32[240, 80, 1, 1]", arg278_1: "f32[480]", arg279_1: "f32[480]", arg280_1: "f32[480]", arg281_1: "f32[480]", arg282_1: "f32[120, 1, 3, 3]", arg283_1: "f32[120, 1, 5, 5]", arg284_1: "f32[120, 1, 7, 7]", arg285_1: "f32[120, 1, 9, 9]", arg286_1: "f32[480]", arg287_1: "f32[480]", arg288_1: "f32[480]", arg289_1: "f32[480]", arg290_1: "f32[80, 480, 1, 1]", arg291_1: "f32[80]", arg292_1: "f32[480, 80, 1, 1]", arg293_1: "f32[480]", arg294_1: "f32[80, 240, 1, 1]", arg295_1: "f32[80, 240, 1, 1]", arg296_1: "f32[160]", arg297_1: "f32[160]", arg298_1: "f32[160]", arg299_1: "f32[160]", arg300_1: "f32[240, 80, 1, 1]", arg301_1: "f32[240, 80, 1, 1]", arg302_1: "f32[480]", arg303_1: "f32[480]", arg304_1: "f32[480]", arg305_1: "f32[480]", arg306_1: "f32[120, 1, 3, 3]", arg307_1: "f32[120, 1, 5, 5]", arg308_1: "f32[120, 1, 7, 7]", arg309_1: "f32[120, 1, 9, 9]", arg310_1: "f32[480]", arg311_1: "f32[480]", arg312_1: "f32[480]", arg313_1: "f32[480]", arg314_1: "f32[80, 480, 1, 1]", arg315_1: "f32[80]", arg316_1: "f32[480, 80, 1, 1]", arg317_1: "f32[480]", arg318_1: "f32[80, 240, 1, 1]", arg319_1: "f32[80, 240, 1, 1]", arg320_1: "f32[160]", arg321_1: "f32[160]", arg322_1: "f32[160]", arg323_1: "f32[160]", arg324_1: "f32[960, 160, 1, 1]", arg325_1: "f32[960]", arg326_1: "f32[960]", arg327_1: "f32[960]", arg328_1: "f32[960]", arg329_1: "f32[240, 1, 3, 3]", arg330_1: "f32[240, 1, 5, 5]", arg331_1: "f32[240, 1, 7, 7]", arg332_1: "f32[240, 1, 9, 9]", arg333_1: "f32[960]", arg334_1: "f32[960]", arg335_1: "f32[960]", arg336_1: "f32[960]", arg337_1: "f32[80, 960, 1, 1]", arg338_1: "f32[80]", arg339_1: "f32[960, 80, 1, 1]", arg340_1: "f32[960]", arg341_1: "f32[264, 960, 1, 1]", arg342_1: "f32[264]", arg343_1: "f32[264]", arg344_1: "f32[264]", arg345_1: "f32[264]", arg346_1: "f32[1584, 264, 1, 1]", arg347_1: "f32[1584]", arg348_1: "f32[1584]", arg349_1: "f32[1584]", arg350_1: "f32[1584]", arg351_1: "f32[396, 1, 3, 3]", arg352_1: "f32[396, 1, 5, 5]", arg353_1: "f32[396, 1, 7, 7]", arg354_1: "f32[396, 1, 9, 9]", arg355_1: "f32[1584]", arg356_1: "f32[1584]", arg357_1: "f32[1584]", arg358_1: "f32[1584]", arg359_1: "f32[132, 1584, 1, 1]", arg360_1: "f32[132]", arg361_1: "f32[1584, 132, 1, 1]", arg362_1: "f32[1584]", arg363_1: "f32[132, 792, 1, 1]", arg364_1: "f32[132, 792, 1, 1]", arg365_1: "f32[264]", arg366_1: "f32[264]", arg367_1: "f32[264]", arg368_1: "f32[264]", arg369_1: "f32[1584, 264, 1, 1]", arg370_1: "f32[1584]", arg371_1: "f32[1584]", arg372_1: "f32[1584]", arg373_1: "f32[1584]", arg374_1: "f32[396, 1, 3, 3]", arg375_1: "f32[396, 1, 5, 5]", arg376_1: "f32[396, 1, 7, 7]", arg377_1: "f32[396, 1, 9, 9]", arg378_1: "f32[1584]", arg379_1: "f32[1584]", arg380_1: "f32[1584]", arg381_1: "f32[1584]", arg382_1: "f32[132, 1584, 1, 1]", arg383_1: "f32[132]", arg384_1: "f32[1584, 132, 1, 1]", arg385_1: "f32[1584]", arg386_1: "f32[132, 792, 1, 1]", arg387_1: "f32[132, 792, 1, 1]", arg388_1: "f32[264]", arg389_1: "f32[264]", arg390_1: "f32[264]", arg391_1: "f32[264]", arg392_1: "f32[1584, 264, 1, 1]", arg393_1: "f32[1584]", arg394_1: "f32[1584]", arg395_1: "f32[1584]", arg396_1: "f32[1584]", arg397_1: "f32[396, 1, 3, 3]", arg398_1: "f32[396, 1, 5, 5]", arg399_1: "f32[396, 1, 7, 7]", arg400_1: "f32[396, 1, 9, 9]", arg401_1: "f32[1584]", arg402_1: "f32[1584]", arg403_1: "f32[1584]", arg404_1: "f32[1584]", arg405_1: "f32[132, 1584, 1, 1]", arg406_1: "f32[132]", arg407_1: "f32[1584, 132, 1, 1]", arg408_1: "f32[1584]", arg409_1: "f32[132, 792, 1, 1]", arg410_1: "f32[132, 792, 1, 1]", arg411_1: "f32[264]", arg412_1: "f32[264]", arg413_1: "f32[264]", arg414_1: "f32[264]", arg415_1: "f32[1536, 264, 1, 1]", arg416_1: "f32[1536]", arg417_1: "f32[1536]", arg418_1: "f32[1536]", arg419_1: "f32[1536]", arg420_1: "f32[1000, 1536]", arg421_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/efficientnet.py:252 in forward_features, code: x = self.conv_stem(x)
        convolution_155: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_464: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_465: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        sub_58: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_465);  convolution_155 = unsqueeze_465 = None
        add_130: "f32[32]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_58: "f32[32]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_58: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_238: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_466: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_238, -1);  mul_238 = None
        unsqueeze_467: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        mul_239: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_469: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_240: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_469);  mul_239 = unsqueeze_469 = None
        unsqueeze_470: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_471: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_131: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_240, unsqueeze_471);  mul_240 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_7: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_131);  add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_156: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu_7, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_472: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_473: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        sub_59: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_473);  convolution_156 = unsqueeze_473 = None
        add_132: "f32[32]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_59: "f32[32]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_59: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_241: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_474: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_241, -1);  mul_241 = None
        unsqueeze_475: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        mul_242: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_477: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_243: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_477);  mul_242 = unsqueeze_477 = None
        unsqueeze_478: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_479: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_133: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_243, unsqueeze_479);  mul_243 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_8: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_133);  add_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_157: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu_8, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_480: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_481: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        sub_60: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_481);  convolution_157 = unsqueeze_481 = None
        add_134: "f32[32]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_60: "f32[32]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_60: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_244: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_482: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_244, -1);  mul_244 = None
        unsqueeze_483: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        mul_245: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_485: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_246: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_485);  mul_245 = unsqueeze_485 = None
        unsqueeze_486: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_487: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_135: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_487);  mul_246 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:197 in forward, code: x = self.drop_path(x) + shortcut
        add_136: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(add_135, relu_7);  add_135 = relu_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_101 = torch.ops.aten.split_with_sizes.default(add_136, [16, 16], 1);  add_136 = None
        getitem_320: "f32[8, 16, 112, 112]" = split_with_sizes_101[0]
        getitem_321: "f32[8, 16, 112, 112]" = split_with_sizes_101[1];  split_with_sizes_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_158: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(getitem_320, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_320 = arg16_1 = None
        convolution_159: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(getitem_321, arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_321 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_41: "f32[8, 192, 112, 112]" = torch.ops.aten.cat.default([convolution_158, convolution_159], 1);  convolution_158 = convolution_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_488: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_489: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        sub_61: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(cat_41, unsqueeze_489);  cat_41 = unsqueeze_489 = None
        add_137: "f32[192]" = torch.ops.aten.add.Tensor(arg19_1, 1e-05);  arg19_1 = None
        sqrt_61: "f32[192]" = torch.ops.aten.sqrt.default(add_137);  add_137 = None
        reciprocal_61: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_247: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_490: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
        unsqueeze_491: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        mul_248: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_493: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_249: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_493);  mul_248 = unsqueeze_493 = None
        unsqueeze_494: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_495: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_138: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(mul_249, unsqueeze_495);  mul_249 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_9: "f32[8, 192, 112, 112]" = torch.ops.aten.relu.default(add_138);  add_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_103 = torch.ops.aten.split_with_sizes.default(relu_9, [64, 64, 64], 1)
        getitem_325: "f32[8, 64, 112, 112]" = split_with_sizes_103[0];  split_with_sizes_103 = None
        split_with_sizes_104 = torch.ops.aten.split_with_sizes.default(relu_9, [64, 64, 64], 1)
        getitem_329: "f32[8, 64, 112, 112]" = split_with_sizes_104[1];  split_with_sizes_104 = None
        split_with_sizes_105 = torch.ops.aten.split_with_sizes.default(relu_9, [64, 64, 64], 1);  relu_9 = None
        getitem_333: "f32[8, 64, 112, 112]" = split_with_sizes_105[2];  split_with_sizes_105 = None
        convolution_160: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_325, arg22_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  getitem_325 = arg22_1 = None
        convolution_161: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_329, arg23_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 64);  getitem_329 = arg23_1 = None
        convolution_162: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_333, arg24_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 64);  getitem_333 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_42: "f32[8, 192, 56, 56]" = torch.ops.aten.cat.default([convolution_160, convolution_161, convolution_162], 1);  convolution_160 = convolution_161 = convolution_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_496: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_497: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        sub_62: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_42, unsqueeze_497);  cat_42 = unsqueeze_497 = None
        add_139: "f32[192]" = torch.ops.aten.add.Tensor(arg26_1, 1e-05);  arg26_1 = None
        sqrt_62: "f32[192]" = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_62: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_250: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_498: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_499: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        mul_251: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_501: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_252: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_501);  mul_251 = unsqueeze_501 = None
        unsqueeze_502: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_503: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_140: "f32[8, 192, 56, 56]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_503);  mul_252 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_10: "f32[8, 192, 56, 56]" = torch.ops.aten.relu.default(add_140);  add_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_107 = torch.ops.aten.split_with_sizes.default(relu_10, [96, 96], 1)
        getitem_336: "f32[8, 96, 56, 56]" = split_with_sizes_107[0];  split_with_sizes_107 = None
        split_with_sizes_108 = torch.ops.aten.split_with_sizes.default(relu_10, [96, 96], 1);  relu_10 = None
        getitem_339: "f32[8, 96, 56, 56]" = split_with_sizes_108[1];  split_with_sizes_108 = None
        convolution_163: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_336, arg29_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_336 = arg29_1 = None
        convolution_164: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_339, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_339 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_43: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([convolution_163, convolution_164], 1);  convolution_163 = convolution_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_504: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_505: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        sub_63: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_43, unsqueeze_505);  cat_43 = unsqueeze_505 = None
        add_141: "f32[40]" = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_63: "f32[40]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_63: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_253: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_506: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_507: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        mul_254: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_509: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_255: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_509);  mul_254 = unsqueeze_509 = None
        unsqueeze_510: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_511: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_142: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_511);  mul_255 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_109 = torch.ops.aten.split_with_sizes.default(add_142, [20, 20], 1)
        getitem_340: "f32[8, 20, 56, 56]" = split_with_sizes_109[0]
        getitem_341: "f32[8, 20, 56, 56]" = split_with_sizes_109[1];  split_with_sizes_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_165: "f32[8, 60, 56, 56]" = torch.ops.aten.convolution.default(getitem_340, arg35_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_340 = arg35_1 = None
        convolution_166: "f32[8, 60, 56, 56]" = torch.ops.aten.convolution.default(getitem_341, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_341 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_44: "f32[8, 120, 56, 56]" = torch.ops.aten.cat.default([convolution_165, convolution_166], 1);  convolution_165 = convolution_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_512: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_513: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        sub_64: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(cat_44, unsqueeze_513);  cat_44 = unsqueeze_513 = None
        add_143: "f32[120]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_64: "f32[120]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_64: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_256: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_514: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_256, -1);  mul_256 = None
        unsqueeze_515: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        mul_257: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_517: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_258: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_257, unsqueeze_517);  mul_257 = unsqueeze_517 = None
        unsqueeze_518: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_519: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_144: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_519);  mul_258 = unsqueeze_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_11: "f32[8, 120, 56, 56]" = torch.ops.aten.relu.default(add_144);  add_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_167: "f32[8, 120, 56, 56]" = torch.ops.aten.convolution.default(relu_11, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  relu_11 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_520: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_521: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        sub_65: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_521);  convolution_167 = unsqueeze_521 = None
        add_145: "f32[120]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_65: "f32[120]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_65: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_259: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_522: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_259, -1);  mul_259 = None
        unsqueeze_523: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        mul_260: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_525: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_261: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_260, unsqueeze_525);  mul_260 = unsqueeze_525 = None
        unsqueeze_526: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_527: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_146: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_527);  mul_261 = unsqueeze_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_12: "f32[8, 120, 56, 56]" = torch.ops.aten.relu.default(add_146);  add_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_111 = torch.ops.aten.split_with_sizes.default(relu_12, [60, 60], 1)
        getitem_344: "f32[8, 60, 56, 56]" = split_with_sizes_111[0];  split_with_sizes_111 = None
        split_with_sizes_112 = torch.ops.aten.split_with_sizes.default(relu_12, [60, 60], 1);  relu_12 = None
        getitem_347: "f32[8, 60, 56, 56]" = split_with_sizes_112[1];  split_with_sizes_112 = None
        convolution_168: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_344, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_344 = arg46_1 = None
        convolution_169: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_347, arg47_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_347 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_45: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([convolution_168, convolution_169], 1);  convolution_168 = convolution_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_528: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_529: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        sub_66: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_45, unsqueeze_529);  cat_45 = unsqueeze_529 = None
        add_147: "f32[40]" = torch.ops.aten.add.Tensor(arg49_1, 1e-05);  arg49_1 = None
        sqrt_66: "f32[40]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_66: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_262: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_530: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_262, -1);  mul_262 = None
        unsqueeze_531: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        mul_263: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_533: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_264: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_533);  mul_263 = unsqueeze_533 = None
        unsqueeze_534: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_535: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_148: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(mul_264, unsqueeze_535);  mul_264 = unsqueeze_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_149: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(add_148, add_142);  add_148 = add_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_170: "f32[8, 240, 56, 56]" = torch.ops.aten.convolution.default(add_149, arg52_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_149 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_536: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_537: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        sub_67: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_537);  convolution_170 = unsqueeze_537 = None
        add_150: "f32[240]" = torch.ops.aten.add.Tensor(arg54_1, 1e-05);  arg54_1 = None
        sqrt_67: "f32[240]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_67: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_265: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_538: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_265, -1);  mul_265 = None
        unsqueeze_539: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        mul_266: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_541: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_267: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_541);  mul_266 = unsqueeze_541 = None
        unsqueeze_542: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_543: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_151: "f32[8, 240, 56, 56]" = torch.ops.aten.add.Tensor(mul_267, unsqueeze_543);  mul_267 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_64: "f32[8, 240, 56, 56]" = torch.ops.aten.sigmoid.default(add_151)
        mul_268: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(add_151, sigmoid_64);  add_151 = sigmoid_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_114 = torch.ops.aten.split_with_sizes.default(mul_268, [60, 60, 60, 60], 1)
        getitem_352: "f32[8, 60, 56, 56]" = split_with_sizes_114[0];  split_with_sizes_114 = None
        split_with_sizes_115 = torch.ops.aten.split_with_sizes.default(mul_268, [60, 60, 60, 60], 1)
        getitem_357: "f32[8, 60, 56, 56]" = split_with_sizes_115[1];  split_with_sizes_115 = None
        split_with_sizes_116 = torch.ops.aten.split_with_sizes.default(mul_268, [60, 60, 60, 60], 1)
        getitem_362: "f32[8, 60, 56, 56]" = split_with_sizes_116[2];  split_with_sizes_116 = None
        split_with_sizes_117 = torch.ops.aten.split_with_sizes.default(mul_268, [60, 60, 60, 60], 1);  mul_268 = None
        getitem_367: "f32[8, 60, 56, 56]" = split_with_sizes_117[3];  split_with_sizes_117 = None
        convolution_171: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(getitem_352, arg57_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 60);  getitem_352 = arg57_1 = None
        convolution_172: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(getitem_357, arg58_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 60);  getitem_357 = arg58_1 = None
        convolution_173: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(getitem_362, arg59_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 60);  getitem_362 = arg59_1 = None
        convolution_174: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(getitem_367, arg60_1, None, [2, 2], [4, 4], [1, 1], False, [0, 0], 60);  getitem_367 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_46: "f32[8, 240, 28, 28]" = torch.ops.aten.cat.default([convolution_171, convolution_172, convolution_173, convolution_174], 1);  convolution_171 = convolution_172 = convolution_173 = convolution_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_544: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_545: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        sub_68: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(cat_46, unsqueeze_545);  cat_46 = unsqueeze_545 = None
        add_152: "f32[240]" = torch.ops.aten.add.Tensor(arg62_1, 1e-05);  arg62_1 = None
        sqrt_68: "f32[240]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_68: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_269: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_546: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_269, -1);  mul_269 = None
        unsqueeze_547: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        mul_270: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_549: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_271: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_549);  mul_270 = unsqueeze_549 = None
        unsqueeze_550: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_551: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_153: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_271, unsqueeze_551);  mul_271 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_65: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_153)
        mul_272: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_153, sigmoid_65);  add_153 = sigmoid_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_17: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_272, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_175: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_17, arg65_1, arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg65_1 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_66: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_175)
        mul_273: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_175, sigmoid_66);  convolution_175 = sigmoid_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_176: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_273, arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_273 = arg67_1 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_67: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_176);  convolution_176 = None
        mul_274: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_272, sigmoid_67);  mul_272 = sigmoid_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_177: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(mul_274, arg69_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_274 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_552: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_553: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        sub_69: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_553);  convolution_177 = unsqueeze_553 = None
        add_154: "f32[56]" = torch.ops.aten.add.Tensor(arg71_1, 1e-05);  arg71_1 = None
        sqrt_69: "f32[56]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_69: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_275: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_554: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_275, -1);  mul_275 = None
        unsqueeze_555: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        mul_276: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_557: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_277: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_557);  mul_276 = unsqueeze_557 = None
        unsqueeze_558: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_559: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_155: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_277, unsqueeze_559);  mul_277 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_118 = torch.ops.aten.split_with_sizes.default(add_155, [28, 28], 1)
        getitem_368: "f32[8, 28, 28, 28]" = split_with_sizes_118[0]
        getitem_369: "f32[8, 28, 28, 28]" = split_with_sizes_118[1];  split_with_sizes_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_178: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_368, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_368 = arg74_1 = None
        convolution_179: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_369, arg75_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_369 = arg75_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_47: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_178, convolution_179], 1);  convolution_178 = convolution_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_560: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_561: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        sub_70: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_47, unsqueeze_561);  cat_47 = unsqueeze_561 = None
        add_156: "f32[336]" = torch.ops.aten.add.Tensor(arg77_1, 1e-05);  arg77_1 = None
        sqrt_70: "f32[336]" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_70: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_278: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_562: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_278, -1);  mul_278 = None
        unsqueeze_563: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        mul_279: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_565: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_280: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_279, unsqueeze_565);  mul_279 = unsqueeze_565 = None
        unsqueeze_566: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_567: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_157: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_280, unsqueeze_567);  mul_280 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_68: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_157)
        mul_281: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_157, sigmoid_68);  add_157 = sigmoid_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_120 = torch.ops.aten.split_with_sizes.default(mul_281, [168, 168], 1)
        getitem_372: "f32[8, 168, 28, 28]" = split_with_sizes_120[0];  split_with_sizes_120 = None
        split_with_sizes_121 = torch.ops.aten.split_with_sizes.default(mul_281, [168, 168], 1);  mul_281 = None
        getitem_375: "f32[8, 168, 28, 28]" = split_with_sizes_121[1];  split_with_sizes_121 = None
        convolution_180: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_372, arg80_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_372 = arg80_1 = None
        convolution_181: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_375, arg81_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_375 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_48: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_180, convolution_181], 1);  convolution_180 = convolution_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_568: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_569: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        sub_71: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_48, unsqueeze_569);  cat_48 = unsqueeze_569 = None
        add_158: "f32[336]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_71: "f32[336]" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_71: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_282: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_570: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
        unsqueeze_571: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        mul_283: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_573: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_284: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_573);  mul_283 = unsqueeze_573 = None
        unsqueeze_574: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_575: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_159: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_575);  mul_284 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_69: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_159)
        mul_285: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_159, sigmoid_69);  add_159 = sigmoid_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_18: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_285, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_182: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_18, arg86_1, arg87_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg86_1 = arg87_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_70: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_182)
        mul_286: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_182, sigmoid_70);  convolution_182 = sigmoid_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_183: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_286, arg88_1, arg89_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_286 = arg88_1 = arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_71: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_183);  convolution_183 = None
        mul_287: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_285, sigmoid_71);  mul_285 = sigmoid_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_122 = torch.ops.aten.split_with_sizes.default(mul_287, [168, 168], 1);  mul_287 = None
        getitem_376: "f32[8, 168, 28, 28]" = split_with_sizes_122[0]
        getitem_377: "f32[8, 168, 28, 28]" = split_with_sizes_122[1];  split_with_sizes_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_184: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_376, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_376 = arg90_1 = None
        convolution_185: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_377, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_377 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_49: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_184, convolution_185], 1);  convolution_184 = convolution_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_576: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_577: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        sub_72: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_49, unsqueeze_577);  cat_49 = unsqueeze_577 = None
        add_160: "f32[56]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_72: "f32[56]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_72: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_288: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_578: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_579: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        mul_289: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_581: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_290: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_581);  mul_289 = unsqueeze_581 = None
        unsqueeze_582: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_583: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_161: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_583);  mul_290 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_162: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_161, add_155);  add_161 = add_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_123 = torch.ops.aten.split_with_sizes.default(add_162, [28, 28], 1)
        getitem_378: "f32[8, 28, 28, 28]" = split_with_sizes_123[0]
        getitem_379: "f32[8, 28, 28, 28]" = split_with_sizes_123[1];  split_with_sizes_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_186: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_378, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_378 = arg96_1 = None
        convolution_187: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_379, arg97_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_379 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_50: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_186, convolution_187], 1);  convolution_186 = convolution_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_584: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_585: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        sub_73: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_50, unsqueeze_585);  cat_50 = unsqueeze_585 = None
        add_163: "f32[336]" = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_73: "f32[336]" = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_73: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_291: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_586: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
        unsqueeze_587: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        mul_292: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_589: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_293: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_589);  mul_292 = unsqueeze_589 = None
        unsqueeze_590: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_591: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_164: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_591);  mul_293 = unsqueeze_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_72: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_164)
        mul_294: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_164, sigmoid_72);  add_164 = sigmoid_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_125 = torch.ops.aten.split_with_sizes.default(mul_294, [168, 168], 1)
        getitem_382: "f32[8, 168, 28, 28]" = split_with_sizes_125[0];  split_with_sizes_125 = None
        split_with_sizes_126 = torch.ops.aten.split_with_sizes.default(mul_294, [168, 168], 1);  mul_294 = None
        getitem_385: "f32[8, 168, 28, 28]" = split_with_sizes_126[1];  split_with_sizes_126 = None
        convolution_188: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_382, arg102_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_382 = arg102_1 = None
        convolution_189: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_385, arg103_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_385 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_51: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_188, convolution_189], 1);  convolution_188 = convolution_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_592: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_593: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        sub_74: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_51, unsqueeze_593);  cat_51 = unsqueeze_593 = None
        add_165: "f32[336]" = torch.ops.aten.add.Tensor(arg105_1, 1e-05);  arg105_1 = None
        sqrt_74: "f32[336]" = torch.ops.aten.sqrt.default(add_165);  add_165 = None
        reciprocal_74: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_295: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_594: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_595: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        mul_296: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_597: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_297: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_597);  mul_296 = unsqueeze_597 = None
        unsqueeze_598: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_599: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_166: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_297, unsqueeze_599);  mul_297 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_73: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_166)
        mul_298: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_73);  add_166 = sigmoid_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_19: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_298, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_190: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_19, arg108_1, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg108_1 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_74: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_190)
        mul_299: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_190, sigmoid_74);  convolution_190 = sigmoid_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_191: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_299, arg110_1, arg111_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_299 = arg110_1 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_75: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_191);  convolution_191 = None
        mul_300: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_298, sigmoid_75);  mul_298 = sigmoid_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_127 = torch.ops.aten.split_with_sizes.default(mul_300, [168, 168], 1);  mul_300 = None
        getitem_386: "f32[8, 168, 28, 28]" = split_with_sizes_127[0]
        getitem_387: "f32[8, 168, 28, 28]" = split_with_sizes_127[1];  split_with_sizes_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_192: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_386, arg112_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_386 = arg112_1 = None
        convolution_193: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_387, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_387 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_52: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_192, convolution_193], 1);  convolution_192 = convolution_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_600: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_601: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        sub_75: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_52, unsqueeze_601);  cat_52 = unsqueeze_601 = None
        add_167: "f32[56]" = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_75: "f32[56]" = torch.ops.aten.sqrt.default(add_167);  add_167 = None
        reciprocal_75: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_301: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_602: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_301, -1);  mul_301 = None
        unsqueeze_603: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        mul_302: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_605: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_303: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_605);  mul_302 = unsqueeze_605 = None
        unsqueeze_606: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_607: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_168: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_303, unsqueeze_607);  mul_303 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_169: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_168, add_162);  add_168 = add_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_128 = torch.ops.aten.split_with_sizes.default(add_169, [28, 28], 1)
        getitem_388: "f32[8, 28, 28, 28]" = split_with_sizes_128[0]
        getitem_389: "f32[8, 28, 28, 28]" = split_with_sizes_128[1];  split_with_sizes_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_194: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_388, arg118_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_388 = arg118_1 = None
        convolution_195: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_389, arg119_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_389 = arg119_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_53: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_194, convolution_195], 1);  convolution_194 = convolution_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_608: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_609: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        sub_76: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_53, unsqueeze_609);  cat_53 = unsqueeze_609 = None
        add_170: "f32[336]" = torch.ops.aten.add.Tensor(arg121_1, 1e-05);  arg121_1 = None
        sqrt_76: "f32[336]" = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_76: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_304: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_610: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_304, -1);  mul_304 = None
        unsqueeze_611: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        mul_305: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_613: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_306: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_305, unsqueeze_613);  mul_305 = unsqueeze_613 = None
        unsqueeze_614: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_615: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_171: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_306, unsqueeze_615);  mul_306 = unsqueeze_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_76: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_171)
        mul_307: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_171, sigmoid_76);  add_171 = sigmoid_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_130 = torch.ops.aten.split_with_sizes.default(mul_307, [168, 168], 1)
        getitem_392: "f32[8, 168, 28, 28]" = split_with_sizes_130[0];  split_with_sizes_130 = None
        split_with_sizes_131 = torch.ops.aten.split_with_sizes.default(mul_307, [168, 168], 1);  mul_307 = None
        getitem_395: "f32[8, 168, 28, 28]" = split_with_sizes_131[1];  split_with_sizes_131 = None
        convolution_196: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_392, arg124_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168);  getitem_392 = arg124_1 = None
        convolution_197: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_395, arg125_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168);  getitem_395 = arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_54: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_196, convolution_197], 1);  convolution_196 = convolution_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_616: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_617: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        sub_77: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_54, unsqueeze_617);  cat_54 = unsqueeze_617 = None
        add_172: "f32[336]" = torch.ops.aten.add.Tensor(arg127_1, 1e-05);  arg127_1 = None
        sqrt_77: "f32[336]" = torch.ops.aten.sqrt.default(add_172);  add_172 = None
        reciprocal_77: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_308: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_618: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_619: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        mul_309: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_621: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_310: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_621);  mul_309 = unsqueeze_621 = None
        unsqueeze_622: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_623: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_173: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_623);  mul_310 = unsqueeze_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_77: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_173)
        mul_311: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_173, sigmoid_77);  add_173 = sigmoid_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_20: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_311, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_198: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_20, arg130_1, arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg130_1 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_78: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_198)
        mul_312: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_198, sigmoid_78);  convolution_198 = sigmoid_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_199: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_312, arg132_1, arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_312 = arg132_1 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_79: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_199);  convolution_199 = None
        mul_313: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_311, sigmoid_79);  mul_311 = sigmoid_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_132 = torch.ops.aten.split_with_sizes.default(mul_313, [168, 168], 1);  mul_313 = None
        getitem_396: "f32[8, 168, 28, 28]" = split_with_sizes_132[0]
        getitem_397: "f32[8, 168, 28, 28]" = split_with_sizes_132[1];  split_with_sizes_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_200: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_396, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_396 = arg134_1 = None
        convolution_201: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_397, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_397 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_55: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_200, convolution_201], 1);  convolution_200 = convolution_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_624: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_625: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        sub_78: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_55, unsqueeze_625);  cat_55 = unsqueeze_625 = None
        add_174: "f32[56]" = torch.ops.aten.add.Tensor(arg137_1, 1e-05);  arg137_1 = None
        sqrt_78: "f32[56]" = torch.ops.aten.sqrt.default(add_174);  add_174 = None
        reciprocal_78: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_314: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_626: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_314, -1);  mul_314 = None
        unsqueeze_627: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        mul_315: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_629: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_316: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_629);  mul_315 = unsqueeze_629 = None
        unsqueeze_630: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_631: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_175: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_316, unsqueeze_631);  mul_316 = unsqueeze_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_176: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_175, add_169);  add_175 = add_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_202: "f32[8, 336, 28, 28]" = torch.ops.aten.convolution.default(add_176, arg140_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_176 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_632: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_633: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        sub_79: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_633);  convolution_202 = unsqueeze_633 = None
        add_177: "f32[336]" = torch.ops.aten.add.Tensor(arg142_1, 1e-05);  arg142_1 = None
        sqrt_79: "f32[336]" = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_79: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_317: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_634: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_317, -1);  mul_317 = None
        unsqueeze_635: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        mul_318: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_637: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_319: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_637);  mul_318 = unsqueeze_637 = None
        unsqueeze_638: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_639: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_178: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_319, unsqueeze_639);  mul_319 = unsqueeze_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_80: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_178)
        mul_320: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_178, sigmoid_80);  add_178 = sigmoid_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_134 = torch.ops.aten.split_with_sizes.default(mul_320, [112, 112, 112], 1)
        getitem_401: "f32[8, 112, 28, 28]" = split_with_sizes_134[0];  split_with_sizes_134 = None
        split_with_sizes_135 = torch.ops.aten.split_with_sizes.default(mul_320, [112, 112, 112], 1)
        getitem_405: "f32[8, 112, 28, 28]" = split_with_sizes_135[1];  split_with_sizes_135 = None
        split_with_sizes_136 = torch.ops.aten.split_with_sizes.default(mul_320, [112, 112, 112], 1);  mul_320 = None
        getitem_409: "f32[8, 112, 28, 28]" = split_with_sizes_136[2];  split_with_sizes_136 = None
        convolution_203: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(getitem_401, arg145_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 112);  getitem_401 = arg145_1 = None
        convolution_204: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(getitem_405, arg146_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 112);  getitem_405 = arg146_1 = None
        convolution_205: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(getitem_409, arg147_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 112);  getitem_409 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_56: "f32[8, 336, 14, 14]" = torch.ops.aten.cat.default([convolution_203, convolution_204, convolution_205], 1);  convolution_203 = convolution_204 = convolution_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_640: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_641: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        sub_80: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(cat_56, unsqueeze_641);  cat_56 = unsqueeze_641 = None
        add_179: "f32[336]" = torch.ops.aten.add.Tensor(arg149_1, 1e-05);  arg149_1 = None
        sqrt_80: "f32[336]" = torch.ops.aten.sqrt.default(add_179);  add_179 = None
        reciprocal_80: "f32[336]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_321: "f32[336]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_642: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
        unsqueeze_643: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        mul_322: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_645: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_323: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_645);  mul_322 = unsqueeze_645 = None
        unsqueeze_646: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_647: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_180: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_647);  mul_323 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_81: "f32[8, 336, 14, 14]" = torch.ops.aten.sigmoid.default(add_180)
        mul_324: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, sigmoid_81);  add_180 = sigmoid_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_21: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_324, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_206: "f32[8, 14, 1, 1]" = torch.ops.aten.convolution.default(mean_21, arg152_1, arg153_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg152_1 = arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_82: "f32[8, 14, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_206)
        mul_325: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_206, sigmoid_82);  convolution_206 = sigmoid_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_207: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_325, arg154_1, arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_325 = arg154_1 = arg155_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_83: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_207);  convolution_207 = None
        mul_326: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_324, sigmoid_83);  mul_324 = sigmoid_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_208: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(mul_326, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_326 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_648: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_649: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        sub_81: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_649);  convolution_208 = unsqueeze_649 = None
        add_181: "f32[104]" = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_81: "f32[104]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_81: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_327: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_650: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_651: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        mul_328: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_653: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_329: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_653);  mul_328 = unsqueeze_653 = None
        unsqueeze_654: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_655: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_182: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_655);  mul_329 = unsqueeze_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_137 = torch.ops.aten.split_with_sizes.default(add_182, [52, 52], 1)
        getitem_410: "f32[8, 52, 14, 14]" = split_with_sizes_137[0]
        getitem_411: "f32[8, 52, 14, 14]" = split_with_sizes_137[1];  split_with_sizes_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_209: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_410, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_410 = arg161_1 = None
        convolution_210: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_411, arg162_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_411 = arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_57: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_209, convolution_210], 1);  convolution_209 = convolution_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_656: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_657: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        sub_82: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_57, unsqueeze_657);  cat_57 = unsqueeze_657 = None
        add_183: "f32[624]" = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_82: "f32[624]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_82: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_330: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_658: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_659: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        mul_331: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_661: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_332: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_661);  mul_331 = unsqueeze_661 = None
        unsqueeze_662: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_663: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_184: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_663);  mul_332 = unsqueeze_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_84: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_184)
        mul_333: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_184, sigmoid_84);  add_184 = sigmoid_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_139 = torch.ops.aten.split_with_sizes.default(mul_333, [156, 156, 156, 156], 1)
        getitem_416: "f32[8, 156, 14, 14]" = split_with_sizes_139[0];  split_with_sizes_139 = None
        split_with_sizes_140 = torch.ops.aten.split_with_sizes.default(mul_333, [156, 156, 156, 156], 1)
        getitem_421: "f32[8, 156, 14, 14]" = split_with_sizes_140[1];  split_with_sizes_140 = None
        split_with_sizes_141 = torch.ops.aten.split_with_sizes.default(mul_333, [156, 156, 156, 156], 1)
        getitem_426: "f32[8, 156, 14, 14]" = split_with_sizes_141[2];  split_with_sizes_141 = None
        split_with_sizes_142 = torch.ops.aten.split_with_sizes.default(mul_333, [156, 156, 156, 156], 1);  mul_333 = None
        getitem_431: "f32[8, 156, 14, 14]" = split_with_sizes_142[3];  split_with_sizes_142 = None
        convolution_211: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_416, arg167_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_416 = arg167_1 = None
        convolution_212: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_421, arg168_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_421 = arg168_1 = None
        convolution_213: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_426, arg169_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_426 = arg169_1 = None
        convolution_214: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_431, arg170_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_431 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_58: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_211, convolution_212, convolution_213, convolution_214], 1);  convolution_211 = convolution_212 = convolution_213 = convolution_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_664: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_665: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        sub_83: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_58, unsqueeze_665);  cat_58 = unsqueeze_665 = None
        add_185: "f32[624]" = torch.ops.aten.add.Tensor(arg172_1, 1e-05);  arg172_1 = None
        sqrt_83: "f32[624]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_83: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_334: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_666: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_334, -1);  mul_334 = None
        unsqueeze_667: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        mul_335: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_669: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_336: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_335, unsqueeze_669);  mul_335 = unsqueeze_669 = None
        unsqueeze_670: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_671: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_186: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_336, unsqueeze_671);  mul_336 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_85: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_186)
        mul_337: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_186, sigmoid_85);  add_186 = sigmoid_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_22: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_337, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_215: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_22, arg175_1, arg176_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg175_1 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_86: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_215)
        mul_338: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_215, sigmoid_86);  convolution_215 = sigmoid_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_216: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_338, arg177_1, arg178_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_338 = arg177_1 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_87: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_216);  convolution_216 = None
        mul_339: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_337, sigmoid_87);  mul_337 = sigmoid_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_143 = torch.ops.aten.split_with_sizes.default(mul_339, [312, 312], 1);  mul_339 = None
        getitem_432: "f32[8, 312, 14, 14]" = split_with_sizes_143[0]
        getitem_433: "f32[8, 312, 14, 14]" = split_with_sizes_143[1];  split_with_sizes_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_217: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_432, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_432 = arg179_1 = None
        convolution_218: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_433, arg180_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_433 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_59: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_217, convolution_218], 1);  convolution_217 = convolution_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_672: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_673: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        sub_84: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_59, unsqueeze_673);  cat_59 = unsqueeze_673 = None
        add_187: "f32[104]" = torch.ops.aten.add.Tensor(arg182_1, 1e-05);  arg182_1 = None
        sqrt_84: "f32[104]" = torch.ops.aten.sqrt.default(add_187);  add_187 = None
        reciprocal_84: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_340: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_674: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_340, -1);  mul_340 = None
        unsqueeze_675: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        mul_341: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_677: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_342: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_341, unsqueeze_677);  mul_341 = unsqueeze_677 = None
        unsqueeze_678: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_679: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_188: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_679);  mul_342 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_189: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_188, add_182);  add_188 = add_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_144 = torch.ops.aten.split_with_sizes.default(add_189, [52, 52], 1)
        getitem_434: "f32[8, 52, 14, 14]" = split_with_sizes_144[0]
        getitem_435: "f32[8, 52, 14, 14]" = split_with_sizes_144[1];  split_with_sizes_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_219: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_434, arg185_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_434 = arg185_1 = None
        convolution_220: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_435, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_435 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_60: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_219, convolution_220], 1);  convolution_219 = convolution_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_680: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_681: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        sub_85: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_60, unsqueeze_681);  cat_60 = unsqueeze_681 = None
        add_190: "f32[624]" = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_85: "f32[624]" = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_85: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_343: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_682: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_343, -1);  mul_343 = None
        unsqueeze_683: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        mul_344: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_685: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_345: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_344, unsqueeze_685);  mul_344 = unsqueeze_685 = None
        unsqueeze_686: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_687: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_191: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_345, unsqueeze_687);  mul_345 = unsqueeze_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_88: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_191)
        mul_346: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_191, sigmoid_88);  add_191 = sigmoid_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_146 = torch.ops.aten.split_with_sizes.default(mul_346, [156, 156, 156, 156], 1)
        getitem_440: "f32[8, 156, 14, 14]" = split_with_sizes_146[0];  split_with_sizes_146 = None
        split_with_sizes_147 = torch.ops.aten.split_with_sizes.default(mul_346, [156, 156, 156, 156], 1)
        getitem_445: "f32[8, 156, 14, 14]" = split_with_sizes_147[1];  split_with_sizes_147 = None
        split_with_sizes_148 = torch.ops.aten.split_with_sizes.default(mul_346, [156, 156, 156, 156], 1)
        getitem_450: "f32[8, 156, 14, 14]" = split_with_sizes_148[2];  split_with_sizes_148 = None
        split_with_sizes_149 = torch.ops.aten.split_with_sizes.default(mul_346, [156, 156, 156, 156], 1);  mul_346 = None
        getitem_455: "f32[8, 156, 14, 14]" = split_with_sizes_149[3];  split_with_sizes_149 = None
        convolution_221: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_440, arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_440 = arg191_1 = None
        convolution_222: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_445, arg192_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_445 = arg192_1 = None
        convolution_223: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_450, arg193_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_450 = arg193_1 = None
        convolution_224: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_455, arg194_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_455 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_61: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_221, convolution_222, convolution_223, convolution_224], 1);  convolution_221 = convolution_222 = convolution_223 = convolution_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_688: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_689: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        sub_86: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_61, unsqueeze_689);  cat_61 = unsqueeze_689 = None
        add_192: "f32[624]" = torch.ops.aten.add.Tensor(arg196_1, 1e-05);  arg196_1 = None
        sqrt_86: "f32[624]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_86: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_347: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_690: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_347, -1);  mul_347 = None
        unsqueeze_691: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        mul_348: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_693: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_349: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_348, unsqueeze_693);  mul_348 = unsqueeze_693 = None
        unsqueeze_694: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_695: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_193: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_695);  mul_349 = unsqueeze_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_89: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_193)
        mul_350: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_193, sigmoid_89);  add_193 = sigmoid_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_23: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_350, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_225: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_23, arg199_1, arg200_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg199_1 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_90: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_225)
        mul_351: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_225, sigmoid_90);  convolution_225 = sigmoid_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_226: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_351, arg201_1, arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_351 = arg201_1 = arg202_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_91: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_226);  convolution_226 = None
        mul_352: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_350, sigmoid_91);  mul_350 = sigmoid_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_150 = torch.ops.aten.split_with_sizes.default(mul_352, [312, 312], 1);  mul_352 = None
        getitem_456: "f32[8, 312, 14, 14]" = split_with_sizes_150[0]
        getitem_457: "f32[8, 312, 14, 14]" = split_with_sizes_150[1];  split_with_sizes_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_227: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_456, arg203_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_456 = arg203_1 = None
        convolution_228: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_457, arg204_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_457 = arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_62: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_227, convolution_228], 1);  convolution_227 = convolution_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_696: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_697: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        sub_87: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_62, unsqueeze_697);  cat_62 = unsqueeze_697 = None
        add_194: "f32[104]" = torch.ops.aten.add.Tensor(arg206_1, 1e-05);  arg206_1 = None
        sqrt_87: "f32[104]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_87: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_353: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_698: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_353, -1);  mul_353 = None
        unsqueeze_699: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        mul_354: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_701: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_355: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_354, unsqueeze_701);  mul_354 = unsqueeze_701 = None
        unsqueeze_702: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_703: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_195: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_355, unsqueeze_703);  mul_355 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_196: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_195, add_189);  add_195 = add_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_151 = torch.ops.aten.split_with_sizes.default(add_196, [52, 52], 1)
        getitem_458: "f32[8, 52, 14, 14]" = split_with_sizes_151[0]
        getitem_459: "f32[8, 52, 14, 14]" = split_with_sizes_151[1];  split_with_sizes_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_229: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_458, arg209_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_458 = arg209_1 = None
        convolution_230: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_459, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_459 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_63: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_229, convolution_230], 1);  convolution_229 = convolution_230 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_704: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_705: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        sub_88: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_63, unsqueeze_705);  cat_63 = unsqueeze_705 = None
        add_197: "f32[624]" = torch.ops.aten.add.Tensor(arg212_1, 1e-05);  arg212_1 = None
        sqrt_88: "f32[624]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_88: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_356: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_706: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_356, -1);  mul_356 = None
        unsqueeze_707: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        mul_357: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_709: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_358: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_709);  mul_357 = unsqueeze_709 = None
        unsqueeze_710: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_711: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_198: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_711);  mul_358 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_92: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_198)
        mul_359: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_198, sigmoid_92);  add_198 = sigmoid_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_153 = torch.ops.aten.split_with_sizes.default(mul_359, [156, 156, 156, 156], 1)
        getitem_464: "f32[8, 156, 14, 14]" = split_with_sizes_153[0];  split_with_sizes_153 = None
        split_with_sizes_154 = torch.ops.aten.split_with_sizes.default(mul_359, [156, 156, 156, 156], 1)
        getitem_469: "f32[8, 156, 14, 14]" = split_with_sizes_154[1];  split_with_sizes_154 = None
        split_with_sizes_155 = torch.ops.aten.split_with_sizes.default(mul_359, [156, 156, 156, 156], 1)
        getitem_474: "f32[8, 156, 14, 14]" = split_with_sizes_155[2];  split_with_sizes_155 = None
        split_with_sizes_156 = torch.ops.aten.split_with_sizes.default(mul_359, [156, 156, 156, 156], 1);  mul_359 = None
        getitem_479: "f32[8, 156, 14, 14]" = split_with_sizes_156[3];  split_with_sizes_156 = None
        convolution_231: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_464, arg215_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156);  getitem_464 = arg215_1 = None
        convolution_232: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_469, arg216_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156);  getitem_469 = arg216_1 = None
        convolution_233: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_474, arg217_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156);  getitem_474 = arg217_1 = None
        convolution_234: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_479, arg218_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156);  getitem_479 = arg218_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_64: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_231, convolution_232, convolution_233, convolution_234], 1);  convolution_231 = convolution_232 = convolution_233 = convolution_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_712: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_713: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        sub_89: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_64, unsqueeze_713);  cat_64 = unsqueeze_713 = None
        add_199: "f32[624]" = torch.ops.aten.add.Tensor(arg220_1, 1e-05);  arg220_1 = None
        sqrt_89: "f32[624]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_89: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_360: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_714: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_715: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        mul_361: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_717: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_362: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_717);  mul_361 = unsqueeze_717 = None
        unsqueeze_718: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_719: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_200: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_719);  mul_362 = unsqueeze_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_93: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_200)
        mul_363: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_200, sigmoid_93);  add_200 = sigmoid_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_24: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_363, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_235: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_24, arg223_1, arg224_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg223_1 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_94: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_235)
        mul_364: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_235, sigmoid_94);  convolution_235 = sigmoid_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_236: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_364, arg225_1, arg226_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_364 = arg225_1 = arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_95: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_236);  convolution_236 = None
        mul_365: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_363, sigmoid_95);  mul_363 = sigmoid_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_157 = torch.ops.aten.split_with_sizes.default(mul_365, [312, 312], 1);  mul_365 = None
        getitem_480: "f32[8, 312, 14, 14]" = split_with_sizes_157[0]
        getitem_481: "f32[8, 312, 14, 14]" = split_with_sizes_157[1];  split_with_sizes_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_237: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_480, arg227_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_480 = arg227_1 = None
        convolution_238: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_481, arg228_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_481 = arg228_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_65: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_237, convolution_238], 1);  convolution_237 = convolution_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_720: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_721: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        sub_90: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_65, unsqueeze_721);  cat_65 = unsqueeze_721 = None
        add_201: "f32[104]" = torch.ops.aten.add.Tensor(arg230_1, 1e-05);  arg230_1 = None
        sqrt_90: "f32[104]" = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_90: "f32[104]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_366: "f32[104]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_722: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_723: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        mul_367: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_725: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_368: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_725);  mul_367 = unsqueeze_725 = None
        unsqueeze_726: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_727: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_202: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_368, unsqueeze_727);  mul_368 = unsqueeze_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_203: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_202, add_196);  add_202 = add_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_239: "f32[8, 624, 14, 14]" = torch.ops.aten.convolution.default(add_203, arg233_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_203 = arg233_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_728: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_729: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        sub_91: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_239, unsqueeze_729);  convolution_239 = unsqueeze_729 = None
        add_204: "f32[624]" = torch.ops.aten.add.Tensor(arg235_1, 1e-05);  arg235_1 = None
        sqrt_91: "f32[624]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_91: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_369: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_730: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_731: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        mul_370: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_733: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_371: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_733);  mul_370 = unsqueeze_733 = None
        unsqueeze_734: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_735: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_205: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_735);  mul_371 = unsqueeze_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_96: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_205)
        mul_372: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_205, sigmoid_96);  add_205 = sigmoid_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_240: "f32[8, 624, 14, 14]" = torch.ops.aten.convolution.default(mul_372, arg238_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 624);  mul_372 = arg238_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_736: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_737: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        sub_92: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_737);  convolution_240 = unsqueeze_737 = None
        add_206: "f32[624]" = torch.ops.aten.add.Tensor(arg240_1, 1e-05);  arg240_1 = None
        sqrt_92: "f32[624]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_92: "f32[624]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_373: "f32[624]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_738: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(mul_373, -1);  mul_373 = None
        unsqueeze_739: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        mul_374: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_741: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_375: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_374, unsqueeze_741);  mul_374 = unsqueeze_741 = None
        unsqueeze_742: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_743: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_207: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_375, unsqueeze_743);  mul_375 = unsqueeze_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_97: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_207)
        mul_376: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_207, sigmoid_97);  add_207 = sigmoid_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_25: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_376, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_241: "f32[8, 52, 1, 1]" = torch.ops.aten.convolution.default(mean_25, arg243_1, arg244_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg243_1 = arg244_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_98: "f32[8, 52, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_241)
        mul_377: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_241, sigmoid_98);  convolution_241 = sigmoid_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_242: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_377, arg245_1, arg246_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_377 = arg245_1 = arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_99: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_242);  convolution_242 = None
        mul_378: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_376, sigmoid_99);  mul_376 = sigmoid_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_243: "f32[8, 160, 14, 14]" = torch.ops.aten.convolution.default(mul_378, arg247_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_378 = arg247_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_744: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_745: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        sub_93: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_243, unsqueeze_745);  convolution_243 = unsqueeze_745 = None
        add_208: "f32[160]" = torch.ops.aten.add.Tensor(arg249_1, 1e-05);  arg249_1 = None
        sqrt_93: "f32[160]" = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_93: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_379: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_746: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_379, -1);  mul_379 = None
        unsqueeze_747: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        mul_380: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_749: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_381: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_380, unsqueeze_749);  mul_380 = unsqueeze_749 = None
        unsqueeze_750: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_751: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_209: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_381, unsqueeze_751);  mul_381 = unsqueeze_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_158 = torch.ops.aten.split_with_sizes.default(add_209, [80, 80], 1)
        getitem_482: "f32[8, 80, 14, 14]" = split_with_sizes_158[0]
        getitem_483: "f32[8, 80, 14, 14]" = split_with_sizes_158[1];  split_with_sizes_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_244: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_482, arg252_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_482 = arg252_1 = None
        convolution_245: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_483, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_483 = arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_66: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_244, convolution_245], 1);  convolution_244 = convolution_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_752: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_753: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        sub_94: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_66, unsqueeze_753);  cat_66 = unsqueeze_753 = None
        add_210: "f32[480]" = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_94: "f32[480]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_94: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_382: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_754: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_382, -1);  mul_382 = None
        unsqueeze_755: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        mul_383: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_757: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_384: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_383, unsqueeze_757);  mul_383 = unsqueeze_757 = None
        unsqueeze_758: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_759: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_211: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_759);  mul_384 = unsqueeze_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_100: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_211)
        mul_385: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_211, sigmoid_100);  add_211 = sigmoid_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_160 = torch.ops.aten.split_with_sizes.default(mul_385, [120, 120, 120, 120], 1)
        getitem_488: "f32[8, 120, 14, 14]" = split_with_sizes_160[0];  split_with_sizes_160 = None
        split_with_sizes_161 = torch.ops.aten.split_with_sizes.default(mul_385, [120, 120, 120, 120], 1)
        getitem_493: "f32[8, 120, 14, 14]" = split_with_sizes_161[1];  split_with_sizes_161 = None
        split_with_sizes_162 = torch.ops.aten.split_with_sizes.default(mul_385, [120, 120, 120, 120], 1)
        getitem_498: "f32[8, 120, 14, 14]" = split_with_sizes_162[2];  split_with_sizes_162 = None
        split_with_sizes_163 = torch.ops.aten.split_with_sizes.default(mul_385, [120, 120, 120, 120], 1);  mul_385 = None
        getitem_503: "f32[8, 120, 14, 14]" = split_with_sizes_163[3];  split_with_sizes_163 = None
        convolution_246: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_488, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_488 = arg258_1 = None
        convolution_247: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_493, arg259_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_493 = arg259_1 = None
        convolution_248: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_498, arg260_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_498 = arg260_1 = None
        convolution_249: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_503, arg261_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_503 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_67: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_246, convolution_247, convolution_248, convolution_249], 1);  convolution_246 = convolution_247 = convolution_248 = convolution_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_760: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_761: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        sub_95: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_67, unsqueeze_761);  cat_67 = unsqueeze_761 = None
        add_212: "f32[480]" = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_95: "f32[480]" = torch.ops.aten.sqrt.default(add_212);  add_212 = None
        reciprocal_95: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_386: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_762: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_386, -1);  mul_386 = None
        unsqueeze_763: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        mul_387: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_765: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_388: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_387, unsqueeze_765);  mul_387 = unsqueeze_765 = None
        unsqueeze_766: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_767: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_213: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_388, unsqueeze_767);  mul_388 = unsqueeze_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_101: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_213)
        mul_389: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_213, sigmoid_101);  add_213 = sigmoid_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_26: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_389, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_250: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_26, arg266_1, arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg266_1 = arg267_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_102: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_250)
        mul_390: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_250, sigmoid_102);  convolution_250 = sigmoid_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_251: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_390, arg268_1, arg269_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_390 = arg268_1 = arg269_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_103: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_251);  convolution_251 = None
        mul_391: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_389, sigmoid_103);  mul_389 = sigmoid_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_164 = torch.ops.aten.split_with_sizes.default(mul_391, [240, 240], 1);  mul_391 = None
        getitem_504: "f32[8, 240, 14, 14]" = split_with_sizes_164[0]
        getitem_505: "f32[8, 240, 14, 14]" = split_with_sizes_164[1];  split_with_sizes_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_252: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_504, arg270_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_504 = arg270_1 = None
        convolution_253: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_505, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_505 = arg271_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_68: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_252, convolution_253], 1);  convolution_252 = convolution_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_768: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_769: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        sub_96: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_68, unsqueeze_769);  cat_68 = unsqueeze_769 = None
        add_214: "f32[160]" = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_96: "f32[160]" = torch.ops.aten.sqrt.default(add_214);  add_214 = None
        reciprocal_96: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_392: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_770: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_392, -1);  mul_392 = None
        unsqueeze_771: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        mul_393: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_773: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_394: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_393, unsqueeze_773);  mul_393 = unsqueeze_773 = None
        unsqueeze_774: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_775: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_215: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_394, unsqueeze_775);  mul_394 = unsqueeze_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_216: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_215, add_209);  add_215 = add_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_165 = torch.ops.aten.split_with_sizes.default(add_216, [80, 80], 1)
        getitem_506: "f32[8, 80, 14, 14]" = split_with_sizes_165[0]
        getitem_507: "f32[8, 80, 14, 14]" = split_with_sizes_165[1];  split_with_sizes_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_254: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_506, arg276_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_506 = arg276_1 = None
        convolution_255: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_507, arg277_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_507 = arg277_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_69: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_254, convolution_255], 1);  convolution_254 = convolution_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_776: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_777: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        sub_97: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_69, unsqueeze_777);  cat_69 = unsqueeze_777 = None
        add_217: "f32[480]" = torch.ops.aten.add.Tensor(arg279_1, 1e-05);  arg279_1 = None
        sqrt_97: "f32[480]" = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_97: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_395: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_778: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_395, -1);  mul_395 = None
        unsqueeze_779: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        mul_396: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_781: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_397: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_396, unsqueeze_781);  mul_396 = unsqueeze_781 = None
        unsqueeze_782: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg281_1, -1);  arg281_1 = None
        unsqueeze_783: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_218: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_397, unsqueeze_783);  mul_397 = unsqueeze_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_104: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_218)
        mul_398: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_218, sigmoid_104);  add_218 = sigmoid_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_167 = torch.ops.aten.split_with_sizes.default(mul_398, [120, 120, 120, 120], 1)
        getitem_512: "f32[8, 120, 14, 14]" = split_with_sizes_167[0];  split_with_sizes_167 = None
        split_with_sizes_168 = torch.ops.aten.split_with_sizes.default(mul_398, [120, 120, 120, 120], 1)
        getitem_517: "f32[8, 120, 14, 14]" = split_with_sizes_168[1];  split_with_sizes_168 = None
        split_with_sizes_169 = torch.ops.aten.split_with_sizes.default(mul_398, [120, 120, 120, 120], 1)
        getitem_522: "f32[8, 120, 14, 14]" = split_with_sizes_169[2];  split_with_sizes_169 = None
        split_with_sizes_170 = torch.ops.aten.split_with_sizes.default(mul_398, [120, 120, 120, 120], 1);  mul_398 = None
        getitem_527: "f32[8, 120, 14, 14]" = split_with_sizes_170[3];  split_with_sizes_170 = None
        convolution_256: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_512, arg282_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_512 = arg282_1 = None
        convolution_257: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_517, arg283_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_517 = arg283_1 = None
        convolution_258: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_522, arg284_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_522 = arg284_1 = None
        convolution_259: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_527, arg285_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_527 = arg285_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_70: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_256, convolution_257, convolution_258, convolution_259], 1);  convolution_256 = convolution_257 = convolution_258 = convolution_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_784: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg286_1, -1);  arg286_1 = None
        unsqueeze_785: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        sub_98: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_70, unsqueeze_785);  cat_70 = unsqueeze_785 = None
        add_219: "f32[480]" = torch.ops.aten.add.Tensor(arg287_1, 1e-05);  arg287_1 = None
        sqrt_98: "f32[480]" = torch.ops.aten.sqrt.default(add_219);  add_219 = None
        reciprocal_98: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_399: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_786: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_399, -1);  mul_399 = None
        unsqueeze_787: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        mul_400: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_789: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_401: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_789);  mul_400 = unsqueeze_789 = None
        unsqueeze_790: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_791: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_220: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_401, unsqueeze_791);  mul_401 = unsqueeze_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_105: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_220)
        mul_402: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_220, sigmoid_105);  add_220 = sigmoid_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_27: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_402, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_260: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_27, arg290_1, arg291_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_27 = arg290_1 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_106: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_260)
        mul_403: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_260, sigmoid_106);  convolution_260 = sigmoid_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_261: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_403, arg292_1, arg293_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_403 = arg292_1 = arg293_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_107: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_261);  convolution_261 = None
        mul_404: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_402, sigmoid_107);  mul_402 = sigmoid_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_171 = torch.ops.aten.split_with_sizes.default(mul_404, [240, 240], 1);  mul_404 = None
        getitem_528: "f32[8, 240, 14, 14]" = split_with_sizes_171[0]
        getitem_529: "f32[8, 240, 14, 14]" = split_with_sizes_171[1];  split_with_sizes_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_262: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_528, arg294_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_528 = arg294_1 = None
        convolution_263: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_529, arg295_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_529 = arg295_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_71: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_262, convolution_263], 1);  convolution_262 = convolution_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_792: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg296_1, -1);  arg296_1 = None
        unsqueeze_793: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        sub_99: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_71, unsqueeze_793);  cat_71 = unsqueeze_793 = None
        add_221: "f32[160]" = torch.ops.aten.add.Tensor(arg297_1, 1e-05);  arg297_1 = None
        sqrt_99: "f32[160]" = torch.ops.aten.sqrt.default(add_221);  add_221 = None
        reciprocal_99: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_405: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_794: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_405, -1);  mul_405 = None
        unsqueeze_795: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        mul_406: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_797: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_407: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_797);  mul_406 = unsqueeze_797 = None
        unsqueeze_798: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_799: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_222: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_407, unsqueeze_799);  mul_407 = unsqueeze_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_223: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_222, add_216);  add_222 = add_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_172 = torch.ops.aten.split_with_sizes.default(add_223, [80, 80], 1)
        getitem_530: "f32[8, 80, 14, 14]" = split_with_sizes_172[0]
        getitem_531: "f32[8, 80, 14, 14]" = split_with_sizes_172[1];  split_with_sizes_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_264: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_530, arg300_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_530 = arg300_1 = None
        convolution_265: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_531, arg301_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_531 = arg301_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_72: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_264, convolution_265], 1);  convolution_264 = convolution_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_800: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_801: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        sub_100: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_72, unsqueeze_801);  cat_72 = unsqueeze_801 = None
        add_224: "f32[480]" = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_100: "f32[480]" = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_100: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_408: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_802: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
        unsqueeze_803: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        mul_409: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_805: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_410: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_805);  mul_409 = unsqueeze_805 = None
        unsqueeze_806: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_807: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_225: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_410, unsqueeze_807);  mul_410 = unsqueeze_807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_108: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_225)
        mul_411: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_225, sigmoid_108);  add_225 = sigmoid_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_174 = torch.ops.aten.split_with_sizes.default(mul_411, [120, 120, 120, 120], 1)
        getitem_536: "f32[8, 120, 14, 14]" = split_with_sizes_174[0];  split_with_sizes_174 = None
        split_with_sizes_175 = torch.ops.aten.split_with_sizes.default(mul_411, [120, 120, 120, 120], 1)
        getitem_541: "f32[8, 120, 14, 14]" = split_with_sizes_175[1];  split_with_sizes_175 = None
        split_with_sizes_176 = torch.ops.aten.split_with_sizes.default(mul_411, [120, 120, 120, 120], 1)
        getitem_546: "f32[8, 120, 14, 14]" = split_with_sizes_176[2];  split_with_sizes_176 = None
        split_with_sizes_177 = torch.ops.aten.split_with_sizes.default(mul_411, [120, 120, 120, 120], 1);  mul_411 = None
        getitem_551: "f32[8, 120, 14, 14]" = split_with_sizes_177[3];  split_with_sizes_177 = None
        convolution_266: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_536, arg306_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  getitem_536 = arg306_1 = None
        convolution_267: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_541, arg307_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_541 = arg307_1 = None
        convolution_268: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_546, arg308_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_546 = arg308_1 = None
        convolution_269: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_551, arg309_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120);  getitem_551 = arg309_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_73: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_266, convolution_267, convolution_268, convolution_269], 1);  convolution_266 = convolution_267 = convolution_268 = convolution_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_808: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_809: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        sub_101: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_73, unsqueeze_809);  cat_73 = unsqueeze_809 = None
        add_226: "f32[480]" = torch.ops.aten.add.Tensor(arg311_1, 1e-05);  arg311_1 = None
        sqrt_101: "f32[480]" = torch.ops.aten.sqrt.default(add_226);  add_226 = None
        reciprocal_101: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_412: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_810: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_412, -1);  mul_412 = None
        unsqueeze_811: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        mul_413: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_813: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_414: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_813);  mul_413 = unsqueeze_813 = None
        unsqueeze_814: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_815: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_227: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_414, unsqueeze_815);  mul_414 = unsqueeze_815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_109: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_227)
        mul_415: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_227, sigmoid_109);  add_227 = sigmoid_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_28: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_415, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_270: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_28, arg314_1, arg315_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_28 = arg314_1 = arg315_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_110: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_270)
        mul_416: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_270, sigmoid_110);  convolution_270 = sigmoid_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_271: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_416, arg316_1, arg317_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_416 = arg316_1 = arg317_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_111: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_271);  convolution_271 = None
        mul_417: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_415, sigmoid_111);  mul_415 = sigmoid_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_178 = torch.ops.aten.split_with_sizes.default(mul_417, [240, 240], 1);  mul_417 = None
        getitem_552: "f32[8, 240, 14, 14]" = split_with_sizes_178[0]
        getitem_553: "f32[8, 240, 14, 14]" = split_with_sizes_178[1];  split_with_sizes_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_272: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_552, arg318_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_552 = arg318_1 = None
        convolution_273: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_553, arg319_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_553 = arg319_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_74: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_272, convolution_273], 1);  convolution_272 = convolution_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_816: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_817: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        sub_102: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_74, unsqueeze_817);  cat_74 = unsqueeze_817 = None
        add_228: "f32[160]" = torch.ops.aten.add.Tensor(arg321_1, 1e-05);  arg321_1 = None
        sqrt_102: "f32[160]" = torch.ops.aten.sqrt.default(add_228);  add_228 = None
        reciprocal_102: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_418: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_818: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_418, -1);  mul_418 = None
        unsqueeze_819: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        mul_419: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_821: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_420: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_821);  mul_419 = unsqueeze_821 = None
        unsqueeze_822: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_823: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_229: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_420, unsqueeze_823);  mul_420 = unsqueeze_823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_230: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_229, add_223);  add_229 = add_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_274: "f32[8, 960, 14, 14]" = torch.ops.aten.convolution.default(add_230, arg324_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_230 = arg324_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_824: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_825: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        sub_103: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_274, unsqueeze_825);  convolution_274 = unsqueeze_825 = None
        add_231: "f32[960]" = torch.ops.aten.add.Tensor(arg326_1, 1e-05);  arg326_1 = None
        sqrt_103: "f32[960]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_103: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_421: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_826: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_421, -1);  mul_421 = None
        unsqueeze_827: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        mul_422: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_829: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_423: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(mul_422, unsqueeze_829);  mul_422 = unsqueeze_829 = None
        unsqueeze_830: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_831: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_232: "f32[8, 960, 14, 14]" = torch.ops.aten.add.Tensor(mul_423, unsqueeze_831);  mul_423 = unsqueeze_831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_112: "f32[8, 960, 14, 14]" = torch.ops.aten.sigmoid.default(add_232)
        mul_424: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(add_232, sigmoid_112);  add_232 = sigmoid_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_180 = torch.ops.aten.split_with_sizes.default(mul_424, [240, 240, 240, 240], 1)
        getitem_558: "f32[8, 240, 14, 14]" = split_with_sizes_180[0];  split_with_sizes_180 = None
        split_with_sizes_181 = torch.ops.aten.split_with_sizes.default(mul_424, [240, 240, 240, 240], 1)
        getitem_563: "f32[8, 240, 14, 14]" = split_with_sizes_181[1];  split_with_sizes_181 = None
        split_with_sizes_182 = torch.ops.aten.split_with_sizes.default(mul_424, [240, 240, 240, 240], 1)
        getitem_568: "f32[8, 240, 14, 14]" = split_with_sizes_182[2];  split_with_sizes_182 = None
        split_with_sizes_183 = torch.ops.aten.split_with_sizes.default(mul_424, [240, 240, 240, 240], 1);  mul_424 = None
        getitem_573: "f32[8, 240, 14, 14]" = split_with_sizes_183[3];  split_with_sizes_183 = None
        convolution_275: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(getitem_558, arg329_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  getitem_558 = arg329_1 = None
        convolution_276: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(getitem_563, arg330_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 240);  getitem_563 = arg330_1 = None
        convolution_277: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(getitem_568, arg331_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 240);  getitem_568 = arg331_1 = None
        convolution_278: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(getitem_573, arg332_1, None, [2, 2], [4, 4], [1, 1], False, [0, 0], 240);  getitem_573 = arg332_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_75: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([convolution_275, convolution_276, convolution_277, convolution_278], 1);  convolution_275 = convolution_276 = convolution_277 = convolution_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_832: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_833: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        sub_104: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_75, unsqueeze_833);  cat_75 = unsqueeze_833 = None
        add_233: "f32[960]" = torch.ops.aten.add.Tensor(arg334_1, 1e-05);  arg334_1 = None
        sqrt_104: "f32[960]" = torch.ops.aten.sqrt.default(add_233);  add_233 = None
        reciprocal_104: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_425: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_834: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_425, -1);  mul_425 = None
        unsqueeze_835: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        mul_426: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_837: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_427: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_837);  mul_426 = unsqueeze_837 = None
        unsqueeze_838: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_839: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_234: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_427, unsqueeze_839);  mul_427 = unsqueeze_839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_113: "f32[8, 960, 7, 7]" = torch.ops.aten.sigmoid.default(add_234)
        mul_428: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_234, sigmoid_113);  add_234 = sigmoid_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_29: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(mul_428, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_279: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_29, arg337_1, arg338_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_29 = arg337_1 = arg338_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_114: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_279)
        mul_429: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_279, sigmoid_114);  convolution_279 = sigmoid_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_280: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(mul_429, arg339_1, arg340_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_429 = arg339_1 = arg340_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_115: "f32[8, 960, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_280);  convolution_280 = None
        mul_430: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_428, sigmoid_115);  mul_428 = sigmoid_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_281: "f32[8, 264, 7, 7]" = torch.ops.aten.convolution.default(mul_430, arg341_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_430 = arg341_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_840: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_841: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        sub_105: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_281, unsqueeze_841);  convolution_281 = unsqueeze_841 = None
        add_235: "f32[264]" = torch.ops.aten.add.Tensor(arg343_1, 1e-05);  arg343_1 = None
        sqrt_105: "f32[264]" = torch.ops.aten.sqrt.default(add_235);  add_235 = None
        reciprocal_105: "f32[264]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_431: "f32[264]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_842: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(mul_431, -1);  mul_431 = None
        unsqueeze_843: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        mul_432: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_845: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_433: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_432, unsqueeze_845);  mul_432 = unsqueeze_845 = None
        unsqueeze_846: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_847: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_236: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_847);  mul_433 = unsqueeze_847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_282: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_236, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg346_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_848: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_849: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        sub_106: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_282, unsqueeze_849);  convolution_282 = unsqueeze_849 = None
        add_237: "f32[1584]" = torch.ops.aten.add.Tensor(arg348_1, 1e-05);  arg348_1 = None
        sqrt_106: "f32[1584]" = torch.ops.aten.sqrt.default(add_237);  add_237 = None
        reciprocal_106: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_434: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_850: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_434, -1);  mul_434 = None
        unsqueeze_851: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        mul_435: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_853: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_436: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_435, unsqueeze_853);  mul_435 = unsqueeze_853 = None
        unsqueeze_854: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_855: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_238: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_436, unsqueeze_855);  mul_436 = unsqueeze_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_116: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_238)
        mul_437: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_238, sigmoid_116);  add_238 = sigmoid_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_185 = torch.ops.aten.split_with_sizes.default(mul_437, [396, 396, 396, 396], 1)
        getitem_578: "f32[8, 396, 7, 7]" = split_with_sizes_185[0];  split_with_sizes_185 = None
        split_with_sizes_186 = torch.ops.aten.split_with_sizes.default(mul_437, [396, 396, 396, 396], 1)
        getitem_583: "f32[8, 396, 7, 7]" = split_with_sizes_186[1];  split_with_sizes_186 = None
        split_with_sizes_187 = torch.ops.aten.split_with_sizes.default(mul_437, [396, 396, 396, 396], 1)
        getitem_588: "f32[8, 396, 7, 7]" = split_with_sizes_187[2];  split_with_sizes_187 = None
        split_with_sizes_188 = torch.ops.aten.split_with_sizes.default(mul_437, [396, 396, 396, 396], 1);  mul_437 = None
        getitem_593: "f32[8, 396, 7, 7]" = split_with_sizes_188[3];  split_with_sizes_188 = None
        convolution_283: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_578, arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_578 = arg351_1 = None
        convolution_284: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_583, arg352_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_583 = arg352_1 = None
        convolution_285: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_588, arg353_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_588 = arg353_1 = None
        convolution_286: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_593, arg354_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_593 = arg354_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_76: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_283, convolution_284, convolution_285, convolution_286], 1);  convolution_283 = convolution_284 = convolution_285 = convolution_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_856: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_857: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        sub_107: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_76, unsqueeze_857);  cat_76 = unsqueeze_857 = None
        add_239: "f32[1584]" = torch.ops.aten.add.Tensor(arg356_1, 1e-05);  arg356_1 = None
        sqrt_107: "f32[1584]" = torch.ops.aten.sqrt.default(add_239);  add_239 = None
        reciprocal_107: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_438: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_858: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
        unsqueeze_859: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        mul_439: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg357_1, -1);  arg357_1 = None
        unsqueeze_861: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_440: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_861);  mul_439 = unsqueeze_861 = None
        unsqueeze_862: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
        unsqueeze_863: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_240: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_863);  mul_440 = unsqueeze_863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_117: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_240)
        mul_441: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_240, sigmoid_117);  add_240 = sigmoid_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_30: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_441, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_287: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_30, arg359_1, arg360_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_30 = arg359_1 = arg360_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_118: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_287)
        mul_442: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_287, sigmoid_118);  convolution_287 = sigmoid_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_288: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_442, arg361_1, arg362_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_442 = arg361_1 = arg362_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_119: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_288);  convolution_288 = None
        mul_443: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_441, sigmoid_119);  mul_441 = sigmoid_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_189 = torch.ops.aten.split_with_sizes.default(mul_443, [792, 792], 1);  mul_443 = None
        getitem_594: "f32[8, 792, 7, 7]" = split_with_sizes_189[0]
        getitem_595: "f32[8, 792, 7, 7]" = split_with_sizes_189[1];  split_with_sizes_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_289: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_594, arg363_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_594 = arg363_1 = None
        convolution_290: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_595, arg364_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_595 = arg364_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_77: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_289, convolution_290], 1);  convolution_289 = convolution_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_864: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_865: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        sub_108: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_77, unsqueeze_865);  cat_77 = unsqueeze_865 = None
        add_241: "f32[264]" = torch.ops.aten.add.Tensor(arg366_1, 1e-05);  arg366_1 = None
        sqrt_108: "f32[264]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_108: "f32[264]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_444: "f32[264]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_866: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_867: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        mul_445: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg367_1, -1);  arg367_1 = None
        unsqueeze_869: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_446: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_869);  mul_445 = unsqueeze_869 = None
        unsqueeze_870: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg368_1, -1);  arg368_1 = None
        unsqueeze_871: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_242: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_871);  mul_446 = unsqueeze_871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_243: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_242, add_236);  add_242 = add_236 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_291: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_243, arg369_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg369_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_872: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
        unsqueeze_873: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        sub_109: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_291, unsqueeze_873);  convolution_291 = unsqueeze_873 = None
        add_244: "f32[1584]" = torch.ops.aten.add.Tensor(arg371_1, 1e-05);  arg371_1 = None
        sqrt_109: "f32[1584]" = torch.ops.aten.sqrt.default(add_244);  add_244 = None
        reciprocal_109: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_447: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_874: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_875: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        mul_448: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg372_1, -1);  arg372_1 = None
        unsqueeze_877: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_449: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_877);  mul_448 = unsqueeze_877 = None
        unsqueeze_878: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg373_1, -1);  arg373_1 = None
        unsqueeze_879: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_245: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_449, unsqueeze_879);  mul_449 = unsqueeze_879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_120: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_245)
        mul_450: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_245, sigmoid_120);  add_245 = sigmoid_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_191 = torch.ops.aten.split_with_sizes.default(mul_450, [396, 396, 396, 396], 1)
        getitem_600: "f32[8, 396, 7, 7]" = split_with_sizes_191[0];  split_with_sizes_191 = None
        split_with_sizes_192 = torch.ops.aten.split_with_sizes.default(mul_450, [396, 396, 396, 396], 1)
        getitem_605: "f32[8, 396, 7, 7]" = split_with_sizes_192[1];  split_with_sizes_192 = None
        split_with_sizes_193 = torch.ops.aten.split_with_sizes.default(mul_450, [396, 396, 396, 396], 1)
        getitem_610: "f32[8, 396, 7, 7]" = split_with_sizes_193[2];  split_with_sizes_193 = None
        split_with_sizes_194 = torch.ops.aten.split_with_sizes.default(mul_450, [396, 396, 396, 396], 1);  mul_450 = None
        getitem_615: "f32[8, 396, 7, 7]" = split_with_sizes_194[3];  split_with_sizes_194 = None
        convolution_292: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_600, arg374_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_600 = arg374_1 = None
        convolution_293: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_605, arg375_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_605 = arg375_1 = None
        convolution_294: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_610, arg376_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_610 = arg376_1 = None
        convolution_295: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_615, arg377_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_615 = arg377_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_78: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_292, convolution_293, convolution_294, convolution_295], 1);  convolution_292 = convolution_293 = convolution_294 = convolution_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_880: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg378_1, -1);  arg378_1 = None
        unsqueeze_881: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        sub_110: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_78, unsqueeze_881);  cat_78 = unsqueeze_881 = None
        add_246: "f32[1584]" = torch.ops.aten.add.Tensor(arg379_1, 1e-05);  arg379_1 = None
        sqrt_110: "f32[1584]" = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_110: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_451: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_882: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_451, -1);  mul_451 = None
        unsqueeze_883: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        mul_452: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_885: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_453: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_452, unsqueeze_885);  mul_452 = unsqueeze_885 = None
        unsqueeze_886: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg381_1, -1);  arg381_1 = None
        unsqueeze_887: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_247: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_453, unsqueeze_887);  mul_453 = unsqueeze_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_121: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_247)
        mul_454: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_247, sigmoid_121);  add_247 = sigmoid_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_31: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_454, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_296: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_31, arg382_1, arg383_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_31 = arg382_1 = arg383_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_122: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_296)
        mul_455: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_296, sigmoid_122);  convolution_296 = sigmoid_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_297: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_455, arg384_1, arg385_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_455 = arg384_1 = arg385_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_123: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_297);  convolution_297 = None
        mul_456: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_454, sigmoid_123);  mul_454 = sigmoid_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_195 = torch.ops.aten.split_with_sizes.default(mul_456, [792, 792], 1);  mul_456 = None
        getitem_616: "f32[8, 792, 7, 7]" = split_with_sizes_195[0]
        getitem_617: "f32[8, 792, 7, 7]" = split_with_sizes_195[1];  split_with_sizes_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_298: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_616, arg386_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_616 = arg386_1 = None
        convolution_299: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_617, arg387_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_617 = arg387_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_79: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_298, convolution_299], 1);  convolution_298 = convolution_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_888: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_889: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        sub_111: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_79, unsqueeze_889);  cat_79 = unsqueeze_889 = None
        add_248: "f32[264]" = torch.ops.aten.add.Tensor(arg389_1, 1e-05);  arg389_1 = None
        sqrt_111: "f32[264]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_111: "f32[264]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_457: "f32[264]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_890: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(mul_457, -1);  mul_457 = None
        unsqueeze_891: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        mul_458: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_893: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_459: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_458, unsqueeze_893);  mul_458 = unsqueeze_893 = None
        unsqueeze_894: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg391_1, -1);  arg391_1 = None
        unsqueeze_895: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_249: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_459, unsqueeze_895);  mul_459 = unsqueeze_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_250: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_249, add_243);  add_249 = add_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_300: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_250, arg392_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg392_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_896: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg393_1, -1);  arg393_1 = None
        unsqueeze_897: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        sub_112: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_300, unsqueeze_897);  convolution_300 = unsqueeze_897 = None
        add_251: "f32[1584]" = torch.ops.aten.add.Tensor(arg394_1, 1e-05);  arg394_1 = None
        sqrt_112: "f32[1584]" = torch.ops.aten.sqrt.default(add_251);  add_251 = None
        reciprocal_112: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_460: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_898: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_460, -1);  mul_460 = None
        unsqueeze_899: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        mul_461: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_901: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_462: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_461, unsqueeze_901);  mul_461 = unsqueeze_901 = None
        unsqueeze_902: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg396_1, -1);  arg396_1 = None
        unsqueeze_903: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_252: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_903);  mul_462 = unsqueeze_903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_124: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_252)
        mul_463: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_252, sigmoid_124);  add_252 = sigmoid_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        split_with_sizes_197 = torch.ops.aten.split_with_sizes.default(mul_463, [396, 396, 396, 396], 1)
        getitem_622: "f32[8, 396, 7, 7]" = split_with_sizes_197[0];  split_with_sizes_197 = None
        split_with_sizes_198 = torch.ops.aten.split_with_sizes.default(mul_463, [396, 396, 396, 396], 1)
        getitem_627: "f32[8, 396, 7, 7]" = split_with_sizes_198[1];  split_with_sizes_198 = None
        split_with_sizes_199 = torch.ops.aten.split_with_sizes.default(mul_463, [396, 396, 396, 396], 1)
        getitem_632: "f32[8, 396, 7, 7]" = split_with_sizes_199[2];  split_with_sizes_199 = None
        split_with_sizes_200 = torch.ops.aten.split_with_sizes.default(mul_463, [396, 396, 396, 396], 1);  mul_463 = None
        getitem_637: "f32[8, 396, 7, 7]" = split_with_sizes_200[3];  split_with_sizes_200 = None
        convolution_301: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_622, arg397_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396);  getitem_622 = arg397_1 = None
        convolution_302: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_627, arg398_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396);  getitem_627 = arg398_1 = None
        convolution_303: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_632, arg399_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396);  getitem_632 = arg399_1 = None
        convolution_304: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_637, arg400_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396);  getitem_637 = arg400_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_80: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_301, convolution_302, convolution_303, convolution_304], 1);  convolution_301 = convolution_302 = convolution_303 = convolution_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_904: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg401_1, -1);  arg401_1 = None
        unsqueeze_905: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        sub_113: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_80, unsqueeze_905);  cat_80 = unsqueeze_905 = None
        add_253: "f32[1584]" = torch.ops.aten.add.Tensor(arg402_1, 1e-05);  arg402_1 = None
        sqrt_113: "f32[1584]" = torch.ops.aten.sqrt.default(add_253);  add_253 = None
        reciprocal_113: "f32[1584]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_464: "f32[1584]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_906: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(mul_464, -1);  mul_464 = None
        unsqueeze_907: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        mul_465: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg403_1, -1);  arg403_1 = None
        unsqueeze_909: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_466: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_465, unsqueeze_909);  mul_465 = unsqueeze_909 = None
        unsqueeze_910: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_911: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_254: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_466, unsqueeze_911);  mul_466 = unsqueeze_911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_125: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_254)
        mul_467: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_254, sigmoid_125);  add_254 = sigmoid_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_32: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_467, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_305: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_32, arg405_1, arg406_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_32 = arg405_1 = arg406_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_126: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_305)
        mul_468: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_305, sigmoid_126);  convolution_305 = sigmoid_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_306: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_468, arg407_1, arg408_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_468 = arg407_1 = arg408_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_127: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_306);  convolution_306 = None
        mul_469: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_467, sigmoid_127);  mul_467 = sigmoid_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:48 in forward, code: x_split = torch.split(x, self.splits, 1)
        split_with_sizes_201 = torch.ops.aten.split_with_sizes.default(mul_469, [792, 792], 1);  mul_469 = None
        getitem_638: "f32[8, 792, 7, 7]" = split_with_sizes_201[0]
        getitem_639: "f32[8, 792, 7, 7]" = split_with_sizes_201[1];  split_with_sizes_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:49 in forward, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        convolution_307: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_638, arg409_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_638 = arg409_1 = None
        convolution_308: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_639, arg410_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_639 = arg410_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mixed_conv2d.py:50 in forward, code: x = torch.cat(x_out, 1)
        cat_81: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_307, convolution_308], 1);  convolution_307 = convolution_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_912: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg411_1, -1);  arg411_1 = None
        unsqueeze_913: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        sub_114: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_81, unsqueeze_913);  cat_81 = unsqueeze_913 = None
        add_255: "f32[264]" = torch.ops.aten.add.Tensor(arg412_1, 1e-05);  arg412_1 = None
        sqrt_114: "f32[264]" = torch.ops.aten.sqrt.default(add_255);  add_255 = None
        reciprocal_114: "f32[264]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_470: "f32[264]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_914: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(mul_470, -1);  mul_470 = None
        unsqueeze_915: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        mul_471: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg413_1, -1);  arg413_1 = None
        unsqueeze_917: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_472: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_471, unsqueeze_917);  mul_471 = unsqueeze_917 = None
        unsqueeze_918: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_919: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_256: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_472, unsqueeze_919);  mul_472 = unsqueeze_919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_257: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_256, add_250);  add_256 = add_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/efficientnet.py:258 in forward_features, code: x = self.conv_head(x)
        convolution_309: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(add_257, arg415_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_257 = arg415_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_920: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg416_1, -1);  arg416_1 = None
        unsqueeze_921: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        sub_115: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_309, unsqueeze_921);  convolution_309 = unsqueeze_921 = None
        add_258: "f32[1536]" = torch.ops.aten.add.Tensor(arg417_1, 1e-05);  arg417_1 = None
        sqrt_115: "f32[1536]" = torch.ops.aten.sqrt.default(add_258);  add_258 = None
        reciprocal_115: "f32[1536]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_473: "f32[1536]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_922: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(mul_473, -1);  mul_473 = None
        unsqueeze_923: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        mul_474: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg418_1, -1);  arg418_1 = None
        unsqueeze_925: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_475: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_474, unsqueeze_925);  mul_474 = unsqueeze_925 = None
        unsqueeze_926: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_927: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_259: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_927);  mul_475 = unsqueeze_927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_13: "f32[8, 1536, 7, 7]" = torch.ops.aten.relu.default(add_259);  add_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_33: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(relu_13, [-1, -2], True);  relu_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 1536]" = torch.ops.aten.reshape.default(mean_33, [8, 1536]);  mean_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/efficientnet.py:266 in forward_head, code: return x if pre_logits else self.classifier(x)
        permute_1: "f32[1536, 1000]" = torch.ops.aten.permute.default(arg420_1, [1, 0]);  arg420_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg421_1, view_1, permute_1);  arg421_1 = view_1 = permute_1 = None
        return (addmm_1,)
        