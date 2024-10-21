class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[32, 1, 3, 3]", arg7_1: "f32[32]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[16, 32, 1, 1]", arg12_1: "f32[16]", arg13_1: "f32[16]", arg14_1: "f32[16]", arg15_1: "f32[16]", arg16_1: "f32[96, 16, 1, 1]", arg17_1: "f32[96]", arg18_1: "f32[96]", arg19_1: "f32[96]", arg20_1: "f32[96]", arg21_1: "f32[96, 1, 3, 3]", arg22_1: "f32[96]", arg23_1: "f32[96]", arg24_1: "f32[96]", arg25_1: "f32[96]", arg26_1: "f32[27, 96, 1, 1]", arg27_1: "f32[27]", arg28_1: "f32[27]", arg29_1: "f32[27]", arg30_1: "f32[27]", arg31_1: "f32[162, 27, 1, 1]", arg32_1: "f32[162]", arg33_1: "f32[162]", arg34_1: "f32[162]", arg35_1: "f32[162]", arg36_1: "f32[162, 1, 3, 3]", arg37_1: "f32[162]", arg38_1: "f32[162]", arg39_1: "f32[162]", arg40_1: "f32[162]", arg41_1: "f32[38, 162, 1, 1]", arg42_1: "f32[38]", arg43_1: "f32[38]", arg44_1: "f32[38]", arg45_1: "f32[38]", arg46_1: "f32[228, 38, 1, 1]", arg47_1: "f32[228]", arg48_1: "f32[228]", arg49_1: "f32[228]", arg50_1: "f32[228]", arg51_1: "f32[228, 1, 3, 3]", arg52_1: "f32[228]", arg53_1: "f32[228]", arg54_1: "f32[228]", arg55_1: "f32[228]", arg56_1: "f32[19, 228, 1, 1]", arg57_1: "f32[19]", arg58_1: "f32[19]", arg59_1: "f32[19]", arg60_1: "f32[19]", arg61_1: "f32[19]", arg62_1: "f32[228, 19, 1, 1]", arg63_1: "f32[228]", arg64_1: "f32[50, 228, 1, 1]", arg65_1: "f32[50]", arg66_1: "f32[50]", arg67_1: "f32[50]", arg68_1: "f32[50]", arg69_1: "f32[300, 50, 1, 1]", arg70_1: "f32[300]", arg71_1: "f32[300]", arg72_1: "f32[300]", arg73_1: "f32[300]", arg74_1: "f32[300, 1, 3, 3]", arg75_1: "f32[300]", arg76_1: "f32[300]", arg77_1: "f32[300]", arg78_1: "f32[300]", arg79_1: "f32[25, 300, 1, 1]", arg80_1: "f32[25]", arg81_1: "f32[25]", arg82_1: "f32[25]", arg83_1: "f32[25]", arg84_1: "f32[25]", arg85_1: "f32[300, 25, 1, 1]", arg86_1: "f32[300]", arg87_1: "f32[61, 300, 1, 1]", arg88_1: "f32[61]", arg89_1: "f32[61]", arg90_1: "f32[61]", arg91_1: "f32[61]", arg92_1: "f32[366, 61, 1, 1]", arg93_1: "f32[366]", arg94_1: "f32[366]", arg95_1: "f32[366]", arg96_1: "f32[366]", arg97_1: "f32[366, 1, 3, 3]", arg98_1: "f32[366]", arg99_1: "f32[366]", arg100_1: "f32[366]", arg101_1: "f32[366]", arg102_1: "f32[30, 366, 1, 1]", arg103_1: "f32[30]", arg104_1: "f32[30]", arg105_1: "f32[30]", arg106_1: "f32[30]", arg107_1: "f32[30]", arg108_1: "f32[366, 30, 1, 1]", arg109_1: "f32[366]", arg110_1: "f32[72, 366, 1, 1]", arg111_1: "f32[72]", arg112_1: "f32[72]", arg113_1: "f32[72]", arg114_1: "f32[72]", arg115_1: "f32[432, 72, 1, 1]", arg116_1: "f32[432]", arg117_1: "f32[432]", arg118_1: "f32[432]", arg119_1: "f32[432]", arg120_1: "f32[432, 1, 3, 3]", arg121_1: "f32[432]", arg122_1: "f32[432]", arg123_1: "f32[432]", arg124_1: "f32[432]", arg125_1: "f32[36, 432, 1, 1]", arg126_1: "f32[36]", arg127_1: "f32[36]", arg128_1: "f32[36]", arg129_1: "f32[36]", arg130_1: "f32[36]", arg131_1: "f32[432, 36, 1, 1]", arg132_1: "f32[432]", arg133_1: "f32[84, 432, 1, 1]", arg134_1: "f32[84]", arg135_1: "f32[84]", arg136_1: "f32[84]", arg137_1: "f32[84]", arg138_1: "f32[504, 84, 1, 1]", arg139_1: "f32[504]", arg140_1: "f32[504]", arg141_1: "f32[504]", arg142_1: "f32[504]", arg143_1: "f32[504, 1, 3, 3]", arg144_1: "f32[504]", arg145_1: "f32[504]", arg146_1: "f32[504]", arg147_1: "f32[504]", arg148_1: "f32[42, 504, 1, 1]", arg149_1: "f32[42]", arg150_1: "f32[42]", arg151_1: "f32[42]", arg152_1: "f32[42]", arg153_1: "f32[42]", arg154_1: "f32[504, 42, 1, 1]", arg155_1: "f32[504]", arg156_1: "f32[95, 504, 1, 1]", arg157_1: "f32[95]", arg158_1: "f32[95]", arg159_1: "f32[95]", arg160_1: "f32[95]", arg161_1: "f32[570, 95, 1, 1]", arg162_1: "f32[570]", arg163_1: "f32[570]", arg164_1: "f32[570]", arg165_1: "f32[570]", arg166_1: "f32[570, 1, 3, 3]", arg167_1: "f32[570]", arg168_1: "f32[570]", arg169_1: "f32[570]", arg170_1: "f32[570]", arg171_1: "f32[47, 570, 1, 1]", arg172_1: "f32[47]", arg173_1: "f32[47]", arg174_1: "f32[47]", arg175_1: "f32[47]", arg176_1: "f32[47]", arg177_1: "f32[570, 47, 1, 1]", arg178_1: "f32[570]", arg179_1: "f32[106, 570, 1, 1]", arg180_1: "f32[106]", arg181_1: "f32[106]", arg182_1: "f32[106]", arg183_1: "f32[106]", arg184_1: "f32[636, 106, 1, 1]", arg185_1: "f32[636]", arg186_1: "f32[636]", arg187_1: "f32[636]", arg188_1: "f32[636]", arg189_1: "f32[636, 1, 3, 3]", arg190_1: "f32[636]", arg191_1: "f32[636]", arg192_1: "f32[636]", arg193_1: "f32[636]", arg194_1: "f32[53, 636, 1, 1]", arg195_1: "f32[53]", arg196_1: "f32[53]", arg197_1: "f32[53]", arg198_1: "f32[53]", arg199_1: "f32[53]", arg200_1: "f32[636, 53, 1, 1]", arg201_1: "f32[636]", arg202_1: "f32[117, 636, 1, 1]", arg203_1: "f32[117]", arg204_1: "f32[117]", arg205_1: "f32[117]", arg206_1: "f32[117]", arg207_1: "f32[702, 117, 1, 1]", arg208_1: "f32[702]", arg209_1: "f32[702]", arg210_1: "f32[702]", arg211_1: "f32[702]", arg212_1: "f32[702, 1, 3, 3]", arg213_1: "f32[702]", arg214_1: "f32[702]", arg215_1: "f32[702]", arg216_1: "f32[702]", arg217_1: "f32[58, 702, 1, 1]", arg218_1: "f32[58]", arg219_1: "f32[58]", arg220_1: "f32[58]", arg221_1: "f32[58]", arg222_1: "f32[58]", arg223_1: "f32[702, 58, 1, 1]", arg224_1: "f32[702]", arg225_1: "f32[128, 702, 1, 1]", arg226_1: "f32[128]", arg227_1: "f32[128]", arg228_1: "f32[128]", arg229_1: "f32[128]", arg230_1: "f32[768, 128, 1, 1]", arg231_1: "f32[768]", arg232_1: "f32[768]", arg233_1: "f32[768]", arg234_1: "f32[768]", arg235_1: "f32[768, 1, 3, 3]", arg236_1: "f32[768]", arg237_1: "f32[768]", arg238_1: "f32[768]", arg239_1: "f32[768]", arg240_1: "f32[64, 768, 1, 1]", arg241_1: "f32[64]", arg242_1: "f32[64]", arg243_1: "f32[64]", arg244_1: "f32[64]", arg245_1: "f32[64]", arg246_1: "f32[768, 64, 1, 1]", arg247_1: "f32[768]", arg248_1: "f32[140, 768, 1, 1]", arg249_1: "f32[140]", arg250_1: "f32[140]", arg251_1: "f32[140]", arg252_1: "f32[140]", arg253_1: "f32[840, 140, 1, 1]", arg254_1: "f32[840]", arg255_1: "f32[840]", arg256_1: "f32[840]", arg257_1: "f32[840]", arg258_1: "f32[840, 1, 3, 3]", arg259_1: "f32[840]", arg260_1: "f32[840]", arg261_1: "f32[840]", arg262_1: "f32[840]", arg263_1: "f32[70, 840, 1, 1]", arg264_1: "f32[70]", arg265_1: "f32[70]", arg266_1: "f32[70]", arg267_1: "f32[70]", arg268_1: "f32[70]", arg269_1: "f32[840, 70, 1, 1]", arg270_1: "f32[840]", arg271_1: "f32[151, 840, 1, 1]", arg272_1: "f32[151]", arg273_1: "f32[151]", arg274_1: "f32[151]", arg275_1: "f32[151]", arg276_1: "f32[906, 151, 1, 1]", arg277_1: "f32[906]", arg278_1: "f32[906]", arg279_1: "f32[906]", arg280_1: "f32[906]", arg281_1: "f32[906, 1, 3, 3]", arg282_1: "f32[906]", arg283_1: "f32[906]", arg284_1: "f32[906]", arg285_1: "f32[906]", arg286_1: "f32[75, 906, 1, 1]", arg287_1: "f32[75]", arg288_1: "f32[75]", arg289_1: "f32[75]", arg290_1: "f32[75]", arg291_1: "f32[75]", arg292_1: "f32[906, 75, 1, 1]", arg293_1: "f32[906]", arg294_1: "f32[162, 906, 1, 1]", arg295_1: "f32[162]", arg296_1: "f32[162]", arg297_1: "f32[162]", arg298_1: "f32[162]", arg299_1: "f32[972, 162, 1, 1]", arg300_1: "f32[972]", arg301_1: "f32[972]", arg302_1: "f32[972]", arg303_1: "f32[972]", arg304_1: "f32[972, 1, 3, 3]", arg305_1: "f32[972]", arg306_1: "f32[972]", arg307_1: "f32[972]", arg308_1: "f32[972]", arg309_1: "f32[81, 972, 1, 1]", arg310_1: "f32[81]", arg311_1: "f32[81]", arg312_1: "f32[81]", arg313_1: "f32[81]", arg314_1: "f32[81]", arg315_1: "f32[972, 81, 1, 1]", arg316_1: "f32[972]", arg317_1: "f32[174, 972, 1, 1]", arg318_1: "f32[174]", arg319_1: "f32[174]", arg320_1: "f32[174]", arg321_1: "f32[174]", arg322_1: "f32[1044, 174, 1, 1]", arg323_1: "f32[1044]", arg324_1: "f32[1044]", arg325_1: "f32[1044]", arg326_1: "f32[1044]", arg327_1: "f32[1044, 1, 3, 3]", arg328_1: "f32[1044]", arg329_1: "f32[1044]", arg330_1: "f32[1044]", arg331_1: "f32[1044]", arg332_1: "f32[87, 1044, 1, 1]", arg333_1: "f32[87]", arg334_1: "f32[87]", arg335_1: "f32[87]", arg336_1: "f32[87]", arg337_1: "f32[87]", arg338_1: "f32[1044, 87, 1, 1]", arg339_1: "f32[1044]", arg340_1: "f32[185, 1044, 1, 1]", arg341_1: "f32[185]", arg342_1: "f32[185]", arg343_1: "f32[185]", arg344_1: "f32[185]", arg345_1: "f32[1280, 185, 1, 1]", arg346_1: "f32[1280]", arg347_1: "f32[1280]", arg348_1: "f32[1280]", arg349_1: "f32[1280]", arg350_1: "f32[1000, 1280]", arg351_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_75: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_135: "f32[32]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_62: "f32[32]" = torch.ops.aten.sqrt.default(add_135);  add_135 = None
        reciprocal_62: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_216: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_497: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_499: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_497);  convolution_75 = unsqueeze_497 = None
        mul_217: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_501: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_218: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_501);  mul_217 = unsqueeze_501 = None
        unsqueeze_502: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_503: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_136: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_503);  mul_218 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_30: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_136)
        mul_219: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_136, sigmoid_30);  add_136 = sigmoid_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_76: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(mul_219, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  mul_219 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_137: "f32[32]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_63: "f32[32]" = torch.ops.aten.sqrt.default(add_137);  add_137 = None
        reciprocal_63: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_220: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_505: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_220, -1);  mul_220 = None
        unsqueeze_507: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_505);  convolution_76 = unsqueeze_505 = None
        mul_221: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_509: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_222: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_509);  mul_221 = unsqueeze_509 = None
        unsqueeze_510: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_511: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_138: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_511);  mul_222 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_16: "f32[8, 32, 112, 112]" = torch.ops.aten.clamp_min.default(add_138, 0.0);  add_138 = None
        clamp_max_16: "f32[8, 32, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6.0);  clamp_min_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_77: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(clamp_max_16, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_16 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_139: "f32[16]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_64: "f32[16]" = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_64: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_223: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_513: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_223, -1);  mul_223 = None
        unsqueeze_515: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_513);  convolution_77 = unsqueeze_513 = None
        mul_224: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_517: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_225: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_517);  mul_224 = unsqueeze_517 = None
        unsqueeze_518: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_519: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_140: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_225, unsqueeze_519);  mul_225 = unsqueeze_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_78: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(add_140, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_140 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_141: "f32[96]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_65: "f32[96]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_65: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_226: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_521: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_226, -1);  mul_226 = None
        unsqueeze_523: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_521);  convolution_78 = unsqueeze_521 = None
        mul_227: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_525: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_228: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_525);  mul_227 = unsqueeze_525 = None
        unsqueeze_526: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_527: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_142: "f32[8, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_228, unsqueeze_527);  mul_228 = unsqueeze_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_31: "f32[8, 96, 112, 112]" = torch.ops.aten.sigmoid.default(add_142)
        mul_229: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(add_142, sigmoid_31);  add_142 = sigmoid_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_79: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(mul_229, arg21_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96);  mul_229 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_143: "f32[96]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_66: "f32[96]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_66: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_230: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_529: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_230, -1);  mul_230 = None
        unsqueeze_531: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_529);  convolution_79 = unsqueeze_529 = None
        mul_231: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_533: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_232: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_533);  mul_231 = unsqueeze_533 = None
        unsqueeze_534: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_535: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_144: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_535);  mul_232 = unsqueeze_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_17: "f32[8, 96, 56, 56]" = torch.ops.aten.clamp_min.default(add_144, 0.0);  add_144 = None
        clamp_max_17: "f32[8, 96, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6.0);  clamp_min_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_80: "f32[8, 27, 56, 56]" = torch.ops.aten.convolution.default(clamp_max_17, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_17 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_145: "f32[27]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_67: "f32[27]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_67: "f32[27]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_233: "f32[27]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536: "f32[27, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_537: "f32[27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538: "f32[27, 1]" = torch.ops.aten.unsqueeze.default(mul_233, -1);  mul_233 = None
        unsqueeze_539: "f32[27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67: "f32[8, 27, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_537);  convolution_80 = unsqueeze_537 = None
        mul_234: "f32[8, 27, 56, 56]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[27, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_541: "f32[27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_235: "f32[8, 27, 56, 56]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_541);  mul_234 = unsqueeze_541 = None
        unsqueeze_542: "f32[27, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_543: "f32[27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_146: "f32[8, 27, 56, 56]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_543);  mul_235 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_81: "f32[8, 162, 56, 56]" = torch.ops.aten.convolution.default(add_146, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_147: "f32[162]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_68: "f32[162]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_68: "f32[162]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_236: "f32[162]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_545: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(mul_236, -1);  mul_236 = None
        unsqueeze_547: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_545);  convolution_81 = unsqueeze_545 = None
        mul_237: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_549: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_238: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_549);  mul_237 = unsqueeze_549 = None
        unsqueeze_550: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_551: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_148: "f32[8, 162, 56, 56]" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_551);  mul_238 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_32: "f32[8, 162, 56, 56]" = torch.ops.aten.sigmoid.default(add_148)
        mul_239: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(add_148, sigmoid_32);  add_148 = sigmoid_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_82: "f32[8, 162, 56, 56]" = torch.ops.aten.convolution.default(mul_239, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 162);  mul_239 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_149: "f32[162]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_69: "f32[162]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_69: "f32[162]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_240: "f32[162]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_553: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_555: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_553);  convolution_82 = unsqueeze_553 = None
        mul_241: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_557: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_242: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_557);  mul_241 = unsqueeze_557 = None
        unsqueeze_558: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_559: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_150: "f32[8, 162, 56, 56]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_559);  mul_242 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_18: "f32[8, 162, 56, 56]" = torch.ops.aten.clamp_min.default(add_150, 0.0);  add_150 = None
        clamp_max_18: "f32[8, 162, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6.0);  clamp_min_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_83: "f32[8, 38, 56, 56]" = torch.ops.aten.convolution.default(clamp_max_18, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_18 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_151: "f32[38]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_70: "f32[38]" = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_70: "f32[38]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_243: "f32[38]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560: "f32[38, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_561: "f32[38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562: "f32[38, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_563: "f32[38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70: "f32[8, 38, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_561);  convolution_83 = unsqueeze_561 = None
        mul_244: "f32[8, 38, 56, 56]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[38, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_565: "f32[38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_245: "f32[8, 38, 56, 56]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_565);  mul_244 = unsqueeze_565 = None
        unsqueeze_566: "f32[38, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_567: "f32[38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_152: "f32[8, 38, 56, 56]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_567);  mul_245 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_46: "f32[8, 27, 56, 56]" = torch.ops.aten.slice.Tensor(add_152, 1, 0, 27)
        add_153: "f32[8, 27, 56, 56]" = torch.ops.aten.add.Tensor(slice_46, add_146);  slice_46 = add_146 = None
        slice_48: "f32[8, 11, 56, 56]" = torch.ops.aten.slice.Tensor(add_152, 1, 27, 9223372036854775807);  add_152 = None
        cat_11: "f32[8, 38, 56, 56]" = torch.ops.aten.cat.default([add_153, slice_48], 1);  add_153 = slice_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_84: "f32[8, 228, 56, 56]" = torch.ops.aten.convolution.default(cat_11, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_11 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_154: "f32[228]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_71: "f32[228]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_71: "f32[228]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_246: "f32[228]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_569: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
        unsqueeze_571: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71: "f32[8, 228, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_569);  convolution_84 = unsqueeze_569 = None
        mul_247: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_573: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_248: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_573);  mul_247 = unsqueeze_573 = None
        unsqueeze_574: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_575: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_155: "f32[8, 228, 56, 56]" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_575);  mul_248 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_33: "f32[8, 228, 56, 56]" = torch.ops.aten.sigmoid.default(add_155)
        mul_249: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(add_155, sigmoid_33);  add_155 = sigmoid_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_85: "f32[8, 228, 28, 28]" = torch.ops.aten.convolution.default(mul_249, arg51_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 228);  mul_249 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_156: "f32[228]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_72: "f32[228]" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_72: "f32[228]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_250: "f32[228]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_577: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_579: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72: "f32[8, 228, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_577);  convolution_85 = unsqueeze_577 = None
        mul_251: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_581: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_252: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_581);  mul_251 = unsqueeze_581 = None
        unsqueeze_582: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_583: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_157: "f32[8, 228, 28, 28]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_583);  mul_252 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "f32[8, 228, 1, 1]" = torch.ops.aten.mean.dim(add_157, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_86: "f32[8, 19, 1, 1]" = torch.ops.aten.convolution.default(mean_14, arg56_1, arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg56_1 = arg57_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_158: "f32[19]" = torch.ops.aten.add.Tensor(arg59_1, 1e-05);  arg59_1 = None
        sqrt_73: "f32[19]" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_73: "f32[19]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_253: "f32[19]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584: "f32[19, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_585: "f32[19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586: "f32[19, 1]" = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_587: "f32[19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73: "f32[8, 19, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_585);  convolution_86 = unsqueeze_585 = None
        mul_254: "f32[8, 19, 1, 1]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[19, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_589: "f32[19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_255: "f32[8, 19, 1, 1]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_589);  mul_254 = unsqueeze_589 = None
        unsqueeze_590: "f32[19, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_591: "f32[19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_159: "f32[8, 19, 1, 1]" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_591);  mul_255 = unsqueeze_591 = None
        relu_13: "f32[8, 19, 1, 1]" = torch.ops.aten.relu.default(add_159);  add_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_87: "f32[8, 228, 1, 1]" = torch.ops.aten.convolution.default(relu_13, arg62_1, arg63_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg62_1 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_34: "f32[8, 228, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_256: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(add_157, sigmoid_34);  add_157 = sigmoid_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_19: "f32[8, 228, 28, 28]" = torch.ops.aten.clamp_min.default(mul_256, 0.0);  mul_256 = None
        clamp_max_19: "f32[8, 228, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6.0);  clamp_min_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_88: "f32[8, 50, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_19, arg64_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_19 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_160: "f32[50]" = torch.ops.aten.add.Tensor(arg66_1, 1e-05);  arg66_1 = None
        sqrt_74: "f32[50]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_74: "f32[50]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_257: "f32[50]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592: "f32[50, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_593: "f32[50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594: "f32[50, 1]" = torch.ops.aten.unsqueeze.default(mul_257, -1);  mul_257 = None
        unsqueeze_595: "f32[50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74: "f32[8, 50, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_593);  convolution_88 = unsqueeze_593 = None
        mul_258: "f32[8, 50, 28, 28]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[50, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_597: "f32[50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_259: "f32[8, 50, 28, 28]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_597);  mul_258 = unsqueeze_597 = None
        unsqueeze_598: "f32[50, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_599: "f32[50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_161: "f32[8, 50, 28, 28]" = torch.ops.aten.add.Tensor(mul_259, unsqueeze_599);  mul_259 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_89: "f32[8, 300, 28, 28]" = torch.ops.aten.convolution.default(add_161, arg69_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_162: "f32[300]" = torch.ops.aten.add.Tensor(arg71_1, 1e-05);  arg71_1 = None
        sqrt_75: "f32[300]" = torch.ops.aten.sqrt.default(add_162);  add_162 = None
        reciprocal_75: "f32[300]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_260: "f32[300]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_601: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(mul_260, -1);  mul_260 = None
        unsqueeze_603: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_601);  convolution_89 = unsqueeze_601 = None
        mul_261: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_605: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_262: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_605);  mul_261 = unsqueeze_605 = None
        unsqueeze_606: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_607: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_163: "f32[8, 300, 28, 28]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_607);  mul_262 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_35: "f32[8, 300, 28, 28]" = torch.ops.aten.sigmoid.default(add_163)
        mul_263: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(add_163, sigmoid_35);  add_163 = sigmoid_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_90: "f32[8, 300, 28, 28]" = torch.ops.aten.convolution.default(mul_263, arg74_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 300);  mul_263 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_164: "f32[300]" = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_76: "f32[300]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_76: "f32[300]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_264: "f32[300]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_609: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_611: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_609);  convolution_90 = unsqueeze_609 = None
        mul_265: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_613: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_266: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_613);  mul_265 = unsqueeze_613 = None
        unsqueeze_614: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_615: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_165: "f32[8, 300, 28, 28]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_615);  mul_266 = unsqueeze_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_15: "f32[8, 300, 1, 1]" = torch.ops.aten.mean.dim(add_165, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_91: "f32[8, 25, 1, 1]" = torch.ops.aten.convolution.default(mean_15, arg79_1, arg80_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg79_1 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_166: "f32[25]" = torch.ops.aten.add.Tensor(arg82_1, 1e-05);  arg82_1 = None
        sqrt_77: "f32[25]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_77: "f32[25]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_267: "f32[25]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616: "f32[25, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_617: "f32[25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618: "f32[25, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_619: "f32[25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77: "f32[8, 25, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_617);  convolution_91 = unsqueeze_617 = None
        mul_268: "f32[8, 25, 1, 1]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[25, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_621: "f32[25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_269: "f32[8, 25, 1, 1]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_621);  mul_268 = unsqueeze_621 = None
        unsqueeze_622: "f32[25, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_623: "f32[25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_167: "f32[8, 25, 1, 1]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_623);  mul_269 = unsqueeze_623 = None
        relu_14: "f32[8, 25, 1, 1]" = torch.ops.aten.relu.default(add_167);  add_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_92: "f32[8, 300, 1, 1]" = torch.ops.aten.convolution.default(relu_14, arg85_1, arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg85_1 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_36: "f32[8, 300, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_92);  convolution_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_270: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(add_165, sigmoid_36);  add_165 = sigmoid_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_20: "f32[8, 300, 28, 28]" = torch.ops.aten.clamp_min.default(mul_270, 0.0);  mul_270 = None
        clamp_max_20: "f32[8, 300, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6.0);  clamp_min_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_93: "f32[8, 61, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_20, arg87_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_20 = arg87_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_168: "f32[61]" = torch.ops.aten.add.Tensor(arg89_1, 1e-05);  arg89_1 = None
        sqrt_78: "f32[61]" = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_78: "f32[61]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_271: "f32[61]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624: "f32[61, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_625: "f32[61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626: "f32[61, 1]" = torch.ops.aten.unsqueeze.default(mul_271, -1);  mul_271 = None
        unsqueeze_627: "f32[61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78: "f32[8, 61, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_625);  convolution_93 = unsqueeze_625 = None
        mul_272: "f32[8, 61, 28, 28]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[61, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_629: "f32[61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_273: "f32[8, 61, 28, 28]" = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_629);  mul_272 = unsqueeze_629 = None
        unsqueeze_630: "f32[61, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_631: "f32[61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_169: "f32[8, 61, 28, 28]" = torch.ops.aten.add.Tensor(mul_273, unsqueeze_631);  mul_273 = unsqueeze_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_50: "f32[8, 50, 28, 28]" = torch.ops.aten.slice.Tensor(add_169, 1, 0, 50)
        add_170: "f32[8, 50, 28, 28]" = torch.ops.aten.add.Tensor(slice_50, add_161);  slice_50 = add_161 = None
        slice_52: "f32[8, 11, 28, 28]" = torch.ops.aten.slice.Tensor(add_169, 1, 50, 9223372036854775807);  add_169 = None
        cat_12: "f32[8, 61, 28, 28]" = torch.ops.aten.cat.default([add_170, slice_52], 1);  add_170 = slice_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_94: "f32[8, 366, 28, 28]" = torch.ops.aten.convolution.default(cat_12, arg92_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_12 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_171: "f32[366]" = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_79: "f32[366]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_79: "f32[366]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_274: "f32[366]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_633: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(mul_274, -1);  mul_274 = None
        unsqueeze_635: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79: "f32[8, 366, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_633);  convolution_94 = unsqueeze_633 = None
        mul_275: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_637: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_276: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_637);  mul_275 = unsqueeze_637 = None
        unsqueeze_638: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_639: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_172: "f32[8, 366, 28, 28]" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_639);  mul_276 = unsqueeze_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_37: "f32[8, 366, 28, 28]" = torch.ops.aten.sigmoid.default(add_172)
        mul_277: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(add_172, sigmoid_37);  add_172 = sigmoid_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_95: "f32[8, 366, 14, 14]" = torch.ops.aten.convolution.default(mul_277, arg97_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 366);  mul_277 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_173: "f32[366]" = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_80: "f32[366]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_80: "f32[366]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_278: "f32[366]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_641: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(mul_278, -1);  mul_278 = None
        unsqueeze_643: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80: "f32[8, 366, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_641);  convolution_95 = unsqueeze_641 = None
        mul_279: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_645: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_280: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(mul_279, unsqueeze_645);  mul_279 = unsqueeze_645 = None
        unsqueeze_646: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_647: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_174: "f32[8, 366, 14, 14]" = torch.ops.aten.add.Tensor(mul_280, unsqueeze_647);  mul_280 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_16: "f32[8, 366, 1, 1]" = torch.ops.aten.mean.dim(add_174, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_96: "f32[8, 30, 1, 1]" = torch.ops.aten.convolution.default(mean_16, arg102_1, arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_16 = arg102_1 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_175: "f32[30]" = torch.ops.aten.add.Tensor(arg105_1, 1e-05);  arg105_1 = None
        sqrt_81: "f32[30]" = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_81: "f32[30]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_281: "f32[30]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648: "f32[30, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_649: "f32[30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650: "f32[30, 1]" = torch.ops.aten.unsqueeze.default(mul_281, -1);  mul_281 = None
        unsqueeze_651: "f32[30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81: "f32[8, 30, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_649);  convolution_96 = unsqueeze_649 = None
        mul_282: "f32[8, 30, 1, 1]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[30, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_653: "f32[30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_283: "f32[8, 30, 1, 1]" = torch.ops.aten.mul.Tensor(mul_282, unsqueeze_653);  mul_282 = unsqueeze_653 = None
        unsqueeze_654: "f32[30, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_655: "f32[30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_176: "f32[8, 30, 1, 1]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_655);  mul_283 = unsqueeze_655 = None
        relu_15: "f32[8, 30, 1, 1]" = torch.ops.aten.relu.default(add_176);  add_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_97: "f32[8, 366, 1, 1]" = torch.ops.aten.convolution.default(relu_15, arg108_1, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg108_1 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_38: "f32[8, 366, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_97);  convolution_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_284: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(add_174, sigmoid_38);  add_174 = sigmoid_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_21: "f32[8, 366, 14, 14]" = torch.ops.aten.clamp_min.default(mul_284, 0.0);  mul_284 = None
        clamp_max_21: "f32[8, 366, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6.0);  clamp_min_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_98: "f32[8, 72, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_21, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_21 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_177: "f32[72]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_82: "f32[72]" = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_82: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_285: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_657: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
        unsqueeze_659: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_657);  convolution_98 = unsqueeze_657 = None
        mul_286: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_661: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_287: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_661);  mul_286 = unsqueeze_661 = None
        unsqueeze_662: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_663: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_178: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_663);  mul_287 = unsqueeze_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_99: "f32[8, 432, 14, 14]" = torch.ops.aten.convolution.default(add_178, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_179: "f32[432]" = torch.ops.aten.add.Tensor(arg117_1, 1e-05);  arg117_1 = None
        sqrt_83: "f32[432]" = torch.ops.aten.sqrt.default(add_179);  add_179 = None
        reciprocal_83: "f32[432]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_288: "f32[432]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_665: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_667: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_665);  convolution_99 = unsqueeze_665 = None
        mul_289: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_669: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_290: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_669);  mul_289 = unsqueeze_669 = None
        unsqueeze_670: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_671: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_180: "f32[8, 432, 14, 14]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_671);  mul_290 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_39: "f32[8, 432, 14, 14]" = torch.ops.aten.sigmoid.default(add_180)
        mul_291: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, sigmoid_39);  add_180 = sigmoid_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_100: "f32[8, 432, 14, 14]" = torch.ops.aten.convolution.default(mul_291, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  mul_291 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_181: "f32[432]" = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_84: "f32[432]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_84: "f32[432]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_292: "f32[432]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_673: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_675: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_673);  convolution_100 = unsqueeze_673 = None
        mul_293: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_677: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_294: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_677);  mul_293 = unsqueeze_677 = None
        unsqueeze_678: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_679: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_182: "f32[8, 432, 14, 14]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_679);  mul_294 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_17: "f32[8, 432, 1, 1]" = torch.ops.aten.mean.dim(add_182, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_101: "f32[8, 36, 1, 1]" = torch.ops.aten.convolution.default(mean_17, arg125_1, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg125_1 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_183: "f32[36]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_85: "f32[36]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_85: "f32[36]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_295: "f32[36]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_681: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_683: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85: "f32[8, 36, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_681);  convolution_101 = unsqueeze_681 = None
        mul_296: "f32[8, 36, 1, 1]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_685: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_297: "f32[8, 36, 1, 1]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_685);  mul_296 = unsqueeze_685 = None
        unsqueeze_686: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_687: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_184: "f32[8, 36, 1, 1]" = torch.ops.aten.add.Tensor(mul_297, unsqueeze_687);  mul_297 = unsqueeze_687 = None
        relu_16: "f32[8, 36, 1, 1]" = torch.ops.aten.relu.default(add_184);  add_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_102: "f32[8, 432, 1, 1]" = torch.ops.aten.convolution.default(relu_16, arg131_1, arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_16 = arg131_1 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_40: "f32[8, 432, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_102);  convolution_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_298: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(add_182, sigmoid_40);  add_182 = sigmoid_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_22: "f32[8, 432, 14, 14]" = torch.ops.aten.clamp_min.default(mul_298, 0.0);  mul_298 = None
        clamp_max_22: "f32[8, 432, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6.0);  clamp_min_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_103: "f32[8, 84, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_22, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_22 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_185: "f32[84]" = torch.ops.aten.add.Tensor(arg135_1, 1e-05);  arg135_1 = None
        sqrt_86: "f32[84]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_86: "f32[84]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_299: "f32[84]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688: "f32[84, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_689: "f32[84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690: "f32[84, 1]" = torch.ops.aten.unsqueeze.default(mul_299, -1);  mul_299 = None
        unsqueeze_691: "f32[84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86: "f32[8, 84, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_689);  convolution_103 = unsqueeze_689 = None
        mul_300: "f32[8, 84, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[84, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_693: "f32[84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_301: "f32[8, 84, 14, 14]" = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_693);  mul_300 = unsqueeze_693 = None
        unsqueeze_694: "f32[84, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_695: "f32[84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_186: "f32[8, 84, 14, 14]" = torch.ops.aten.add.Tensor(mul_301, unsqueeze_695);  mul_301 = unsqueeze_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_54: "f32[8, 72, 14, 14]" = torch.ops.aten.slice.Tensor(add_186, 1, 0, 72)
        add_187: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(slice_54, add_178);  slice_54 = add_178 = None
        slice_56: "f32[8, 12, 14, 14]" = torch.ops.aten.slice.Tensor(add_186, 1, 72, 9223372036854775807);  add_186 = None
        cat_13: "f32[8, 84, 14, 14]" = torch.ops.aten.cat.default([add_187, slice_56], 1);  add_187 = slice_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_104: "f32[8, 504, 14, 14]" = torch.ops.aten.convolution.default(cat_13, arg138_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_188: "f32[504]" = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_87: "f32[504]" = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_87: "f32[504]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_302: "f32[504]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_697: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(mul_302, -1);  mul_302 = None
        unsqueeze_699: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_697);  convolution_104 = unsqueeze_697 = None
        mul_303: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_701: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_304: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(mul_303, unsqueeze_701);  mul_303 = unsqueeze_701 = None
        unsqueeze_702: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_703: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_189: "f32[8, 504, 14, 14]" = torch.ops.aten.add.Tensor(mul_304, unsqueeze_703);  mul_304 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_41: "f32[8, 504, 14, 14]" = torch.ops.aten.sigmoid.default(add_189)
        mul_305: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(add_189, sigmoid_41);  add_189 = sigmoid_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_105: "f32[8, 504, 14, 14]" = torch.ops.aten.convolution.default(mul_305, arg143_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 504);  mul_305 = arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_190: "f32[504]" = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_88: "f32[504]" = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_88: "f32[504]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_306: "f32[504]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_705: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
        unsqueeze_707: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_705);  convolution_105 = unsqueeze_705 = None
        mul_307: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_709: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_308: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_709);  mul_307 = unsqueeze_709 = None
        unsqueeze_710: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_711: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_191: "f32[8, 504, 14, 14]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_711);  mul_308 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_18: "f32[8, 504, 1, 1]" = torch.ops.aten.mean.dim(add_191, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_106: "f32[8, 42, 1, 1]" = torch.ops.aten.convolution.default(mean_18, arg148_1, arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg148_1 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_192: "f32[42]" = torch.ops.aten.add.Tensor(arg151_1, 1e-05);  arg151_1 = None
        sqrt_89: "f32[42]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_89: "f32[42]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_309: "f32[42]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712: "f32[42, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_713: "f32[42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714: "f32[42, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_715: "f32[42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89: "f32[8, 42, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_713);  convolution_106 = unsqueeze_713 = None
        mul_310: "f32[8, 42, 1, 1]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[42, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_717: "f32[42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_311: "f32[8, 42, 1, 1]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_717);  mul_310 = unsqueeze_717 = None
        unsqueeze_718: "f32[42, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_719: "f32[42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_193: "f32[8, 42, 1, 1]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_719);  mul_311 = unsqueeze_719 = None
        relu_17: "f32[8, 42, 1, 1]" = torch.ops.aten.relu.default(add_193);  add_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_107: "f32[8, 504, 1, 1]" = torch.ops.aten.convolution.default(relu_17, arg154_1, arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg154_1 = arg155_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_42: "f32[8, 504, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_107);  convolution_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_312: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(add_191, sigmoid_42);  add_191 = sigmoid_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_23: "f32[8, 504, 14, 14]" = torch.ops.aten.clamp_min.default(mul_312, 0.0);  mul_312 = None
        clamp_max_23: "f32[8, 504, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6.0);  clamp_min_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_108: "f32[8, 95, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_23, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_23 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_194: "f32[95]" = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_90: "f32[95]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_90: "f32[95]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_313: "f32[95]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720: "f32[95, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_721: "f32[95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722: "f32[95, 1]" = torch.ops.aten.unsqueeze.default(mul_313, -1);  mul_313 = None
        unsqueeze_723: "f32[95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90: "f32[8, 95, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_721);  convolution_108 = unsqueeze_721 = None
        mul_314: "f32[8, 95, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[95, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_725: "f32[95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_315: "f32[8, 95, 14, 14]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_725);  mul_314 = unsqueeze_725 = None
        unsqueeze_726: "f32[95, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_727: "f32[95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_195: "f32[8, 95, 14, 14]" = torch.ops.aten.add.Tensor(mul_315, unsqueeze_727);  mul_315 = unsqueeze_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_58: "f32[8, 84, 14, 14]" = torch.ops.aten.slice.Tensor(add_195, 1, 0, 84)
        add_196: "f32[8, 84, 14, 14]" = torch.ops.aten.add.Tensor(slice_58, cat_13);  slice_58 = cat_13 = None
        slice_60: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(add_195, 1, 84, 9223372036854775807);  add_195 = None
        cat_14: "f32[8, 95, 14, 14]" = torch.ops.aten.cat.default([add_196, slice_60], 1);  add_196 = slice_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_109: "f32[8, 570, 14, 14]" = torch.ops.aten.convolution.default(cat_14, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_197: "f32[570]" = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_91: "f32[570]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_91: "f32[570]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_316: "f32[570]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_729: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(mul_316, -1);  mul_316 = None
        unsqueeze_731: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_729);  convolution_109 = unsqueeze_729 = None
        mul_317: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_733: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_318: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_733);  mul_317 = unsqueeze_733 = None
        unsqueeze_734: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_735: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_198: "f32[8, 570, 14, 14]" = torch.ops.aten.add.Tensor(mul_318, unsqueeze_735);  mul_318 = unsqueeze_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_43: "f32[8, 570, 14, 14]" = torch.ops.aten.sigmoid.default(add_198)
        mul_319: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(add_198, sigmoid_43);  add_198 = sigmoid_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_110: "f32[8, 570, 14, 14]" = torch.ops.aten.convolution.default(mul_319, arg166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 570);  mul_319 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_199: "f32[570]" = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_92: "f32[570]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_92: "f32[570]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_320: "f32[570]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_737: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_739: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_737);  convolution_110 = unsqueeze_737 = None
        mul_321: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_741: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_322: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_741);  mul_321 = unsqueeze_741 = None
        unsqueeze_742: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_743: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_200: "f32[8, 570, 14, 14]" = torch.ops.aten.add.Tensor(mul_322, unsqueeze_743);  mul_322 = unsqueeze_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_19: "f32[8, 570, 1, 1]" = torch.ops.aten.mean.dim(add_200, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_111: "f32[8, 47, 1, 1]" = torch.ops.aten.convolution.default(mean_19, arg171_1, arg172_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg171_1 = arg172_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_201: "f32[47]" = torch.ops.aten.add.Tensor(arg174_1, 1e-05);  arg174_1 = None
        sqrt_93: "f32[47]" = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_93: "f32[47]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_323: "f32[47]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744: "f32[47, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_745: "f32[47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746: "f32[47, 1]" = torch.ops.aten.unsqueeze.default(mul_323, -1);  mul_323 = None
        unsqueeze_747: "f32[47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93: "f32[8, 47, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_745);  convolution_111 = unsqueeze_745 = None
        mul_324: "f32[8, 47, 1, 1]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f32[47, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_749: "f32[47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_325: "f32[8, 47, 1, 1]" = torch.ops.aten.mul.Tensor(mul_324, unsqueeze_749);  mul_324 = unsqueeze_749 = None
        unsqueeze_750: "f32[47, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_751: "f32[47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_202: "f32[8, 47, 1, 1]" = torch.ops.aten.add.Tensor(mul_325, unsqueeze_751);  mul_325 = unsqueeze_751 = None
        relu_18: "f32[8, 47, 1, 1]" = torch.ops.aten.relu.default(add_202);  add_202 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_112: "f32[8, 570, 1, 1]" = torch.ops.aten.convolution.default(relu_18, arg177_1, arg178_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_18 = arg177_1 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_44: "f32[8, 570, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_112);  convolution_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_326: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(add_200, sigmoid_44);  add_200 = sigmoid_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_24: "f32[8, 570, 14, 14]" = torch.ops.aten.clamp_min.default(mul_326, 0.0);  mul_326 = None
        clamp_max_24: "f32[8, 570, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6.0);  clamp_min_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_113: "f32[8, 106, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_24, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_24 = arg179_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_203: "f32[106]" = torch.ops.aten.add.Tensor(arg181_1, 1e-05);  arg181_1 = None
        sqrt_94: "f32[106]" = torch.ops.aten.sqrt.default(add_203);  add_203 = None
        reciprocal_94: "f32[106]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_327: "f32[106]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752: "f32[106, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_753: "f32[106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754: "f32[106, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_755: "f32[106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94: "f32[8, 106, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_753);  convolution_113 = unsqueeze_753 = None
        mul_328: "f32[8, 106, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f32[106, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_757: "f32[106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_329: "f32[8, 106, 14, 14]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_757);  mul_328 = unsqueeze_757 = None
        unsqueeze_758: "f32[106, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_759: "f32[106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_204: "f32[8, 106, 14, 14]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_759);  mul_329 = unsqueeze_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_62: "f32[8, 95, 14, 14]" = torch.ops.aten.slice.Tensor(add_204, 1, 0, 95)
        add_205: "f32[8, 95, 14, 14]" = torch.ops.aten.add.Tensor(slice_62, cat_14);  slice_62 = cat_14 = None
        slice_64: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(add_204, 1, 95, 9223372036854775807);  add_204 = None
        cat_15: "f32[8, 106, 14, 14]" = torch.ops.aten.cat.default([add_205, slice_64], 1);  add_205 = slice_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_114: "f32[8, 636, 14, 14]" = torch.ops.aten.convolution.default(cat_15, arg184_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg184_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_206: "f32[636]" = torch.ops.aten.add.Tensor(arg186_1, 1e-05);  arg186_1 = None
        sqrt_95: "f32[636]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_95: "f32[636]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_330: "f32[636]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_761: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_763: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_761);  convolution_114 = unsqueeze_761 = None
        mul_331: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_765: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_332: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_765);  mul_331 = unsqueeze_765 = None
        unsqueeze_766: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_767: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_207: "f32[8, 636, 14, 14]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_767);  mul_332 = unsqueeze_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_45: "f32[8, 636, 14, 14]" = torch.ops.aten.sigmoid.default(add_207)
        mul_333: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(add_207, sigmoid_45);  add_207 = sigmoid_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_115: "f32[8, 636, 14, 14]" = torch.ops.aten.convolution.default(mul_333, arg189_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 636);  mul_333 = arg189_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_208: "f32[636]" = torch.ops.aten.add.Tensor(arg191_1, 1e-05);  arg191_1 = None
        sqrt_96: "f32[636]" = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_96: "f32[636]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_334: "f32[636]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_769: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(mul_334, -1);  mul_334 = None
        unsqueeze_771: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_769);  convolution_115 = unsqueeze_769 = None
        mul_335: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_773: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_336: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(mul_335, unsqueeze_773);  mul_335 = unsqueeze_773 = None
        unsqueeze_774: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_775: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_209: "f32[8, 636, 14, 14]" = torch.ops.aten.add.Tensor(mul_336, unsqueeze_775);  mul_336 = unsqueeze_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_20: "f32[8, 636, 1, 1]" = torch.ops.aten.mean.dim(add_209, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_116: "f32[8, 53, 1, 1]" = torch.ops.aten.convolution.default(mean_20, arg194_1, arg195_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg194_1 = arg195_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_210: "f32[53]" = torch.ops.aten.add.Tensor(arg197_1, 1e-05);  arg197_1 = None
        sqrt_97: "f32[53]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_97: "f32[53]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_337: "f32[53]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776: "f32[53, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_777: "f32[53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778: "f32[53, 1]" = torch.ops.aten.unsqueeze.default(mul_337, -1);  mul_337 = None
        unsqueeze_779: "f32[53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97: "f32[8, 53, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_777);  convolution_116 = unsqueeze_777 = None
        mul_338: "f32[8, 53, 1, 1]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f32[53, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_781: "f32[53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_339: "f32[8, 53, 1, 1]" = torch.ops.aten.mul.Tensor(mul_338, unsqueeze_781);  mul_338 = unsqueeze_781 = None
        unsqueeze_782: "f32[53, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_783: "f32[53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_211: "f32[8, 53, 1, 1]" = torch.ops.aten.add.Tensor(mul_339, unsqueeze_783);  mul_339 = unsqueeze_783 = None
        relu_19: "f32[8, 53, 1, 1]" = torch.ops.aten.relu.default(add_211);  add_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_117: "f32[8, 636, 1, 1]" = torch.ops.aten.convolution.default(relu_19, arg200_1, arg201_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_19 = arg200_1 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_46: "f32[8, 636, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_117);  convolution_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_340: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(add_209, sigmoid_46);  add_209 = sigmoid_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_25: "f32[8, 636, 14, 14]" = torch.ops.aten.clamp_min.default(mul_340, 0.0);  mul_340 = None
        clamp_max_25: "f32[8, 636, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6.0);  clamp_min_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_118: "f32[8, 117, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_25, arg202_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_25 = arg202_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_212: "f32[117]" = torch.ops.aten.add.Tensor(arg204_1, 1e-05);  arg204_1 = None
        sqrt_98: "f32[117]" = torch.ops.aten.sqrt.default(add_212);  add_212 = None
        reciprocal_98: "f32[117]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_341: "f32[117]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784: "f32[117, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_785: "f32[117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786: "f32[117, 1]" = torch.ops.aten.unsqueeze.default(mul_341, -1);  mul_341 = None
        unsqueeze_787: "f32[117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98: "f32[8, 117, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_785);  convolution_118 = unsqueeze_785 = None
        mul_342: "f32[8, 117, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788: "f32[117, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_789: "f32[117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_343: "f32[8, 117, 14, 14]" = torch.ops.aten.mul.Tensor(mul_342, unsqueeze_789);  mul_342 = unsqueeze_789 = None
        unsqueeze_790: "f32[117, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_791: "f32[117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_213: "f32[8, 117, 14, 14]" = torch.ops.aten.add.Tensor(mul_343, unsqueeze_791);  mul_343 = unsqueeze_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_66: "f32[8, 106, 14, 14]" = torch.ops.aten.slice.Tensor(add_213, 1, 0, 106)
        add_214: "f32[8, 106, 14, 14]" = torch.ops.aten.add.Tensor(slice_66, cat_15);  slice_66 = cat_15 = None
        slice_68: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(add_213, 1, 106, 9223372036854775807);  add_213 = None
        cat_16: "f32[8, 117, 14, 14]" = torch.ops.aten.cat.default([add_214, slice_68], 1);  add_214 = slice_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_119: "f32[8, 702, 14, 14]" = torch.ops.aten.convolution.default(cat_16, arg207_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg207_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_215: "f32[702]" = torch.ops.aten.add.Tensor(arg209_1, 1e-05);  arg209_1 = None
        sqrt_99: "f32[702]" = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_99: "f32[702]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_344: "f32[702]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_793: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(mul_344, -1);  mul_344 = None
        unsqueeze_795: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_793);  convolution_119 = unsqueeze_793 = None
        mul_345: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_797: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_346: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(mul_345, unsqueeze_797);  mul_345 = unsqueeze_797 = None
        unsqueeze_798: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_799: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_216: "f32[8, 702, 14, 14]" = torch.ops.aten.add.Tensor(mul_346, unsqueeze_799);  mul_346 = unsqueeze_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_47: "f32[8, 702, 14, 14]" = torch.ops.aten.sigmoid.default(add_216)
        mul_347: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(add_216, sigmoid_47);  add_216 = sigmoid_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_120: "f32[8, 702, 14, 14]" = torch.ops.aten.convolution.default(mul_347, arg212_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 702);  mul_347 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_217: "f32[702]" = torch.ops.aten.add.Tensor(arg214_1, 1e-05);  arg214_1 = None
        sqrt_100: "f32[702]" = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_100: "f32[702]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_348: "f32[702]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_801: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_803: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_801);  convolution_120 = unsqueeze_801 = None
        mul_349: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_805: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_350: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_805);  mul_349 = unsqueeze_805 = None
        unsqueeze_806: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_807: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_218: "f32[8, 702, 14, 14]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_807);  mul_350 = unsqueeze_807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_21: "f32[8, 702, 1, 1]" = torch.ops.aten.mean.dim(add_218, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_121: "f32[8, 58, 1, 1]" = torch.ops.aten.convolution.default(mean_21, arg217_1, arg218_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg217_1 = arg218_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_219: "f32[58]" = torch.ops.aten.add.Tensor(arg220_1, 1e-05);  arg220_1 = None
        sqrt_101: "f32[58]" = torch.ops.aten.sqrt.default(add_219);  add_219 = None
        reciprocal_101: "f32[58]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_351: "f32[58]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_809: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_811: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101: "f32[8, 58, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_809);  convolution_121 = unsqueeze_809 = None
        mul_352: "f32[8, 58, 1, 1]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_813: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_353: "f32[8, 58, 1, 1]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_813);  mul_352 = unsqueeze_813 = None
        unsqueeze_814: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_815: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_220: "f32[8, 58, 1, 1]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_815);  mul_353 = unsqueeze_815 = None
        relu_20: "f32[8, 58, 1, 1]" = torch.ops.aten.relu.default(add_220);  add_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_122: "f32[8, 702, 1, 1]" = torch.ops.aten.convolution.default(relu_20, arg223_1, arg224_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = arg223_1 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_48: "f32[8, 702, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_122);  convolution_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_354: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(add_218, sigmoid_48);  add_218 = sigmoid_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_26: "f32[8, 702, 14, 14]" = torch.ops.aten.clamp_min.default(mul_354, 0.0);  mul_354 = None
        clamp_max_26: "f32[8, 702, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6.0);  clamp_min_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_123: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_26, arg225_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_26 = arg225_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_221: "f32[128]" = torch.ops.aten.add.Tensor(arg227_1, 1e-05);  arg227_1 = None
        sqrt_102: "f32[128]" = torch.ops.aten.sqrt.default(add_221);  add_221 = None
        reciprocal_102: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_355: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_817: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_355, -1);  mul_355 = None
        unsqueeze_819: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_817);  convolution_123 = unsqueeze_817 = None
        mul_356: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_821: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_357: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_356, unsqueeze_821);  mul_356 = unsqueeze_821 = None
        unsqueeze_822: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_823: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_222: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_357, unsqueeze_823);  mul_357 = unsqueeze_823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_70: "f32[8, 117, 14, 14]" = torch.ops.aten.slice.Tensor(add_222, 1, 0, 117)
        add_223: "f32[8, 117, 14, 14]" = torch.ops.aten.add.Tensor(slice_70, cat_16);  slice_70 = cat_16 = None
        slice_72: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(add_222, 1, 117, 9223372036854775807);  add_222 = None
        cat_17: "f32[8, 128, 14, 14]" = torch.ops.aten.cat.default([add_223, slice_72], 1);  add_223 = slice_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_124: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(cat_17, arg230_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_17 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_224: "f32[768]" = torch.ops.aten.add.Tensor(arg232_1, 1e-05);  arg232_1 = None
        sqrt_103: "f32[768]" = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_103: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_358: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_825: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_358, -1);  mul_358 = None
        unsqueeze_827: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_825);  convolution_124 = unsqueeze_825 = None
        mul_359: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_829: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_360: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_359, unsqueeze_829);  mul_359 = unsqueeze_829 = None
        unsqueeze_830: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_831: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_225: "f32[8, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_360, unsqueeze_831);  mul_360 = unsqueeze_831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_49: "f32[8, 768, 14, 14]" = torch.ops.aten.sigmoid.default(add_225)
        mul_361: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(add_225, sigmoid_49);  add_225 = sigmoid_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_125: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_361, arg235_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 768);  mul_361 = arg235_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_226: "f32[768]" = torch.ops.aten.add.Tensor(arg237_1, 1e-05);  arg237_1 = None
        sqrt_104: "f32[768]" = torch.ops.aten.sqrt.default(add_226);  add_226 = None
        reciprocal_104: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_362: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_833: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_362, -1);  mul_362 = None
        unsqueeze_835: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_833);  convolution_125 = unsqueeze_833 = None
        mul_363: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_837: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_364: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_363, unsqueeze_837);  mul_363 = unsqueeze_837 = None
        unsqueeze_838: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_839: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_227: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_364, unsqueeze_839);  mul_364 = unsqueeze_839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_22: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_227, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_126: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_22, arg240_1, arg241_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg240_1 = arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_228: "f32[64]" = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_105: "f32[64]" = torch.ops.aten.sqrt.default(add_228);  add_228 = None
        reciprocal_105: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_365: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_841: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_365, -1);  mul_365 = None
        unsqueeze_843: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_841);  convolution_126 = unsqueeze_841 = None
        mul_366: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_845: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_367: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_366, unsqueeze_845);  mul_366 = unsqueeze_845 = None
        unsqueeze_846: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_847: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_229: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_367, unsqueeze_847);  mul_367 = unsqueeze_847 = None
        relu_21: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_229);  add_229 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_127: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(relu_21, arg246_1, arg247_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg246_1 = arg247_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_50: "f32[8, 768, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_127);  convolution_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_368: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_227, sigmoid_50);  add_227 = sigmoid_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_27: "f32[8, 768, 7, 7]" = torch.ops.aten.clamp_min.default(mul_368, 0.0);  mul_368 = None
        clamp_max_27: "f32[8, 768, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6.0);  clamp_min_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_128: "f32[8, 140, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_27, arg248_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_27 = arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_230: "f32[140]" = torch.ops.aten.add.Tensor(arg250_1, 1e-05);  arg250_1 = None
        sqrt_106: "f32[140]" = torch.ops.aten.sqrt.default(add_230);  add_230 = None
        reciprocal_106: "f32[140]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_369: "f32[140]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848: "f32[140, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_849: "f32[140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850: "f32[140, 1]" = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_851: "f32[140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106: "f32[8, 140, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_849);  convolution_128 = unsqueeze_849 = None
        mul_370: "f32[8, 140, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852: "f32[140, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_853: "f32[140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_371: "f32[8, 140, 7, 7]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_853);  mul_370 = unsqueeze_853 = None
        unsqueeze_854: "f32[140, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_855: "f32[140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_231: "f32[8, 140, 7, 7]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_855);  mul_371 = unsqueeze_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_129: "f32[8, 840, 7, 7]" = torch.ops.aten.convolution.default(add_231, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_232: "f32[840]" = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_107: "f32[840]" = torch.ops.aten.sqrt.default(add_232);  add_232 = None
        reciprocal_107: "f32[840]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_372: "f32[840]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_857: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_859: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_857);  convolution_129 = unsqueeze_857 = None
        mul_373: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_861: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_374: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_861);  mul_373 = unsqueeze_861 = None
        unsqueeze_862: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_863: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_233: "f32[8, 840, 7, 7]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_863);  mul_374 = unsqueeze_863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_51: "f32[8, 840, 7, 7]" = torch.ops.aten.sigmoid.default(add_233)
        mul_375: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(add_233, sigmoid_51);  add_233 = sigmoid_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_130: "f32[8, 840, 7, 7]" = torch.ops.aten.convolution.default(mul_375, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 840);  mul_375 = arg258_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_234: "f32[840]" = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
        sqrt_108: "f32[840]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_108: "f32[840]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_376: "f32[840]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_865: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(mul_376, -1);  mul_376 = None
        unsqueeze_867: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_865);  convolution_130 = unsqueeze_865 = None
        mul_377: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_869: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_378: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(mul_377, unsqueeze_869);  mul_377 = unsqueeze_869 = None
        unsqueeze_870: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_871: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_235: "f32[8, 840, 7, 7]" = torch.ops.aten.add.Tensor(mul_378, unsqueeze_871);  mul_378 = unsqueeze_871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_23: "f32[8, 840, 1, 1]" = torch.ops.aten.mean.dim(add_235, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_131: "f32[8, 70, 1, 1]" = torch.ops.aten.convolution.default(mean_23, arg263_1, arg264_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg263_1 = arg264_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_236: "f32[70]" = torch.ops.aten.add.Tensor(arg266_1, 1e-05);  arg266_1 = None
        sqrt_109: "f32[70]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_109: "f32[70]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_379: "f32[70]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872: "f32[70, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_873: "f32[70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874: "f32[70, 1]" = torch.ops.aten.unsqueeze.default(mul_379, -1);  mul_379 = None
        unsqueeze_875: "f32[70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109: "f32[8, 70, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_873);  convolution_131 = unsqueeze_873 = None
        mul_380: "f32[8, 70, 1, 1]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876: "f32[70, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_877: "f32[70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_381: "f32[8, 70, 1, 1]" = torch.ops.aten.mul.Tensor(mul_380, unsqueeze_877);  mul_380 = unsqueeze_877 = None
        unsqueeze_878: "f32[70, 1]" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_879: "f32[70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_237: "f32[8, 70, 1, 1]" = torch.ops.aten.add.Tensor(mul_381, unsqueeze_879);  mul_381 = unsqueeze_879 = None
        relu_22: "f32[8, 70, 1, 1]" = torch.ops.aten.relu.default(add_237);  add_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_132: "f32[8, 840, 1, 1]" = torch.ops.aten.convolution.default(relu_22, arg269_1, arg270_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_22 = arg269_1 = arg270_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_52: "f32[8, 840, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_132);  convolution_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_382: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(add_235, sigmoid_52);  add_235 = sigmoid_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_28: "f32[8, 840, 7, 7]" = torch.ops.aten.clamp_min.default(mul_382, 0.0);  mul_382 = None
        clamp_max_28: "f32[8, 840, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6.0);  clamp_min_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_133: "f32[8, 151, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_28, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_28 = arg271_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_238: "f32[151]" = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_110: "f32[151]" = torch.ops.aten.sqrt.default(add_238);  add_238 = None
        reciprocal_110: "f32[151]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_383: "f32[151]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880: "f32[151, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_881: "f32[151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882: "f32[151, 1]" = torch.ops.aten.unsqueeze.default(mul_383, -1);  mul_383 = None
        unsqueeze_883: "f32[151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110: "f32[8, 151, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_881);  convolution_133 = unsqueeze_881 = None
        mul_384: "f32[8, 151, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884: "f32[151, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_885: "f32[151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_385: "f32[8, 151, 7, 7]" = torch.ops.aten.mul.Tensor(mul_384, unsqueeze_885);  mul_384 = unsqueeze_885 = None
        unsqueeze_886: "f32[151, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_887: "f32[151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_239: "f32[8, 151, 7, 7]" = torch.ops.aten.add.Tensor(mul_385, unsqueeze_887);  mul_385 = unsqueeze_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_74: "f32[8, 140, 7, 7]" = torch.ops.aten.slice.Tensor(add_239, 1, 0, 140)
        add_240: "f32[8, 140, 7, 7]" = torch.ops.aten.add.Tensor(slice_74, add_231);  slice_74 = add_231 = None
        slice_76: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(add_239, 1, 140, 9223372036854775807);  add_239 = None
        cat_18: "f32[8, 151, 7, 7]" = torch.ops.aten.cat.default([add_240, slice_76], 1);  add_240 = slice_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_134: "f32[8, 906, 7, 7]" = torch.ops.aten.convolution.default(cat_18, arg276_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg276_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_241: "f32[906]" = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
        sqrt_111: "f32[906]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_111: "f32[906]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_386: "f32[906]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_889: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(mul_386, -1);  mul_386 = None
        unsqueeze_891: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_889);  convolution_134 = unsqueeze_889 = None
        mul_387: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_893: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_388: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(mul_387, unsqueeze_893);  mul_387 = unsqueeze_893 = None
        unsqueeze_894: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_895: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_242: "f32[8, 906, 7, 7]" = torch.ops.aten.add.Tensor(mul_388, unsqueeze_895);  mul_388 = unsqueeze_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_53: "f32[8, 906, 7, 7]" = torch.ops.aten.sigmoid.default(add_242)
        mul_389: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(add_242, sigmoid_53);  add_242 = sigmoid_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_135: "f32[8, 906, 7, 7]" = torch.ops.aten.convolution.default(mul_389, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 906);  mul_389 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_243: "f32[906]" = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_112: "f32[906]" = torch.ops.aten.sqrt.default(add_243);  add_243 = None
        reciprocal_112: "f32[906]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_390: "f32[906]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_897: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
        unsqueeze_899: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_897);  convolution_135 = unsqueeze_897 = None
        mul_391: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_901: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_392: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_901);  mul_391 = unsqueeze_901 = None
        unsqueeze_902: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_903: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_244: "f32[8, 906, 7, 7]" = torch.ops.aten.add.Tensor(mul_392, unsqueeze_903);  mul_392 = unsqueeze_903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_24: "f32[8, 906, 1, 1]" = torch.ops.aten.mean.dim(add_244, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_136: "f32[8, 75, 1, 1]" = torch.ops.aten.convolution.default(mean_24, arg286_1, arg287_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg286_1 = arg287_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_245: "f32[75]" = torch.ops.aten.add.Tensor(arg289_1, 1e-05);  arg289_1 = None
        sqrt_113: "f32[75]" = torch.ops.aten.sqrt.default(add_245);  add_245 = None
        reciprocal_113: "f32[75]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_393: "f32[75]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904: "f32[75, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_905: "f32[75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906: "f32[75, 1]" = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
        unsqueeze_907: "f32[75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113: "f32[8, 75, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_905);  convolution_136 = unsqueeze_905 = None
        mul_394: "f32[8, 75, 1, 1]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f32[75, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_909: "f32[75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_395: "f32[8, 75, 1, 1]" = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_909);  mul_394 = unsqueeze_909 = None
        unsqueeze_910: "f32[75, 1]" = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_911: "f32[75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_246: "f32[8, 75, 1, 1]" = torch.ops.aten.add.Tensor(mul_395, unsqueeze_911);  mul_395 = unsqueeze_911 = None
        relu_23: "f32[8, 75, 1, 1]" = torch.ops.aten.relu.default(add_246);  add_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_137: "f32[8, 906, 1, 1]" = torch.ops.aten.convolution.default(relu_23, arg292_1, arg293_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg292_1 = arg293_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_54: "f32[8, 906, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_137);  convolution_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_396: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(add_244, sigmoid_54);  add_244 = sigmoid_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_29: "f32[8, 906, 7, 7]" = torch.ops.aten.clamp_min.default(mul_396, 0.0);  mul_396 = None
        clamp_max_29: "f32[8, 906, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6.0);  clamp_min_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_138: "f32[8, 162, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_29, arg294_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_29 = arg294_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_247: "f32[162]" = torch.ops.aten.add.Tensor(arg296_1, 1e-05);  arg296_1 = None
        sqrt_114: "f32[162]" = torch.ops.aten.sqrt.default(add_247);  add_247 = None
        reciprocal_114: "f32[162]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_397: "f32[162]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_913: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(mul_397, -1);  mul_397 = None
        unsqueeze_915: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114: "f32[8, 162, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_913);  convolution_138 = unsqueeze_913 = None
        mul_398: "f32[8, 162, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_917: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_399: "f32[8, 162, 7, 7]" = torch.ops.aten.mul.Tensor(mul_398, unsqueeze_917);  mul_398 = unsqueeze_917 = None
        unsqueeze_918: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_919: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_248: "f32[8, 162, 7, 7]" = torch.ops.aten.add.Tensor(mul_399, unsqueeze_919);  mul_399 = unsqueeze_919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_78: "f32[8, 151, 7, 7]" = torch.ops.aten.slice.Tensor(add_248, 1, 0, 151)
        add_249: "f32[8, 151, 7, 7]" = torch.ops.aten.add.Tensor(slice_78, cat_18);  slice_78 = cat_18 = None
        slice_80: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(add_248, 1, 151, 9223372036854775807);  add_248 = None
        cat_19: "f32[8, 162, 7, 7]" = torch.ops.aten.cat.default([add_249, slice_80], 1);  add_249 = slice_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_139: "f32[8, 972, 7, 7]" = torch.ops.aten.convolution.default(cat_19, arg299_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg299_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_250: "f32[972]" = torch.ops.aten.add.Tensor(arg301_1, 1e-05);  arg301_1 = None
        sqrt_115: "f32[972]" = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_115: "f32[972]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_400: "f32[972]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_921: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(mul_400, -1);  mul_400 = None
        unsqueeze_923: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_921);  convolution_139 = unsqueeze_921 = None
        mul_401: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_925: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_402: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(mul_401, unsqueeze_925);  mul_401 = unsqueeze_925 = None
        unsqueeze_926: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_927: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_251: "f32[8, 972, 7, 7]" = torch.ops.aten.add.Tensor(mul_402, unsqueeze_927);  mul_402 = unsqueeze_927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_55: "f32[8, 972, 7, 7]" = torch.ops.aten.sigmoid.default(add_251)
        mul_403: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(add_251, sigmoid_55);  add_251 = sigmoid_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_140: "f32[8, 972, 7, 7]" = torch.ops.aten.convolution.default(mul_403, arg304_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 972);  mul_403 = arg304_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_252: "f32[972]" = torch.ops.aten.add.Tensor(arg306_1, 1e-05);  arg306_1 = None
        sqrt_116: "f32[972]" = torch.ops.aten.sqrt.default(add_252);  add_252 = None
        reciprocal_116: "f32[972]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_404: "f32[972]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_929: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(mul_404, -1);  mul_404 = None
        unsqueeze_931: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_929);  convolution_140 = unsqueeze_929 = None
        mul_405: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
        unsqueeze_933: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_406: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_933);  mul_405 = unsqueeze_933 = None
        unsqueeze_934: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_935: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_253: "f32[8, 972, 7, 7]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_935);  mul_406 = unsqueeze_935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_25: "f32[8, 972, 1, 1]" = torch.ops.aten.mean.dim(add_253, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_141: "f32[8, 81, 1, 1]" = torch.ops.aten.convolution.default(mean_25, arg309_1, arg310_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg309_1 = arg310_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_254: "f32[81]" = torch.ops.aten.add.Tensor(arg312_1, 1e-05);  arg312_1 = None
        sqrt_117: "f32[81]" = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_117: "f32[81]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_407: "f32[81]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936: "f32[81, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_937: "f32[81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938: "f32[81, 1]" = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_939: "f32[81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117: "f32[8, 81, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_937);  convolution_141 = unsqueeze_937 = None
        mul_408: "f32[8, 81, 1, 1]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940: "f32[81, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_941: "f32[81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_409: "f32[8, 81, 1, 1]" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_941);  mul_408 = unsqueeze_941 = None
        unsqueeze_942: "f32[81, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_943: "f32[81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_255: "f32[8, 81, 1, 1]" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_943);  mul_409 = unsqueeze_943 = None
        relu_24: "f32[8, 81, 1, 1]" = torch.ops.aten.relu.default(add_255);  add_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_142: "f32[8, 972, 1, 1]" = torch.ops.aten.convolution.default(relu_24, arg315_1, arg316_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_24 = arg315_1 = arg316_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_56: "f32[8, 972, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_142);  convolution_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_410: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(add_253, sigmoid_56);  add_253 = sigmoid_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_30: "f32[8, 972, 7, 7]" = torch.ops.aten.clamp_min.default(mul_410, 0.0);  mul_410 = None
        clamp_max_30: "f32[8, 972, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6.0);  clamp_min_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_143: "f32[8, 174, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_30, arg317_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_30 = arg317_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_256: "f32[174]" = torch.ops.aten.add.Tensor(arg319_1, 1e-05);  arg319_1 = None
        sqrt_118: "f32[174]" = torch.ops.aten.sqrt.default(add_256);  add_256 = None
        reciprocal_118: "f32[174]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_411: "f32[174]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944: "f32[174, 1]" = torch.ops.aten.unsqueeze.default(arg318_1, -1);  arg318_1 = None
        unsqueeze_945: "f32[174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946: "f32[174, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_947: "f32[174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118: "f32[8, 174, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_945);  convolution_143 = unsqueeze_945 = None
        mul_412: "f32[8, 174, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948: "f32[174, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_949: "f32[174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_413: "f32[8, 174, 7, 7]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_949);  mul_412 = unsqueeze_949 = None
        unsqueeze_950: "f32[174, 1]" = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_951: "f32[174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_257: "f32[8, 174, 7, 7]" = torch.ops.aten.add.Tensor(mul_413, unsqueeze_951);  mul_413 = unsqueeze_951 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_82: "f32[8, 162, 7, 7]" = torch.ops.aten.slice.Tensor(add_257, 1, 0, 162)
        add_258: "f32[8, 162, 7, 7]" = torch.ops.aten.add.Tensor(slice_82, cat_19);  slice_82 = cat_19 = None
        slice_84: "f32[8, 12, 7, 7]" = torch.ops.aten.slice.Tensor(add_257, 1, 162, 9223372036854775807);  add_257 = None
        cat_20: "f32[8, 174, 7, 7]" = torch.ops.aten.cat.default([add_258, slice_84], 1);  add_258 = slice_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_144: "f32[8, 1044, 7, 7]" = torch.ops.aten.convolution.default(cat_20, arg322_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg322_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_259: "f32[1044]" = torch.ops.aten.add.Tensor(arg324_1, 1e-05);  arg324_1 = None
        sqrt_119: "f32[1044]" = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_119: "f32[1044]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_414: "f32[1044]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_953: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(mul_414, -1);  mul_414 = None
        unsqueeze_955: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_953);  convolution_144 = unsqueeze_953 = None
        mul_415: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_957: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_416: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(mul_415, unsqueeze_957);  mul_415 = unsqueeze_957 = None
        unsqueeze_958: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
        unsqueeze_959: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_260: "f32[8, 1044, 7, 7]" = torch.ops.aten.add.Tensor(mul_416, unsqueeze_959);  mul_416 = unsqueeze_959 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_57: "f32[8, 1044, 7, 7]" = torch.ops.aten.sigmoid.default(add_260)
        mul_417: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(add_260, sigmoid_57);  add_260 = sigmoid_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_145: "f32[8, 1044, 7, 7]" = torch.ops.aten.convolution.default(mul_417, arg327_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1044);  mul_417 = arg327_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_261: "f32[1044]" = torch.ops.aten.add.Tensor(arg329_1, 1e-05);  arg329_1 = None
        sqrt_120: "f32[1044]" = torch.ops.aten.sqrt.default(add_261);  add_261 = None
        reciprocal_120: "f32[1044]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_418: "f32[1044]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_961: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(mul_418, -1);  mul_418 = None
        unsqueeze_963: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_961);  convolution_145 = unsqueeze_961 = None
        mul_419: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_965: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_420: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_965);  mul_419 = unsqueeze_965 = None
        unsqueeze_966: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_967: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_262: "f32[8, 1044, 7, 7]" = torch.ops.aten.add.Tensor(mul_420, unsqueeze_967);  mul_420 = unsqueeze_967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_26: "f32[8, 1044, 1, 1]" = torch.ops.aten.mean.dim(add_262, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_146: "f32[8, 87, 1, 1]" = torch.ops.aten.convolution.default(mean_26, arg332_1, arg333_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg332_1 = arg333_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        add_263: "f32[87]" = torch.ops.aten.add.Tensor(arg335_1, 1e-05);  arg335_1 = None
        sqrt_121: "f32[87]" = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_121: "f32[87]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_421: "f32[87]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968: "f32[87, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_969: "f32[87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970: "f32[87, 1]" = torch.ops.aten.unsqueeze.default(mul_421, -1);  mul_421 = None
        unsqueeze_971: "f32[87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121: "f32[8, 87, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_969);  convolution_146 = unsqueeze_969 = None
        mul_422: "f32[8, 87, 1, 1]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972: "f32[87, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_973: "f32[87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_423: "f32[8, 87, 1, 1]" = torch.ops.aten.mul.Tensor(mul_422, unsqueeze_973);  mul_422 = unsqueeze_973 = None
        unsqueeze_974: "f32[87, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_975: "f32[87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_264: "f32[8, 87, 1, 1]" = torch.ops.aten.add.Tensor(mul_423, unsqueeze_975);  mul_423 = unsqueeze_975 = None
        relu_25: "f32[8, 87, 1, 1]" = torch.ops.aten.relu.default(add_264);  add_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_147: "f32[8, 1044, 1, 1]" = torch.ops.aten.convolution.default(relu_25, arg338_1, arg339_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_25 = arg338_1 = arg339_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_58: "f32[8, 1044, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_147);  convolution_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_424: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(add_262, sigmoid_58);  add_262 = sigmoid_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:87 in forward, code: x = self.act_dw(x)
        clamp_min_31: "f32[8, 1044, 7, 7]" = torch.ops.aten.clamp_min.default(mul_424, 0.0);  mul_424 = None
        clamp_max_31: "f32[8, 1044, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_31, 6.0);  clamp_min_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_148: "f32[8, 185, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_31, arg340_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_31 = arg340_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_265: "f32[185]" = torch.ops.aten.add.Tensor(arg342_1, 1e-05);  arg342_1 = None
        sqrt_122: "f32[185]" = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_122: "f32[185]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_425: "f32[185]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976: "f32[185, 1]" = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
        unsqueeze_977: "f32[185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978: "f32[185, 1]" = torch.ops.aten.unsqueeze.default(mul_425, -1);  mul_425 = None
        unsqueeze_979: "f32[185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122: "f32[8, 185, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_977);  convolution_148 = unsqueeze_977 = None
        mul_426: "f32[8, 185, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980: "f32[185, 1]" = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
        unsqueeze_981: "f32[185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_427: "f32[8, 185, 7, 7]" = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_981);  mul_426 = unsqueeze_981 = None
        unsqueeze_982: "f32[185, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_983: "f32[185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_266: "f32[8, 185, 7, 7]" = torch.ops.aten.add.Tensor(mul_427, unsqueeze_983);  mul_427 = unsqueeze_983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/rexnet.py:92 in forward, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        slice_86: "f32[8, 174, 7, 7]" = torch.ops.aten.slice.Tensor(add_266, 1, 0, 174)
        add_267: "f32[8, 174, 7, 7]" = torch.ops.aten.add.Tensor(slice_86, cat_20);  slice_86 = cat_20 = None
        slice_88: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(add_266, 1, 174, 9223372036854775807);  add_266 = None
        cat_21: "f32[8, 185, 7, 7]" = torch.ops.aten.cat.default([add_267, slice_88], 1);  add_267 = slice_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_149: "f32[8, 1280, 7, 7]" = torch.ops.aten.convolution.default(cat_21, arg345_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_21 = arg345_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_268: "f32[1280]" = torch.ops.aten.add.Tensor(arg347_1, 1e-05);  arg347_1 = None
        sqrt_123: "f32[1280]" = torch.ops.aten.sqrt.default(add_268);  add_268 = None
        reciprocal_123: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_428: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
        unsqueeze_985: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_428, -1);  mul_428 = None
        unsqueeze_987: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_985);  convolution_149 = unsqueeze_985 = None
        mul_429: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg348_1, -1);  arg348_1 = None
        unsqueeze_989: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_430: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_429, unsqueeze_989);  mul_429 = unsqueeze_989 = None
        unsqueeze_990: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_991: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_269: "f32[8, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_430, unsqueeze_991);  mul_430 = unsqueeze_991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_59: "f32[8, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(add_269)
        mul_431: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_269, sigmoid_59);  add_269 = sigmoid_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_27: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_431, [-1, -2], True);  mul_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 1280]" = torch.ops.aten.view.default(mean_27, [8, 1280]);  mean_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_1: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg351_1, view_1, permute_1);  arg351_1 = view_1 = permute_1 = None
        return (addmm_1,)
        