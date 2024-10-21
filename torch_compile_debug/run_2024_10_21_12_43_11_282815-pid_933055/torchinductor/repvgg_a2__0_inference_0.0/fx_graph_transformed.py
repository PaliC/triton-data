class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 1, 1]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64, 3, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[96, 64, 1, 1]", arg12_1: "f32[96]", arg13_1: "f32[96]", arg14_1: "f32[96]", arg15_1: "f32[96]", arg16_1: "f32[96, 64, 3, 3]", arg17_1: "f32[96]", arg18_1: "f32[96]", arg19_1: "f32[96]", arg20_1: "f32[96]", arg21_1: "f32[96]", arg22_1: "f32[96]", arg23_1: "f32[96]", arg24_1: "f32[96]", arg25_1: "f32[96, 96, 1, 1]", arg26_1: "f32[96]", arg27_1: "f32[96]", arg28_1: "f32[96]", arg29_1: "f32[96]", arg30_1: "f32[96, 96, 3, 3]", arg31_1: "f32[96]", arg32_1: "f32[96]", arg33_1: "f32[96]", arg34_1: "f32[96]", arg35_1: "f32[192, 96, 1, 1]", arg36_1: "f32[192]", arg37_1: "f32[192]", arg38_1: "f32[192]", arg39_1: "f32[192]", arg40_1: "f32[192, 96, 3, 3]", arg41_1: "f32[192]", arg42_1: "f32[192]", arg43_1: "f32[192]", arg44_1: "f32[192]", arg45_1: "f32[192]", arg46_1: "f32[192]", arg47_1: "f32[192]", arg48_1: "f32[192]", arg49_1: "f32[192, 192, 1, 1]", arg50_1: "f32[192]", arg51_1: "f32[192]", arg52_1: "f32[192]", arg53_1: "f32[192]", arg54_1: "f32[192, 192, 3, 3]", arg55_1: "f32[192]", arg56_1: "f32[192]", arg57_1: "f32[192]", arg58_1: "f32[192]", arg59_1: "f32[192]", arg60_1: "f32[192]", arg61_1: "f32[192]", arg62_1: "f32[192]", arg63_1: "f32[192, 192, 1, 1]", arg64_1: "f32[192]", arg65_1: "f32[192]", arg66_1: "f32[192]", arg67_1: "f32[192]", arg68_1: "f32[192, 192, 3, 3]", arg69_1: "f32[192]", arg70_1: "f32[192]", arg71_1: "f32[192]", arg72_1: "f32[192]", arg73_1: "f32[192]", arg74_1: "f32[192]", arg75_1: "f32[192]", arg76_1: "f32[192]", arg77_1: "f32[192, 192, 1, 1]", arg78_1: "f32[192]", arg79_1: "f32[192]", arg80_1: "f32[192]", arg81_1: "f32[192]", arg82_1: "f32[192, 192, 3, 3]", arg83_1: "f32[192]", arg84_1: "f32[192]", arg85_1: "f32[192]", arg86_1: "f32[192]", arg87_1: "f32[384, 192, 1, 1]", arg88_1: "f32[384]", arg89_1: "f32[384]", arg90_1: "f32[384]", arg91_1: "f32[384]", arg92_1: "f32[384, 192, 3, 3]", arg93_1: "f32[384]", arg94_1: "f32[384]", arg95_1: "f32[384]", arg96_1: "f32[384]", arg97_1: "f32[384]", arg98_1: "f32[384]", arg99_1: "f32[384]", arg100_1: "f32[384]", arg101_1: "f32[384, 384, 1, 1]", arg102_1: "f32[384]", arg103_1: "f32[384]", arg104_1: "f32[384]", arg105_1: "f32[384]", arg106_1: "f32[384, 384, 3, 3]", arg107_1: "f32[384]", arg108_1: "f32[384]", arg109_1: "f32[384]", arg110_1: "f32[384]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[384]", arg114_1: "f32[384]", arg115_1: "f32[384, 384, 1, 1]", arg116_1: "f32[384]", arg117_1: "f32[384]", arg118_1: "f32[384]", arg119_1: "f32[384]", arg120_1: "f32[384, 384, 3, 3]", arg121_1: "f32[384]", arg122_1: "f32[384]", arg123_1: "f32[384]", arg124_1: "f32[384]", arg125_1: "f32[384]", arg126_1: "f32[384]", arg127_1: "f32[384]", arg128_1: "f32[384]", arg129_1: "f32[384, 384, 1, 1]", arg130_1: "f32[384]", arg131_1: "f32[384]", arg132_1: "f32[384]", arg133_1: "f32[384]", arg134_1: "f32[384, 384, 3, 3]", arg135_1: "f32[384]", arg136_1: "f32[384]", arg137_1: "f32[384]", arg138_1: "f32[384]", arg139_1: "f32[384]", arg140_1: "f32[384]", arg141_1: "f32[384]", arg142_1: "f32[384]", arg143_1: "f32[384, 384, 1, 1]", arg144_1: "f32[384]", arg145_1: "f32[384]", arg146_1: "f32[384]", arg147_1: "f32[384]", arg148_1: "f32[384, 384, 3, 3]", arg149_1: "f32[384]", arg150_1: "f32[384]", arg151_1: "f32[384]", arg152_1: "f32[384]", arg153_1: "f32[384]", arg154_1: "f32[384]", arg155_1: "f32[384]", arg156_1: "f32[384]", arg157_1: "f32[384, 384, 1, 1]", arg158_1: "f32[384]", arg159_1: "f32[384]", arg160_1: "f32[384]", arg161_1: "f32[384]", arg162_1: "f32[384, 384, 3, 3]", arg163_1: "f32[384]", arg164_1: "f32[384]", arg165_1: "f32[384]", arg166_1: "f32[384]", arg167_1: "f32[384]", arg168_1: "f32[384]", arg169_1: "f32[384]", arg170_1: "f32[384]", arg171_1: "f32[384, 384, 1, 1]", arg172_1: "f32[384]", arg173_1: "f32[384]", arg174_1: "f32[384]", arg175_1: "f32[384]", arg176_1: "f32[384, 384, 3, 3]", arg177_1: "f32[384]", arg178_1: "f32[384]", arg179_1: "f32[384]", arg180_1: "f32[384]", arg181_1: "f32[384]", arg182_1: "f32[384]", arg183_1: "f32[384]", arg184_1: "f32[384]", arg185_1: "f32[384, 384, 1, 1]", arg186_1: "f32[384]", arg187_1: "f32[384]", arg188_1: "f32[384]", arg189_1: "f32[384]", arg190_1: "f32[384, 384, 3, 3]", arg191_1: "f32[384]", arg192_1: "f32[384]", arg193_1: "f32[384]", arg194_1: "f32[384]", arg195_1: "f32[384]", arg196_1: "f32[384]", arg197_1: "f32[384]", arg198_1: "f32[384]", arg199_1: "f32[384, 384, 1, 1]", arg200_1: "f32[384]", arg201_1: "f32[384]", arg202_1: "f32[384]", arg203_1: "f32[384]", arg204_1: "f32[384, 384, 3, 3]", arg205_1: "f32[384]", arg206_1: "f32[384]", arg207_1: "f32[384]", arg208_1: "f32[384]", arg209_1: "f32[384]", arg210_1: "f32[384]", arg211_1: "f32[384]", arg212_1: "f32[384]", arg213_1: "f32[384, 384, 1, 1]", arg214_1: "f32[384]", arg215_1: "f32[384]", arg216_1: "f32[384]", arg217_1: "f32[384]", arg218_1: "f32[384, 384, 3, 3]", arg219_1: "f32[384]", arg220_1: "f32[384]", arg221_1: "f32[384]", arg222_1: "f32[384]", arg223_1: "f32[384]", arg224_1: "f32[384]", arg225_1: "f32[384]", arg226_1: "f32[384]", arg227_1: "f32[384, 384, 1, 1]", arg228_1: "f32[384]", arg229_1: "f32[384]", arg230_1: "f32[384]", arg231_1: "f32[384]", arg232_1: "f32[384, 384, 3, 3]", arg233_1: "f32[384]", arg234_1: "f32[384]", arg235_1: "f32[384]", arg236_1: "f32[384]", arg237_1: "f32[384]", arg238_1: "f32[384]", arg239_1: "f32[384]", arg240_1: "f32[384]", arg241_1: "f32[384, 384, 1, 1]", arg242_1: "f32[384]", arg243_1: "f32[384]", arg244_1: "f32[384]", arg245_1: "f32[384]", arg246_1: "f32[384, 384, 3, 3]", arg247_1: "f32[384]", arg248_1: "f32[384]", arg249_1: "f32[384]", arg250_1: "f32[384]", arg251_1: "f32[384]", arg252_1: "f32[384]", arg253_1: "f32[384]", arg254_1: "f32[384]", arg255_1: "f32[384, 384, 1, 1]", arg256_1: "f32[384]", arg257_1: "f32[384]", arg258_1: "f32[384]", arg259_1: "f32[384]", arg260_1: "f32[384, 384, 3, 3]", arg261_1: "f32[384]", arg262_1: "f32[384]", arg263_1: "f32[384]", arg264_1: "f32[384]", arg265_1: "f32[384]", arg266_1: "f32[384]", arg267_1: "f32[384]", arg268_1: "f32[384]", arg269_1: "f32[384, 384, 1, 1]", arg270_1: "f32[384]", arg271_1: "f32[384]", arg272_1: "f32[384]", arg273_1: "f32[384]", arg274_1: "f32[384, 384, 3, 3]", arg275_1: "f32[384]", arg276_1: "f32[384]", arg277_1: "f32[384]", arg278_1: "f32[384]", arg279_1: "f32[1408, 384, 1, 1]", arg280_1: "f32[1408]", arg281_1: "f32[1408]", arg282_1: "f32[1408]", arg283_1: "f32[1408]", arg284_1: "f32[1408, 384, 3, 3]", arg285_1: "f32[1408]", arg286_1: "f32[1408]", arg287_1: "f32[1408]", arg288_1: "f32[1408]", arg289_1: "f32[1000, 1408]", arg290_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_44: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_488: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_489: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        sub_61: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_489);  convolution_44 = unsqueeze_489 = None
        add_161: "f32[64]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_61: "f32[64]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_61: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_183: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_490: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_491: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        mul_184: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_493: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_185: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
        unsqueeze_494: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_495: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_162: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_45: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_496: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_497: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        sub_62: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_497);  convolution_45 = unsqueeze_497 = None
        add_163: "f32[64]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_62: "f32[64]" = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_62: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_186: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_498: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_499: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        mul_187: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_501: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_188: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
        unsqueeze_502: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_503: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_164: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:545 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_165: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(add_162, add_164);  add_162 = add_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_22: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_165);  add_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_46: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(relu_22, arg11_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_504: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_505: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        sub_63: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_505);  convolution_46 = unsqueeze_505 = None
        add_166: "f32[96]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_63: "f32[96]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_63: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_189: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_506: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_507: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        mul_190: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_509: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_191: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
        unsqueeze_510: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_511: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_167: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_47: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(relu_22, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_512: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_513: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        sub_64: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_513);  convolution_47 = unsqueeze_513 = None
        add_168: "f32[96]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_64: "f32[96]" = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_64: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_192: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_514: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        unsqueeze_515: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        mul_193: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_517: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_194: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
        unsqueeze_518: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_519: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_169: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:545 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_170: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_167, add_169);  add_167 = add_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_23: "f32[8, 96, 56, 56]" = torch.ops.aten.relu.default(add_170);  add_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_48: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(relu_23, arg25_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_528: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_529: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        sub_66: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_529);  convolution_48 = unsqueeze_529 = None
        add_173: "f32[96]" = torch.ops.aten.add.Tensor(arg27_1, 1e-05);  arg27_1 = None
        sqrt_66: "f32[96]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_66: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_198: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_530: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_531: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        mul_199: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_533: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_200: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
        unsqueeze_534: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_535: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_174: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_49: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(relu_23, arg30_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_536: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_537: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        sub_67: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_537);  convolution_49 = unsqueeze_537 = None
        add_175: "f32[96]" = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_67: "f32[96]" = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_67: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_201: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_538: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
        unsqueeze_539: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        mul_202: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_541: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_203: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
        unsqueeze_542: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_543: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_176: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_177: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_174, add_176);  add_174 = add_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_520: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_521: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        sub_65: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(relu_23, unsqueeze_521);  relu_23 = unsqueeze_521 = None
        add_171: "f32[96]" = torch.ops.aten.add.Tensor(arg22_1, 1e-05);  arg22_1 = None
        sqrt_65: "f32[96]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_65: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_195: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_522: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_523: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        mul_196: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_525: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_197: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
        unsqueeze_526: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_527: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_172: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_178: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_177, add_172);  add_177 = add_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_24: "f32[8, 96, 56, 56]" = torch.ops.aten.relu.default(add_178);  add_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_50: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_24, arg35_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_544: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_545: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        sub_68: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_545);  convolution_50 = unsqueeze_545 = None
        add_179: "f32[192]" = torch.ops.aten.add.Tensor(arg37_1, 1e-05);  arg37_1 = None
        sqrt_68: "f32[192]" = torch.ops.aten.sqrt.default(add_179);  add_179 = None
        reciprocal_68: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_204: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_546: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
        unsqueeze_547: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        mul_205: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_549: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_206: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
        unsqueeze_550: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_551: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_180: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_51: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_24, arg40_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_24 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_552: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_553: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        sub_69: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_553);  convolution_51 = unsqueeze_553 = None
        add_181: "f32[192]" = torch.ops.aten.add.Tensor(arg42_1, 1e-05);  arg42_1 = None
        sqrt_69: "f32[192]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_69: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_207: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_554: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_555: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        mul_208: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_557: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_209: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
        unsqueeze_558: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_559: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_182: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:545 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_183: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_180, add_182);  add_180 = add_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_25: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_183);  add_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_52: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_25, arg49_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg49_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_568: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_569: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        sub_71: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_569);  convolution_52 = unsqueeze_569 = None
        add_186: "f32[192]" = torch.ops.aten.add.Tensor(arg51_1, 1e-05);  arg51_1 = None
        sqrt_71: "f32[192]" = torch.ops.aten.sqrt.default(add_186);  add_186 = None
        reciprocal_71: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_213: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_570: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
        unsqueeze_571: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        mul_214: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_573: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_215: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
        unsqueeze_574: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_575: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_187: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_53: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_25, arg54_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_576: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_577: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        sub_72: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_577);  convolution_53 = unsqueeze_577 = None
        add_188: "f32[192]" = torch.ops.aten.add.Tensor(arg56_1, 1e-05);  arg56_1 = None
        sqrt_72: "f32[192]" = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_72: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_216: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_578: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_579: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        mul_217: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_581: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_218: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
        unsqueeze_582: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_583: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_189: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_190: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_187, add_189);  add_187 = add_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_560: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_561: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        sub_70: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(relu_25, unsqueeze_561);  relu_25 = unsqueeze_561 = None
        add_184: "f32[192]" = torch.ops.aten.add.Tensor(arg46_1, 1e-05);  arg46_1 = None
        sqrt_70: "f32[192]" = torch.ops.aten.sqrt.default(add_184);  add_184 = None
        reciprocal_70: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_210: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_562: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_563: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        mul_211: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_565: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_212: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
        unsqueeze_566: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_567: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_185: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_191: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_190, add_185);  add_190 = add_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_26: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_191);  add_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_54: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_26, arg63_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_592: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_593: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        sub_74: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_593);  convolution_54 = unsqueeze_593 = None
        add_194: "f32[192]" = torch.ops.aten.add.Tensor(arg65_1, 1e-05);  arg65_1 = None
        sqrt_74: "f32[192]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_74: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_222: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_594: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
        unsqueeze_595: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        mul_223: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_597: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_224: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
        unsqueeze_598: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_599: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_195: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_55: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_26, arg68_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_600: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_601: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        sub_75: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_601);  convolution_55 = unsqueeze_601 = None
        add_196: "f32[192]" = torch.ops.aten.add.Tensor(arg70_1, 1e-05);  arg70_1 = None
        sqrt_75: "f32[192]" = torch.ops.aten.sqrt.default(add_196);  add_196 = None
        reciprocal_75: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_225: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_602: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
        unsqueeze_603: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        mul_226: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_605: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_227: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
        unsqueeze_606: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_607: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_197: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_198: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_195, add_197);  add_195 = add_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_584: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_585: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        sub_73: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(relu_26, unsqueeze_585);  relu_26 = unsqueeze_585 = None
        add_192: "f32[192]" = torch.ops.aten.add.Tensor(arg60_1, 1e-05);  arg60_1 = None
        sqrt_73: "f32[192]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_73: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_219: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_586: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_587: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        mul_220: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_589: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_221: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
        unsqueeze_590: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_591: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_193: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_199: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_198, add_193);  add_198 = add_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_27: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_199);  add_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_56: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_27, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_616: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_617: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        sub_77: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_617);  convolution_56 = unsqueeze_617 = None
        add_202: "f32[192]" = torch.ops.aten.add.Tensor(arg79_1, 1e-05);  arg79_1 = None
        sqrt_77: "f32[192]" = torch.ops.aten.sqrt.default(add_202);  add_202 = None
        reciprocal_77: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_231: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_618: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
        unsqueeze_619: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        mul_232: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_621: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_233: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
        unsqueeze_622: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_623: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_203: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_57: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_27, arg82_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_624: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_625: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        sub_78: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_625);  convolution_57 = unsqueeze_625 = None
        add_204: "f32[192]" = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_78: "f32[192]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_78: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_234: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_626: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
        unsqueeze_627: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        mul_235: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_629: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_236: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
        unsqueeze_630: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_631: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_205: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_206: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_203, add_205);  add_203 = add_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_608: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_609: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        sub_76: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(relu_27, unsqueeze_609);  relu_27 = unsqueeze_609 = None
        add_200: "f32[192]" = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_76: "f32[192]" = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_76: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_228: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_610: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_611: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        mul_229: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_613: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_230: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
        unsqueeze_614: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_615: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_201: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_207: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_206, add_201);  add_206 = add_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_28: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_207);  add_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_58: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_28, arg87_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg87_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_632: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_633: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        sub_79: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_633);  convolution_58 = unsqueeze_633 = None
        add_208: "f32[384]" = torch.ops.aten.add.Tensor(arg89_1, 1e-05);  arg89_1 = None
        sqrt_79: "f32[384]" = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_79: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_237: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_634: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_635: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        mul_238: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_637: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_239: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
        unsqueeze_638: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_639: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_209: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_59: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_28, arg92_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_28 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_640: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_641: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        sub_80: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_641);  convolution_59 = unsqueeze_641 = None
        add_210: "f32[384]" = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_80: "f32[384]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_80: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_240: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_642: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_643: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        mul_241: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_645: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_242: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
        unsqueeze_646: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_647: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_211: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:545 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_212: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_209, add_211);  add_209 = add_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_29: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_212);  add_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_60: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_29, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_656: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_657: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        sub_82: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_657);  convolution_60 = unsqueeze_657 = None
        add_215: "f32[384]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_82: "f32[384]" = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_82: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_246: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_658: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
        unsqueeze_659: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        mul_247: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_661: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_248: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_661);  mul_247 = unsqueeze_661 = None
        unsqueeze_662: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_663: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_216: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_663);  mul_248 = unsqueeze_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_61: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_29, arg106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_664: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_665: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        sub_83: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_665);  convolution_61 = unsqueeze_665 = None
        add_217: "f32[384]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_83: "f32[384]" = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_83: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_249: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_666: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
        unsqueeze_667: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        mul_250: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_669: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_251: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_669);  mul_250 = unsqueeze_669 = None
        unsqueeze_670: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_671: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_218: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_671);  mul_251 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_219: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_216, add_218);  add_216 = add_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_648: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_649: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        sub_81: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_29, unsqueeze_649);  relu_29 = unsqueeze_649 = None
        add_213: "f32[384]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_81: "f32[384]" = torch.ops.aten.sqrt.default(add_213);  add_213 = None
        reciprocal_81: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_243: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_650: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_651: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        mul_244: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_653: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_245: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
        unsqueeze_654: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_655: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_214: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_220: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_219, add_214);  add_219 = add_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_30: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_220);  add_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_62: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_30, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_680: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_681: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        sub_85: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_681);  convolution_62 = unsqueeze_681 = None
        add_223: "f32[384]" = torch.ops.aten.add.Tensor(arg117_1, 1e-05);  arg117_1 = None
        sqrt_85: "f32[384]" = torch.ops.aten.sqrt.default(add_223);  add_223 = None
        reciprocal_85: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_255: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_682: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_683: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        mul_256: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_685: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_257: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
        unsqueeze_686: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_687: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_224: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_63: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_30, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_688: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_689: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        sub_86: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_689);  convolution_63 = unsqueeze_689 = None
        add_225: "f32[384]" = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_86: "f32[384]" = torch.ops.aten.sqrt.default(add_225);  add_225 = None
        reciprocal_86: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_258: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_690: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
        unsqueeze_691: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        mul_259: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_693: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_260: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
        unsqueeze_694: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_695: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_226: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_227: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_224, add_226);  add_224 = add_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_672: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_673: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        sub_84: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_30, unsqueeze_673);  relu_30 = unsqueeze_673 = None
        add_221: "f32[384]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_84: "f32[384]" = torch.ops.aten.sqrt.default(add_221);  add_221 = None
        reciprocal_84: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_252: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_674: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
        unsqueeze_675: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        mul_253: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_677: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_254: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_677);  mul_253 = unsqueeze_677 = None
        unsqueeze_678: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_679: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_222: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_679);  mul_254 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_228: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_227, add_222);  add_227 = add_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_31: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_228);  add_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_64: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_31, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_704: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_705: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        sub_88: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_705);  convolution_64 = unsqueeze_705 = None
        add_231: "f32[384]" = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_88: "f32[384]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_88: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_264: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_706: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_707: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        mul_265: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_709: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_266: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
        unsqueeze_710: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_711: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_232: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_65: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_31, arg134_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_712: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_713: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        sub_89: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_713);  convolution_65 = unsqueeze_713 = None
        add_233: "f32[384]" = torch.ops.aten.add.Tensor(arg136_1, 1e-05);  arg136_1 = None
        sqrt_89: "f32[384]" = torch.ops.aten.sqrt.default(add_233);  add_233 = None
        reciprocal_89: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_267: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_714: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_715: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        mul_268: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_717: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_269: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
        unsqueeze_718: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_719: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_234: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_235: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_232, add_234);  add_232 = add_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_696: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_697: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        sub_87: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_31, unsqueeze_697);  relu_31 = unsqueeze_697 = None
        add_229: "f32[384]" = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
        sqrt_87: "f32[384]" = torch.ops.aten.sqrt.default(add_229);  add_229 = None
        reciprocal_87: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_261: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_698: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_699: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        mul_262: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_701: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_263: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
        unsqueeze_702: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_703: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_230: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_236: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_235, add_230);  add_235 = add_230 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_32: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_236);  add_236 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_66: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_32, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_728: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_729: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        sub_91: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_729);  convolution_66 = unsqueeze_729 = None
        add_239: "f32[384]" = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_91: "f32[384]" = torch.ops.aten.sqrt.default(add_239);  add_239 = None
        reciprocal_91: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_273: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_730: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_731: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        mul_274: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_733: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_275: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
        unsqueeze_734: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_735: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_240: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_67: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_32, arg148_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_736: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_737: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        sub_92: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_737);  convolution_67 = unsqueeze_737 = None
        add_241: "f32[384]" = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_92: "f32[384]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_92: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_276: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_738: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
        unsqueeze_739: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        mul_277: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_741: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_278: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
        unsqueeze_742: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_743: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_242: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_243: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_240, add_242);  add_240 = add_242 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_720: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_721: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        sub_90: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_32, unsqueeze_721);  relu_32 = unsqueeze_721 = None
        add_237: "f32[384]" = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_90: "f32[384]" = torch.ops.aten.sqrt.default(add_237);  add_237 = None
        reciprocal_90: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_270: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_722: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_723: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        mul_271: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_725: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_272: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
        unsqueeze_726: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_727: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_238: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_244: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_243, add_238);  add_243 = add_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_33: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_244);  add_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_68: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_33, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_752: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_753: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        sub_94: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_753);  convolution_68 = unsqueeze_753 = None
        add_247: "f32[384]" = torch.ops.aten.add.Tensor(arg159_1, 1e-05);  arg159_1 = None
        sqrt_94: "f32[384]" = torch.ops.aten.sqrt.default(add_247);  add_247 = None
        reciprocal_94: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_282: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_754: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
        unsqueeze_755: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        mul_283: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_757: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_284: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
        unsqueeze_758: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_759: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_248: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_69: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_33, arg162_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_760: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_761: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        sub_95: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_761);  convolution_69 = unsqueeze_761 = None
        add_249: "f32[384]" = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_95: "f32[384]" = torch.ops.aten.sqrt.default(add_249);  add_249 = None
        reciprocal_95: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_285: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_762: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
        unsqueeze_763: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        mul_286: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_765: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_287: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
        unsqueeze_766: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_767: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_250: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_251: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_248, add_250);  add_248 = add_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_744: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_745: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        sub_93: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_33, unsqueeze_745);  relu_33 = unsqueeze_745 = None
        add_245: "f32[384]" = torch.ops.aten.add.Tensor(arg154_1, 1e-05);  arg154_1 = None
        sqrt_93: "f32[384]" = torch.ops.aten.sqrt.default(add_245);  add_245 = None
        reciprocal_93: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_279: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_746: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
        unsqueeze_747: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        mul_280: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_749: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_281: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
        unsqueeze_750: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_751: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_246: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_252: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_251, add_246);  add_251 = add_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_34: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_252);  add_252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_70: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_34, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_776: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_777: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        sub_97: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_777);  convolution_70 = unsqueeze_777 = None
        add_255: "f32[384]" = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_97: "f32[384]" = torch.ops.aten.sqrt.default(add_255);  add_255 = None
        reciprocal_97: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_291: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_778: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
        unsqueeze_779: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        mul_292: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_781: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_293: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
        unsqueeze_782: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_783: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_256: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_71: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_34, arg176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_784: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_785: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        sub_98: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_785);  convolution_71 = unsqueeze_785 = None
        add_257: "f32[384]" = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_98: "f32[384]" = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_98: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_294: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_786: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
        unsqueeze_787: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        mul_295: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_789: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_296: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
        unsqueeze_790: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_791: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_258: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_259: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_256, add_258);  add_256 = add_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_768: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_769: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        sub_96: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_34, unsqueeze_769);  relu_34 = unsqueeze_769 = None
        add_253: "f32[384]" = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_96: "f32[384]" = torch.ops.aten.sqrt.default(add_253);  add_253 = None
        reciprocal_96: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_288: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_770: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_771: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        mul_289: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_773: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_290: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
        unsqueeze_774: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_775: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_254: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_260: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_259, add_254);  add_259 = add_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_35: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_260);  add_260 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_72: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_35, arg185_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg185_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_800: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_801: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        sub_100: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_801);  convolution_72 = unsqueeze_801 = None
        add_263: "f32[384]" = torch.ops.aten.add.Tensor(arg187_1, 1e-05);  arg187_1 = None
        sqrt_100: "f32[384]" = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_100: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_300: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_802: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_803: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        mul_301: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_805: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_302: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
        unsqueeze_806: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_807: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_264: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_73: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_35, arg190_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_808: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_809: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        sub_101: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_809);  convolution_73 = unsqueeze_809 = None
        add_265: "f32[384]" = torch.ops.aten.add.Tensor(arg192_1, 1e-05);  arg192_1 = None
        sqrt_101: "f32[384]" = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_101: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_303: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_810: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_811: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        mul_304: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_813: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_305: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
        unsqueeze_814: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_815: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_266: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_267: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_264, add_266);  add_264 = add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_792: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_793: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        sub_99: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_35, unsqueeze_793);  relu_35 = unsqueeze_793 = None
        add_261: "f32[384]" = torch.ops.aten.add.Tensor(arg182_1, 1e-05);  arg182_1 = None
        sqrt_99: "f32[384]" = torch.ops.aten.sqrt.default(add_261);  add_261 = None
        reciprocal_99: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_297: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_794: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
        unsqueeze_795: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        mul_298: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_797: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_299: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
        unsqueeze_798: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_799: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_262: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_268: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_267, add_262);  add_267 = add_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_36: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_268);  add_268 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_74: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_36, arg199_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg199_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_824: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_825: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        sub_103: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_825);  convolution_74 = unsqueeze_825 = None
        add_271: "f32[384]" = torch.ops.aten.add.Tensor(arg201_1, 1e-05);  arg201_1 = None
        sqrt_103: "f32[384]" = torch.ops.aten.sqrt.default(add_271);  add_271 = None
        reciprocal_103: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_309: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_826: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_827: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        mul_310: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_829: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_311: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
        unsqueeze_830: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_831: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_272: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_75: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_36, arg204_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_832: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_833: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        sub_104: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_833);  convolution_75 = unsqueeze_833 = None
        add_273: "f32[384]" = torch.ops.aten.add.Tensor(arg206_1, 1e-05);  arg206_1 = None
        sqrt_104: "f32[384]" = torch.ops.aten.sqrt.default(add_273);  add_273 = None
        reciprocal_104: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_312: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_834: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
        unsqueeze_835: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        mul_313: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_837: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_314: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_837);  mul_313 = unsqueeze_837 = None
        unsqueeze_838: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_839: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_274: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_839);  mul_314 = unsqueeze_839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_275: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_272, add_274);  add_272 = add_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_816: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_817: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        sub_102: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_36, unsqueeze_817);  relu_36 = unsqueeze_817 = None
        add_269: "f32[384]" = torch.ops.aten.add.Tensor(arg196_1, 1e-05);  arg196_1 = None
        sqrt_102: "f32[384]" = torch.ops.aten.sqrt.default(add_269);  add_269 = None
        reciprocal_102: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_306: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_818: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
        unsqueeze_819: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        mul_307: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_821: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_308: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
        unsqueeze_822: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_823: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_270: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_276: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_275, add_270);  add_275 = add_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_37: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_276);  add_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_76: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_37, arg213_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg213_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_848: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_849: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        sub_106: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_849);  convolution_76 = unsqueeze_849 = None
        add_279: "f32[384]" = torch.ops.aten.add.Tensor(arg215_1, 1e-05);  arg215_1 = None
        sqrt_106: "f32[384]" = torch.ops.aten.sqrt.default(add_279);  add_279 = None
        reciprocal_106: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_318: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_850: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_318, -1);  mul_318 = None
        unsqueeze_851: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        mul_319: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_853: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_320: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_853);  mul_319 = unsqueeze_853 = None
        unsqueeze_854: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_855: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_280: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_855);  mul_320 = unsqueeze_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_77: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_37, arg218_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg218_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_856: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_857: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        sub_107: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_857);  convolution_77 = unsqueeze_857 = None
        add_281: "f32[384]" = torch.ops.aten.add.Tensor(arg220_1, 1e-05);  arg220_1 = None
        sqrt_107: "f32[384]" = torch.ops.aten.sqrt.default(add_281);  add_281 = None
        reciprocal_107: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_321: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_858: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
        unsqueeze_859: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        mul_322: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_861: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_323: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_861);  mul_322 = unsqueeze_861 = None
        unsqueeze_862: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_863: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_282: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_863);  mul_323 = unsqueeze_863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_283: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_280, add_282);  add_280 = add_282 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_840: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_841: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        sub_105: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_37, unsqueeze_841);  relu_37 = unsqueeze_841 = None
        add_277: "f32[384]" = torch.ops.aten.add.Tensor(arg210_1, 1e-05);  arg210_1 = None
        sqrt_105: "f32[384]" = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_105: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_315: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_842: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
        unsqueeze_843: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        mul_316: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_845: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_317: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_845);  mul_316 = unsqueeze_845 = None
        unsqueeze_846: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_847: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_278: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_847);  mul_317 = unsqueeze_847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_284: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_283, add_278);  add_283 = add_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_38: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_284);  add_284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_78: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_38, arg227_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg227_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_872: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_873: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        sub_109: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_873);  convolution_78 = unsqueeze_873 = None
        add_287: "f32[384]" = torch.ops.aten.add.Tensor(arg229_1, 1e-05);  arg229_1 = None
        sqrt_109: "f32[384]" = torch.ops.aten.sqrt.default(add_287);  add_287 = None
        reciprocal_109: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_327: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_874: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_875: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        mul_328: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_877: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_329: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_877);  mul_328 = unsqueeze_877 = None
        unsqueeze_878: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_879: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_288: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_879);  mul_329 = unsqueeze_879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_79: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_38, arg232_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg232_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_880: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_881: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        sub_110: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_881);  convolution_79 = unsqueeze_881 = None
        add_289: "f32[384]" = torch.ops.aten.add.Tensor(arg234_1, 1e-05);  arg234_1 = None
        sqrt_110: "f32[384]" = torch.ops.aten.sqrt.default(add_289);  add_289 = None
        reciprocal_110: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_330: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_882: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_883: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        mul_331: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_885: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_332: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_885);  mul_331 = unsqueeze_885 = None
        unsqueeze_886: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_887: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_290: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_887);  mul_332 = unsqueeze_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_291: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_288, add_290);  add_288 = add_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_864: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_865: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        sub_108: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_38, unsqueeze_865);  relu_38 = unsqueeze_865 = None
        add_285: "f32[384]" = torch.ops.aten.add.Tensor(arg224_1, 1e-05);  arg224_1 = None
        sqrt_108: "f32[384]" = torch.ops.aten.sqrt.default(add_285);  add_285 = None
        reciprocal_108: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_324: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_866: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
        unsqueeze_867: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        mul_325: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_869: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_326: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_869);  mul_325 = unsqueeze_869 = None
        unsqueeze_870: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_871: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_286: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_871);  mul_326 = unsqueeze_871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_292: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_291, add_286);  add_291 = add_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_39: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_292);  add_292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_80: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_39, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_896: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_897: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        sub_112: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_897);  convolution_80 = unsqueeze_897 = None
        add_295: "f32[384]" = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_112: "f32[384]" = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_112: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_336: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_898: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_899: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        mul_337: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_901: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_338: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
        unsqueeze_902: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_903: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_296: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_81: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_39, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_904: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_905: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        sub_113: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_905);  convolution_81 = unsqueeze_905 = None
        add_297: "f32[384]" = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_113: "f32[384]" = torch.ops.aten.sqrt.default(add_297);  add_297 = None
        reciprocal_113: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_339: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_906: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_907: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        mul_340: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_909: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_341: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
        unsqueeze_910: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_911: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_298: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_299: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_296, add_298);  add_296 = add_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_888: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_889: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        sub_111: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_39, unsqueeze_889);  relu_39 = unsqueeze_889 = None
        add_293: "f32[384]" = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_111: "f32[384]" = torch.ops.aten.sqrt.default(add_293);  add_293 = None
        reciprocal_111: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_333: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_890: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_891: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        mul_334: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_893: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_335: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
        unsqueeze_894: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_895: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_294: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_300: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_299, add_294);  add_299 = add_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_40: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_300);  add_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_82: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_40, arg255_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg255_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_920: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_921: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        sub_115: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_921);  convolution_82 = unsqueeze_921 = None
        add_303: "f32[384]" = torch.ops.aten.add.Tensor(arg257_1, 1e-05);  arg257_1 = None
        sqrt_115: "f32[384]" = torch.ops.aten.sqrt.default(add_303);  add_303 = None
        reciprocal_115: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_345: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_922: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
        unsqueeze_923: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        mul_346: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_925: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_347: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
        unsqueeze_926: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_927: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_304: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_83: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_40, arg260_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg260_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_928: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_929: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        sub_116: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_929);  convolution_83 = unsqueeze_929 = None
        add_305: "f32[384]" = torch.ops.aten.add.Tensor(arg262_1, 1e-05);  arg262_1 = None
        sqrt_116: "f32[384]" = torch.ops.aten.sqrt.default(add_305);  add_305 = None
        reciprocal_116: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_348: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_930: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_931: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        mul_349: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_933: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_350: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
        unsqueeze_934: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_935: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_306: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_307: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_304, add_306);  add_304 = add_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_912: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_913: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        sub_114: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_40, unsqueeze_913);  relu_40 = unsqueeze_913 = None
        add_301: "f32[384]" = torch.ops.aten.add.Tensor(arg252_1, 1e-05);  arg252_1 = None
        sqrt_114: "f32[384]" = torch.ops.aten.sqrt.default(add_301);  add_301 = None
        reciprocal_114: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_342: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_914: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_915: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        mul_343: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_917: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_344: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
        unsqueeze_918: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_919: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_302: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_308: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_307, add_302);  add_307 = add_302 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_41: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_308);  add_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_84: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_41, arg269_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg269_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_944: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_945: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        sub_118: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_945);  convolution_84 = unsqueeze_945 = None
        add_311: "f32[384]" = torch.ops.aten.add.Tensor(arg271_1, 1e-05);  arg271_1 = None
        sqrt_118: "f32[384]" = torch.ops.aten.sqrt.default(add_311);  add_311 = None
        reciprocal_118: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_354: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_946: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
        unsqueeze_947: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        mul_355: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_949: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_356: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
        unsqueeze_950: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_951: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_312: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_85: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_41, arg274_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg274_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_952: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_953: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        sub_119: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_953);  convolution_85 = unsqueeze_953 = None
        add_313: "f32[384]" = torch.ops.aten.add.Tensor(arg276_1, 1e-05);  arg276_1 = None
        sqrt_119: "f32[384]" = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_119: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_357: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_954: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_955: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        mul_358: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_957: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_359: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
        unsqueeze_958: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg278_1, -1);  arg278_1 = None
        unsqueeze_959: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_314: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:548 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_315: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_312, add_314);  add_312 = add_314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_936: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_937: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        sub_117: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_41, unsqueeze_937);  relu_41 = unsqueeze_937 = None
        add_309: "f32[384]" = torch.ops.aten.add.Tensor(arg266_1, 1e-05);  arg266_1 = None
        sqrt_117: "f32[384]" = torch.ops.aten.sqrt.default(add_309);  add_309 = None
        reciprocal_117: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_351: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_938: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_939: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        mul_352: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_941: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_353: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
        unsqueeze_942: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_943: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_310: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:550 in forward, code: x += identity
        add_316: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_315, add_310);  add_315 = add_310 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_42: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_316);  add_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_86: "f32[8, 1408, 7, 7]" = torch.ops.aten.convolution.default(relu_42, arg279_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg279_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_960: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_961: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        sub_120: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_961);  convolution_86 = unsqueeze_961 = None
        add_317: "f32[1408]" = torch.ops.aten.add.Tensor(arg281_1, 1e-05);  arg281_1 = None
        sqrt_120: "f32[1408]" = torch.ops.aten.sqrt.default(add_317);  add_317 = None
        reciprocal_120: "f32[1408]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_360: "f32[1408]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_962: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_963: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        mul_361: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_965: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_362: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
        unsqueeze_966: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_967: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_318: "f32[8, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_87: "f32[8, 1408, 7, 7]" = torch.ops.aten.convolution.default(relu_42, arg284_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_42 = arg284_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_968: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_969: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        sub_121: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_969);  convolution_87 = unsqueeze_969 = None
        add_319: "f32[1408]" = torch.ops.aten.add.Tensor(arg286_1, 1e-05);  arg286_1 = None
        sqrt_121: "f32[1408]" = torch.ops.aten.sqrt.default(add_319);  add_319 = None
        reciprocal_121: "f32[1408]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_363: "f32[1408]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_970: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
        unsqueeze_971: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        mul_364: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_973: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_365: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_973);  mul_364 = unsqueeze_973 = None
        unsqueeze_974: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_975: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_320: "f32[8, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_365, unsqueeze_975);  mul_365 = unsqueeze_975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:545 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_321: "f32[8, 1408, 7, 7]" = torch.ops.aten.add.Tensor(add_318, add_320);  add_318 = add_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:552 in forward, code: return self.act(x)
        relu_43: "f32[8, 1408, 7, 7]" = torch.ops.aten.relu.default(add_321);  add_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 1408, 1, 1]" = torch.ops.aten.mean.dim(relu_43, [-1, -2], True);  relu_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 1408]" = torch.ops.aten.reshape.default(mean_1, [8, 1408]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_1: "f32[1408, 1000]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg290_1, view_1, permute_1);  arg290_1 = view_1 = permute_1 = None
        return (addmm_1,)
        