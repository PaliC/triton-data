class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[16]", arg3_1: "f32[16]", arg4_1: "f32[16]", arg5_1: "f32[16]", arg6_1: "f32[16, 1, 3, 3]", arg7_1: "f32[16]", arg8_1: "f32[16]", arg9_1: "f32[16]", arg10_1: "f32[16]", arg11_1: "f32[16, 16, 1, 1]", arg12_1: "f32[16]", arg13_1: "f32[16]", arg14_1: "f32[16]", arg15_1: "f32[16]", arg16_1: "f32[64, 16, 1, 1]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[64, 1, 3, 3]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[24, 64, 1, 1]", arg27_1: "f32[24]", arg28_1: "f32[24]", arg29_1: "f32[24]", arg30_1: "f32[24]", arg31_1: "f32[72, 24, 1, 1]", arg32_1: "f32[72]", arg33_1: "f32[72]", arg34_1: "f32[72]", arg35_1: "f32[72]", arg36_1: "f32[72, 1, 3, 3]", arg37_1: "f32[72]", arg38_1: "f32[72]", arg39_1: "f32[72]", arg40_1: "f32[72]", arg41_1: "f32[24, 72, 1, 1]", arg42_1: "f32[24]", arg43_1: "f32[24]", arg44_1: "f32[24]", arg45_1: "f32[24]", arg46_1: "f32[72, 24, 1, 1]", arg47_1: "f32[72]", arg48_1: "f32[72]", arg49_1: "f32[72]", arg50_1: "f32[72]", arg51_1: "f32[72, 1, 5, 5]", arg52_1: "f32[72]", arg53_1: "f32[72]", arg54_1: "f32[72]", arg55_1: "f32[72]", arg56_1: "f32[24, 72, 1, 1]", arg57_1: "f32[24]", arg58_1: "f32[72, 24, 1, 1]", arg59_1: "f32[72]", arg60_1: "f32[40, 72, 1, 1]", arg61_1: "f32[40]", arg62_1: "f32[40]", arg63_1: "f32[40]", arg64_1: "f32[40]", arg65_1: "f32[120, 40, 1, 1]", arg66_1: "f32[120]", arg67_1: "f32[120]", arg68_1: "f32[120]", arg69_1: "f32[120]", arg70_1: "f32[120, 1, 5, 5]", arg71_1: "f32[120]", arg72_1: "f32[120]", arg73_1: "f32[120]", arg74_1: "f32[120]", arg75_1: "f32[32, 120, 1, 1]", arg76_1: "f32[32]", arg77_1: "f32[120, 32, 1, 1]", arg78_1: "f32[120]", arg79_1: "f32[40, 120, 1, 1]", arg80_1: "f32[40]", arg81_1: "f32[40]", arg82_1: "f32[40]", arg83_1: "f32[40]", arg84_1: "f32[120, 40, 1, 1]", arg85_1: "f32[120]", arg86_1: "f32[120]", arg87_1: "f32[120]", arg88_1: "f32[120]", arg89_1: "f32[120, 1, 5, 5]", arg90_1: "f32[120]", arg91_1: "f32[120]", arg92_1: "f32[120]", arg93_1: "f32[120]", arg94_1: "f32[32, 120, 1, 1]", arg95_1: "f32[32]", arg96_1: "f32[120, 32, 1, 1]", arg97_1: "f32[120]", arg98_1: "f32[40, 120, 1, 1]", arg99_1: "f32[40]", arg100_1: "f32[40]", arg101_1: "f32[40]", arg102_1: "f32[40]", arg103_1: "f32[240, 40, 1, 1]", arg104_1: "f32[240]", arg105_1: "f32[240]", arg106_1: "f32[240]", arg107_1: "f32[240]", arg108_1: "f32[240, 1, 3, 3]", arg109_1: "f32[240]", arg110_1: "f32[240]", arg111_1: "f32[240]", arg112_1: "f32[240]", arg113_1: "f32[80, 240, 1, 1]", arg114_1: "f32[80]", arg115_1: "f32[80]", arg116_1: "f32[80]", arg117_1: "f32[80]", arg118_1: "f32[200, 80, 1, 1]", arg119_1: "f32[200]", arg120_1: "f32[200]", arg121_1: "f32[200]", arg122_1: "f32[200]", arg123_1: "f32[200, 1, 3, 3]", arg124_1: "f32[200]", arg125_1: "f32[200]", arg126_1: "f32[200]", arg127_1: "f32[200]", arg128_1: "f32[80, 200, 1, 1]", arg129_1: "f32[80]", arg130_1: "f32[80]", arg131_1: "f32[80]", arg132_1: "f32[80]", arg133_1: "f32[184, 80, 1, 1]", arg134_1: "f32[184]", arg135_1: "f32[184]", arg136_1: "f32[184]", arg137_1: "f32[184]", arg138_1: "f32[184, 1, 3, 3]", arg139_1: "f32[184]", arg140_1: "f32[184]", arg141_1: "f32[184]", arg142_1: "f32[184]", arg143_1: "f32[80, 184, 1, 1]", arg144_1: "f32[80]", arg145_1: "f32[80]", arg146_1: "f32[80]", arg147_1: "f32[80]", arg148_1: "f32[184, 80, 1, 1]", arg149_1: "f32[184]", arg150_1: "f32[184]", arg151_1: "f32[184]", arg152_1: "f32[184]", arg153_1: "f32[184, 1, 3, 3]", arg154_1: "f32[184]", arg155_1: "f32[184]", arg156_1: "f32[184]", arg157_1: "f32[184]", arg158_1: "f32[80, 184, 1, 1]", arg159_1: "f32[80]", arg160_1: "f32[80]", arg161_1: "f32[80]", arg162_1: "f32[80]", arg163_1: "f32[480, 80, 1, 1]", arg164_1: "f32[480]", arg165_1: "f32[480]", arg166_1: "f32[480]", arg167_1: "f32[480]", arg168_1: "f32[480, 1, 3, 3]", arg169_1: "f32[480]", arg170_1: "f32[480]", arg171_1: "f32[480]", arg172_1: "f32[480]", arg173_1: "f32[120, 480, 1, 1]", arg174_1: "f32[120]", arg175_1: "f32[480, 120, 1, 1]", arg176_1: "f32[480]", arg177_1: "f32[112, 480, 1, 1]", arg178_1: "f32[112]", arg179_1: "f32[112]", arg180_1: "f32[112]", arg181_1: "f32[112]", arg182_1: "f32[672, 112, 1, 1]", arg183_1: "f32[672]", arg184_1: "f32[672]", arg185_1: "f32[672]", arg186_1: "f32[672]", arg187_1: "f32[672, 1, 3, 3]", arg188_1: "f32[672]", arg189_1: "f32[672]", arg190_1: "f32[672]", arg191_1: "f32[672]", arg192_1: "f32[168, 672, 1, 1]", arg193_1: "f32[168]", arg194_1: "f32[672, 168, 1, 1]", arg195_1: "f32[672]", arg196_1: "f32[112, 672, 1, 1]", arg197_1: "f32[112]", arg198_1: "f32[112]", arg199_1: "f32[112]", arg200_1: "f32[112]", arg201_1: "f32[672, 112, 1, 1]", arg202_1: "f32[672]", arg203_1: "f32[672]", arg204_1: "f32[672]", arg205_1: "f32[672]", arg206_1: "f32[672, 1, 5, 5]", arg207_1: "f32[672]", arg208_1: "f32[672]", arg209_1: "f32[672]", arg210_1: "f32[672]", arg211_1: "f32[168, 672, 1, 1]", arg212_1: "f32[168]", arg213_1: "f32[672, 168, 1, 1]", arg214_1: "f32[672]", arg215_1: "f32[160, 672, 1, 1]", arg216_1: "f32[160]", arg217_1: "f32[160]", arg218_1: "f32[160]", arg219_1: "f32[160]", arg220_1: "f32[960, 160, 1, 1]", arg221_1: "f32[960]", arg222_1: "f32[960]", arg223_1: "f32[960]", arg224_1: "f32[960]", arg225_1: "f32[960, 1, 5, 5]", arg226_1: "f32[960]", arg227_1: "f32[960]", arg228_1: "f32[960]", arg229_1: "f32[960]", arg230_1: "f32[240, 960, 1, 1]", arg231_1: "f32[240]", arg232_1: "f32[960, 240, 1, 1]", arg233_1: "f32[960]", arg234_1: "f32[160, 960, 1, 1]", arg235_1: "f32[160]", arg236_1: "f32[160]", arg237_1: "f32[160]", arg238_1: "f32[160]", arg239_1: "f32[960, 160, 1, 1]", arg240_1: "f32[960]", arg241_1: "f32[960]", arg242_1: "f32[960]", arg243_1: "f32[960]", arg244_1: "f32[960, 1, 5, 5]", arg245_1: "f32[960]", arg246_1: "f32[960]", arg247_1: "f32[960]", arg248_1: "f32[960]", arg249_1: "f32[240, 960, 1, 1]", arg250_1: "f32[240]", arg251_1: "f32[960, 240, 1, 1]", arg252_1: "f32[960]", arg253_1: "f32[160, 960, 1, 1]", arg254_1: "f32[160]", arg255_1: "f32[160]", arg256_1: "f32[160]", arg257_1: "f32[160]", arg258_1: "f32[960, 160, 1, 1]", arg259_1: "f32[960]", arg260_1: "f32[960]", arg261_1: "f32[960]", arg262_1: "f32[960]", arg263_1: "f32[1280, 960, 1, 1]", arg264_1: "f32[1280]", arg265_1: "f32[1000, 1280]", arg266_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:251 in forward_features, code: x = self.conv_stem(x)
        convolution_63: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_131: "f32[16]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_46: "f32[16]" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
        reciprocal_46: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_167: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_369: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_167, -1);  mul_167 = None
        unsqueeze_371: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_46: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_369);  convolution_63 = unsqueeze_369 = None
        mul_168: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_373: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_169: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_373);  mul_168 = unsqueeze_373 = None
        unsqueeze_374: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_375: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_132: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_375);  mul_169 = unsqueeze_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_133: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_132, 3)
        clamp_min_29: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_133, 0);  add_133 = None
        clamp_max_29: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
        mul_170: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_132, clamp_max_29);  add_132 = clamp_max_29 = None
        div_29: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_170, 6);  mul_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_64: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(div_29, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_134: "f32[16]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_47: "f32[16]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_47: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_171: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_377: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_379: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_47: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_377);  convolution_64 = unsqueeze_377 = None
        mul_172: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_381: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_173: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_381);  mul_172 = unsqueeze_381 = None
        unsqueeze_382: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_383: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_135: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_383);  mul_173 = unsqueeze_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_19: "f32[8, 16, 112, 112]" = torch.ops.aten.relu.default(add_135);  add_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_65: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(relu_19, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_19 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_136: "f32[16]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_48: "f32[16]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_48: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_174: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_385: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_387: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_48: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_385);  convolution_65 = unsqueeze_385 = None
        mul_175: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_389: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_176: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_389);  mul_175 = unsqueeze_389 = None
        unsqueeze_390: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_391: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_137: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_391);  mul_176 = unsqueeze_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:197 in forward, code: x = self.drop_path(x) + shortcut
        add_138: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_137, div_29);  add_137 = div_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_66: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(add_138, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_138 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_139: "f32[64]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_49: "f32[64]" = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_49: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_177: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_393: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
        unsqueeze_395: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_49: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_393);  convolution_66 = unsqueeze_393 = None
        mul_178: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_397: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_179: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_397);  mul_178 = unsqueeze_397 = None
        unsqueeze_398: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_399: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_140: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_399);  mul_179 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_20: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_140);  add_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_67: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_20, arg21_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  relu_20 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_141: "f32[64]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_50: "f32[64]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_50: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_180: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_401: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
        unsqueeze_403: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_50: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_401);  convolution_67 = unsqueeze_401 = None
        mul_181: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_405: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_182: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_405);  mul_181 = unsqueeze_405 = None
        unsqueeze_406: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_407: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_142: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_407);  mul_182 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_21: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_142);  add_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_68: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_21, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_143: "f32[24]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_51: "f32[24]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_51: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_183: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_409: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_411: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_51: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_409);  convolution_68 = unsqueeze_409 = None
        mul_184: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_413: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_185: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_413);  mul_184 = unsqueeze_413 = None
        unsqueeze_414: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_415: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_144: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_415);  mul_185 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_69: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(add_144, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_145: "f32[72]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_52: "f32[72]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_52: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_186: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_417: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_419: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_52: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_417);  convolution_69 = unsqueeze_417 = None
        mul_187: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_421: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_188: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_421);  mul_187 = unsqueeze_421 = None
        unsqueeze_422: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_423: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_146: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_423);  mul_188 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_22: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_146);  add_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_70: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(relu_22, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72);  relu_22 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_147: "f32[72]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_53: "f32[72]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_53: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_189: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_425: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_427: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_53: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_425);  convolution_70 = unsqueeze_425 = None
        mul_190: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_429: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_191: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_429);  mul_190 = unsqueeze_429 = None
        unsqueeze_430: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_431: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_148: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_431);  mul_191 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_23: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_148);  add_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_71: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_23, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_149: "f32[24]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_54: "f32[24]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_54: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_192: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_433: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        unsqueeze_435: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_54: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_433);  convolution_71 = unsqueeze_433 = None
        mul_193: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
        unsqueeze_436: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_437: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_194: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_437);  mul_193 = unsqueeze_437 = None
        unsqueeze_438: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_439: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_150: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_439);  mul_194 = unsqueeze_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_151: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_150, add_144);  add_150 = add_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_72: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(add_151, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_151 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_152: "f32[72]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_55: "f32[72]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_55: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_195: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_441: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_443: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_55: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_441);  convolution_72 = unsqueeze_441 = None
        mul_196: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
        unsqueeze_444: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_445: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_197: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_445);  mul_196 = unsqueeze_445 = None
        unsqueeze_446: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_447: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_153: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_447);  mul_197 = unsqueeze_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_24: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_153);  add_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_73: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_24, arg51_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72);  relu_24 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_154: "f32[72]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_56: "f32[72]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_56: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_198: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_449: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_451: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_56: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_449);  convolution_73 = unsqueeze_449 = None
        mul_199: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
        unsqueeze_452: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_453: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_200: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_453);  mul_199 = unsqueeze_453 = None
        unsqueeze_454: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_455: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_155: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_455);  mul_200 = unsqueeze_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_25: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_155);  add_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "f32[8, 72, 1, 1]" = torch.ops.aten.mean.dim(relu_25, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_74: "f32[8, 24, 1, 1]" = torch.ops.aten.convolution.default(mean_9, arg56_1, arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg56_1 = arg57_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_26: "f32[8, 24, 1, 1]" = torch.ops.aten.relu.default(convolution_74);  convolution_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_75: "f32[8, 72, 1, 1]" = torch.ops.aten.convolution.default(relu_26, arg58_1, arg59_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_26 = arg58_1 = arg59_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_156: "f32[8, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_75, 3);  convolution_75 = None
        clamp_min_30: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_156, 0);  add_156 = None
        clamp_max_30: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6);  clamp_min_30 = None
        div_30: "f32[8, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_30, 6);  clamp_max_30 = None
        mul_201: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(relu_25, div_30);  relu_25 = div_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_76: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_201, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_201 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_157: "f32[40]" = torch.ops.aten.add.Tensor(arg62_1, 1e-05);  arg62_1 = None
        sqrt_57: "f32[40]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_57: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_202: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_457: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_202, -1);  mul_202 = None
        unsqueeze_459: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_57: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_457);  convolution_76 = unsqueeze_457 = None
        mul_203: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
        unsqueeze_460: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_461: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_204: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_461);  mul_203 = unsqueeze_461 = None
        unsqueeze_462: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_463: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_158: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_463);  mul_204 = unsqueeze_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_77: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_158, arg65_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_159: "f32[120]" = torch.ops.aten.add.Tensor(arg67_1, 1e-05);  arg67_1 = None
        sqrt_58: "f32[120]" = torch.ops.aten.sqrt.default(add_159);  add_159 = None
        reciprocal_58: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_205: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_465: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_205, -1);  mul_205 = None
        unsqueeze_467: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_58: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_465);  convolution_77 = unsqueeze_465 = None
        mul_206: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_469: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_207: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_206, unsqueeze_469);  mul_206 = unsqueeze_469 = None
        unsqueeze_470: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_471: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_160: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_207, unsqueeze_471);  mul_207 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_27: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_160);  add_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_78: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_27, arg70_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  relu_27 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_161: "f32[120]" = torch.ops.aten.add.Tensor(arg72_1, 1e-05);  arg72_1 = None
        sqrt_59: "f32[120]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_59: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_208: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_473: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
        unsqueeze_475: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_59: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_473);  convolution_78 = unsqueeze_473 = None
        mul_209: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_477: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_210: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_477);  mul_209 = unsqueeze_477 = None
        unsqueeze_478: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_479: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_162: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_479);  mul_210 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_28: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_162);  add_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_28, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_79: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_10, arg75_1, arg76_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg75_1 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_29: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_79);  convolution_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_80: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_29, arg77_1, arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_29 = arg77_1 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_163: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_80, 3);  convolution_80 = None
        clamp_min_31: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_163, 0);  add_163 = None
        clamp_max_31: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_31, 6);  clamp_min_31 = None
        div_31: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_31, 6);  clamp_max_31 = None
        mul_211: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(relu_28, div_31);  relu_28 = div_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_81: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_211, arg79_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_211 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_164: "f32[40]" = torch.ops.aten.add.Tensor(arg81_1, 1e-05);  arg81_1 = None
        sqrt_60: "f32[40]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_60: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_212: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_481: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_212, -1);  mul_212 = None
        unsqueeze_483: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_60: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_481);  convolution_81 = unsqueeze_481 = None
        mul_213: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_485: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_214: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_485);  mul_213 = unsqueeze_485 = None
        unsqueeze_486: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_487: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_165: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_487);  mul_214 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_166: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_165, add_158);  add_165 = add_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_82: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_166, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_167: "f32[120]" = torch.ops.aten.add.Tensor(arg86_1, 1e-05);  arg86_1 = None
        sqrt_61: "f32[120]" = torch.ops.aten.sqrt.default(add_167);  add_167 = None
        reciprocal_61: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_215: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_489: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_215, -1);  mul_215 = None
        unsqueeze_491: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_61: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_489);  convolution_82 = unsqueeze_489 = None
        mul_216: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_493: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_217: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_216, unsqueeze_493);  mul_216 = unsqueeze_493 = None
        unsqueeze_494: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_495: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_168: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_217, unsqueeze_495);  mul_217 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_30: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_168);  add_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_83: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_30, arg89_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  relu_30 = arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_169: "f32[120]" = torch.ops.aten.add.Tensor(arg91_1, 1e-05);  arg91_1 = None
        sqrt_62: "f32[120]" = torch.ops.aten.sqrt.default(add_169);  add_169 = None
        reciprocal_62: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_218: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_497: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_218, -1);  mul_218 = None
        unsqueeze_499: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_497);  convolution_83 = unsqueeze_497 = None
        mul_219: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_501: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_220: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_219, unsqueeze_501);  mul_219 = unsqueeze_501 = None
        unsqueeze_502: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_503: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_170: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_220, unsqueeze_503);  mul_220 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_31: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_170);  add_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_31, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_84: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_11, arg94_1, arg95_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg94_1 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_32: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_84);  convolution_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_85: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_32, arg96_1, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_32 = arg96_1 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_171: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_85, 3);  convolution_85 = None
        clamp_min_32: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_171, 0);  add_171 = None
        clamp_max_32: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_32, 6);  clamp_min_32 = None
        div_32: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_32, 6);  clamp_max_32 = None
        mul_221: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(relu_31, div_32);  relu_31 = div_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_86: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_221, arg98_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_221 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_172: "f32[40]" = torch.ops.aten.add.Tensor(arg100_1, 1e-05);  arg100_1 = None
        sqrt_63: "f32[40]" = torch.ops.aten.sqrt.default(add_172);  add_172 = None
        reciprocal_63: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_222: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_505: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
        unsqueeze_507: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_505);  convolution_86 = unsqueeze_505 = None
        mul_223: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_509: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_224: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_509);  mul_223 = unsqueeze_509 = None
        unsqueeze_510: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_511: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_173: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_511);  mul_224 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_174: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_173, add_166);  add_173 = add_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_87: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(add_174, arg103_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_174 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_175: "f32[240]" = torch.ops.aten.add.Tensor(arg105_1, 1e-05);  arg105_1 = None
        sqrt_64: "f32[240]" = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_64: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_225: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_513: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
        unsqueeze_515: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_513);  convolution_87 = unsqueeze_513 = None
        mul_226: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_517: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_227: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_517);  mul_226 = unsqueeze_517 = None
        unsqueeze_518: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_519: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_176: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_519);  mul_227 = unsqueeze_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_177: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(add_176, 3)
        clamp_min_33: "f32[8, 240, 28, 28]" = torch.ops.aten.clamp_min.default(add_177, 0);  add_177 = None
        clamp_max_33: "f32[8, 240, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_33, 6);  clamp_min_33 = None
        mul_228: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_176, clamp_max_33);  add_176 = clamp_max_33 = None
        div_33: "f32[8, 240, 28, 28]" = torch.ops.aten.div.Tensor(mul_228, 6);  mul_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_88: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(div_33, arg108_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  div_33 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_178: "f32[240]" = torch.ops.aten.add.Tensor(arg110_1, 1e-05);  arg110_1 = None
        sqrt_65: "f32[240]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_65: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_229: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_521: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_229, -1);  mul_229 = None
        unsqueeze_523: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_521);  convolution_88 = unsqueeze_521 = None
        mul_230: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_525: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_231: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_525);  mul_230 = unsqueeze_525 = None
        unsqueeze_526: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_527: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_179: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_231, unsqueeze_527);  mul_231 = unsqueeze_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_180: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(add_179, 3)
        clamp_min_34: "f32[8, 240, 14, 14]" = torch.ops.aten.clamp_min.default(add_180, 0);  add_180 = None
        clamp_max_34: "f32[8, 240, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_34, 6);  clamp_min_34 = None
        mul_232: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_179, clamp_max_34);  add_179 = clamp_max_34 = None
        div_34: "f32[8, 240, 14, 14]" = torch.ops.aten.div.Tensor(mul_232, 6);  mul_232 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_89: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(div_34, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_34 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_181: "f32[80]" = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_66: "f32[80]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_66: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_233: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_529: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_233, -1);  mul_233 = None
        unsqueeze_531: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_529);  convolution_89 = unsqueeze_529 = None
        mul_234: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_533: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_235: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_533);  mul_234 = unsqueeze_533 = None
        unsqueeze_534: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_535: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_182: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_535);  mul_235 = unsqueeze_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_90: "f32[8, 200, 14, 14]" = torch.ops.aten.convolution.default(add_182, arg118_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_183: "f32[200]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_67: "f32[200]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_67: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_236: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_537: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_236, -1);  mul_236 = None
        unsqueeze_539: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_537);  convolution_90 = unsqueeze_537 = None
        mul_237: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_541: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_238: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_541);  mul_237 = unsqueeze_541 = None
        unsqueeze_542: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_543: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_184: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_543);  mul_238 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_185: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_184, 3)
        clamp_min_35: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_185, 0);  add_185 = None
        clamp_max_35: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_35, 6);  clamp_min_35 = None
        mul_239: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_184, clamp_max_35);  add_184 = clamp_max_35 = None
        div_35: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_239, 6);  mul_239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_91: "f32[8, 200, 14, 14]" = torch.ops.aten.convolution.default(div_35, arg123_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 200);  div_35 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_186: "f32[200]" = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_68: "f32[200]" = torch.ops.aten.sqrt.default(add_186);  add_186 = None
        reciprocal_68: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_240: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_545: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_547: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_545);  convolution_91 = unsqueeze_545 = None
        mul_241: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_549: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_242: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_549);  mul_241 = unsqueeze_549 = None
        unsqueeze_550: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_551: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_187: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_551);  mul_242 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_188: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_187, 3)
        clamp_min_36: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_188, 0);  add_188 = None
        clamp_max_36: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_36, 6);  clamp_min_36 = None
        mul_243: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_187, clamp_max_36);  add_187 = clamp_max_36 = None
        div_36: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_243, 6);  mul_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_92: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(div_36, arg128_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_36 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_189: "f32[80]" = torch.ops.aten.add.Tensor(arg130_1, 1e-05);  arg130_1 = None
        sqrt_69: "f32[80]" = torch.ops.aten.sqrt.default(add_189);  add_189 = None
        reciprocal_69: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_244: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_553: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_244, -1);  mul_244 = None
        unsqueeze_555: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_553);  convolution_92 = unsqueeze_553 = None
        mul_245: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_557: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_246: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_557);  mul_245 = unsqueeze_557 = None
        unsqueeze_558: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_559: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_190: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_559);  mul_246 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_191: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_190, add_182);  add_190 = add_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_93: "f32[8, 184, 14, 14]" = torch.ops.aten.convolution.default(add_191, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_192: "f32[184]" = torch.ops.aten.add.Tensor(arg135_1, 1e-05);  arg135_1 = None
        sqrt_70: "f32[184]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_70: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_247: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_561: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
        unsqueeze_563: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_561);  convolution_93 = unsqueeze_561 = None
        mul_248: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_565: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_249: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_565);  mul_248 = unsqueeze_565 = None
        unsqueeze_566: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_567: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_193: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_249, unsqueeze_567);  mul_249 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_194: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_193, 3)
        clamp_min_37: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_194, 0);  add_194 = None
        clamp_max_37: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_37, 6);  clamp_min_37 = None
        mul_250: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_193, clamp_max_37);  add_193 = clamp_max_37 = None
        div_37: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_250, 6);  mul_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_94: "f32[8, 184, 14, 14]" = torch.ops.aten.convolution.default(div_37, arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184);  div_37 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_195: "f32[184]" = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_71: "f32[184]" = torch.ops.aten.sqrt.default(add_195);  add_195 = None
        reciprocal_71: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_251: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_569: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_251, -1);  mul_251 = None
        unsqueeze_571: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_569);  convolution_94 = unsqueeze_569 = None
        mul_252: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_573: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_253: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_573);  mul_252 = unsqueeze_573 = None
        unsqueeze_574: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_575: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_196: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_575);  mul_253 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_197: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_196, 3)
        clamp_min_38: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_197, 0);  add_197 = None
        clamp_max_38: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_38, 6);  clamp_min_38 = None
        mul_254: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_196, clamp_max_38);  add_196 = clamp_max_38 = None
        div_38: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_254, 6);  mul_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_95: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(div_38, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_38 = arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_198: "f32[80]" = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_72: "f32[80]" = torch.ops.aten.sqrt.default(add_198);  add_198 = None
        reciprocal_72: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_255: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_577: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_579: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_577);  convolution_95 = unsqueeze_577 = None
        mul_256: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_581: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_257: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_581);  mul_256 = unsqueeze_581 = None
        unsqueeze_582: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_583: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_199: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_583);  mul_257 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_200: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_199, add_191);  add_199 = add_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_96: "f32[8, 184, 14, 14]" = torch.ops.aten.convolution.default(add_200, arg148_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_201: "f32[184]" = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_73: "f32[184]" = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_73: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_258: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_585: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
        unsqueeze_587: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_585);  convolution_96 = unsqueeze_585 = None
        mul_259: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_589: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_260: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_589);  mul_259 = unsqueeze_589 = None
        unsqueeze_590: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_591: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_202: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_591);  mul_260 = unsqueeze_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_203: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_202, 3)
        clamp_min_39: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_203, 0);  add_203 = None
        clamp_max_39: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_39, 6);  clamp_min_39 = None
        mul_261: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_202, clamp_max_39);  add_202 = clamp_max_39 = None
        div_39: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_261, 6);  mul_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_97: "f32[8, 184, 14, 14]" = torch.ops.aten.convolution.default(div_39, arg153_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184);  div_39 = arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_204: "f32[184]" = torch.ops.aten.add.Tensor(arg155_1, 1e-05);  arg155_1 = None
        sqrt_74: "f32[184]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_74: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_262: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_593: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_262, -1);  mul_262 = None
        unsqueeze_595: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_593);  convolution_97 = unsqueeze_593 = None
        mul_263: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_597: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_264: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_597);  mul_263 = unsqueeze_597 = None
        unsqueeze_598: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_599: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_205: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_264, unsqueeze_599);  mul_264 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_206: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_205, 3)
        clamp_min_40: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_206, 0);  add_206 = None
        clamp_max_40: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_40, 6);  clamp_min_40 = None
        mul_265: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_205, clamp_max_40);  add_205 = clamp_max_40 = None
        div_40: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_265, 6);  mul_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_98: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(div_40, arg158_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_40 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_207: "f32[80]" = torch.ops.aten.add.Tensor(arg160_1, 1e-05);  arg160_1 = None
        sqrt_75: "f32[80]" = torch.ops.aten.sqrt.default(add_207);  add_207 = None
        reciprocal_75: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_266: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_601: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_266, -1);  mul_266 = None
        unsqueeze_603: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_601);  convolution_98 = unsqueeze_601 = None
        mul_267: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_605: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_268: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_267, unsqueeze_605);  mul_267 = unsqueeze_605 = None
        unsqueeze_606: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_607: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_208: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_268, unsqueeze_607);  mul_268 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_209: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_208, add_200);  add_208 = add_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_99: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_209, arg163_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_209 = arg163_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_210: "f32[480]" = torch.ops.aten.add.Tensor(arg165_1, 1e-05);  arg165_1 = None
        sqrt_76: "f32[480]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_76: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_269: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_609: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_269, -1);  mul_269 = None
        unsqueeze_611: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_609);  convolution_99 = unsqueeze_609 = None
        mul_270: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_613: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_271: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_613);  mul_270 = unsqueeze_613 = None
        unsqueeze_614: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_615: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_211: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_271, unsqueeze_615);  mul_271 = unsqueeze_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_212: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_211, 3)
        clamp_min_41: "f32[8, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_212, 0);  add_212 = None
        clamp_max_41: "f32[8, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_41, 6);  clamp_min_41 = None
        mul_272: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_211, clamp_max_41);  add_211 = clamp_max_41 = None
        div_41: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_272, 6);  mul_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_100: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(div_41, arg168_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  div_41 = arg168_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_213: "f32[480]" = torch.ops.aten.add.Tensor(arg170_1, 1e-05);  arg170_1 = None
        sqrt_77: "f32[480]" = torch.ops.aten.sqrt.default(add_213);  add_213 = None
        reciprocal_77: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_273: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_617: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_619: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_617);  convolution_100 = unsqueeze_617 = None
        mul_274: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_621: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_275: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_621);  mul_274 = unsqueeze_621 = None
        unsqueeze_622: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_623: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_214: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_623);  mul_275 = unsqueeze_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_215: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_214, 3)
        clamp_min_42: "f32[8, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_215, 0);  add_215 = None
        clamp_max_42: "f32[8, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_42, 6);  clamp_min_42 = None
        mul_276: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_214, clamp_max_42);  add_214 = clamp_max_42 = None
        div_42: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_276, 6);  mul_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_12: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(div_42, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_101: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(mean_12, arg173_1, arg174_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_12 = arg173_1 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_33: "f32[8, 120, 1, 1]" = torch.ops.aten.relu.default(convolution_101);  convolution_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_102: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(relu_33, arg175_1, arg176_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_33 = arg175_1 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_216: "f32[8, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_102, 3);  convolution_102 = None
        clamp_min_43: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_216, 0);  add_216 = None
        clamp_max_43: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_43, 6);  clamp_min_43 = None
        div_43: "f32[8, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_43, 6);  clamp_max_43 = None
        mul_277: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(div_42, div_43);  div_42 = div_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_103: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_277, arg177_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_277 = arg177_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_217: "f32[112]" = torch.ops.aten.add.Tensor(arg179_1, 1e-05);  arg179_1 = None
        sqrt_78: "f32[112]" = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_78: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_278: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_625: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_278, -1);  mul_278 = None
        unsqueeze_627: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_625);  convolution_103 = unsqueeze_625 = None
        mul_279: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_629: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_280: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_279, unsqueeze_629);  mul_279 = unsqueeze_629 = None
        unsqueeze_630: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_631: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_218: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_280, unsqueeze_631);  mul_280 = unsqueeze_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_104: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_218, arg182_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_219: "f32[672]" = torch.ops.aten.add.Tensor(arg184_1, 1e-05);  arg184_1 = None
        sqrt_79: "f32[672]" = torch.ops.aten.sqrt.default(add_219);  add_219 = None
        reciprocal_79: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_281: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_633: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_281, -1);  mul_281 = None
        unsqueeze_635: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_633);  convolution_104 = unsqueeze_633 = None
        mul_282: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_637: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_283: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_282, unsqueeze_637);  mul_282 = unsqueeze_637 = None
        unsqueeze_638: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_639: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_220: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_639);  mul_283 = unsqueeze_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_221: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_220, 3)
        clamp_min_44: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_221, 0);  add_221 = None
        clamp_max_44: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_44, 6);  clamp_min_44 = None
        mul_284: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_220, clamp_max_44);  add_220 = clamp_max_44 = None
        div_44: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_284, 6);  mul_284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_105: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(div_44, arg187_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 672);  div_44 = arg187_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_222: "f32[672]" = torch.ops.aten.add.Tensor(arg189_1, 1e-05);  arg189_1 = None
        sqrt_80: "f32[672]" = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_80: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_285: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_641: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
        unsqueeze_643: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_641);  convolution_105 = unsqueeze_641 = None
        mul_286: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_645: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_287: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_645);  mul_286 = unsqueeze_645 = None
        unsqueeze_646: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_647: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_223: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_647);  mul_287 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_224: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_223, 3)
        clamp_min_45: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_224, 0);  add_224 = None
        clamp_max_45: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_45, 6);  clamp_min_45 = None
        mul_288: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_223, clamp_max_45);  add_223 = clamp_max_45 = None
        div_45: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_288, 6);  mul_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_13: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(div_45, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_106: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_13, arg192_1, arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg192_1 = arg193_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_34: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_106);  convolution_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_107: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_34, arg194_1, arg195_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_34 = arg194_1 = arg195_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_225: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_107, 3);  convolution_107 = None
        clamp_min_46: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_225, 0);  add_225 = None
        clamp_max_46: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_46, 6);  clamp_min_46 = None
        div_46: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_46, 6);  clamp_max_46 = None
        mul_289: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(div_45, div_46);  div_45 = div_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_108: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_289, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_289 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_226: "f32[112]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_81: "f32[112]" = torch.ops.aten.sqrt.default(add_226);  add_226 = None
        reciprocal_81: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_290: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_649: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_290, -1);  mul_290 = None
        unsqueeze_651: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_649);  convolution_108 = unsqueeze_649 = None
        mul_291: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_653: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_292: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_291, unsqueeze_653);  mul_291 = unsqueeze_653 = None
        unsqueeze_654: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_655: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_227: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_655);  mul_292 = unsqueeze_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_228: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_227, add_218);  add_227 = add_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_109: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_228, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_228 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_229: "f32[672]" = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_82: "f32[672]" = torch.ops.aten.sqrt.default(add_229);  add_229 = None
        reciprocal_82: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_293: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_657: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_293, -1);  mul_293 = None
        unsqueeze_659: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_657);  convolution_109 = unsqueeze_657 = None
        mul_294: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_661: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_295: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_661);  mul_294 = unsqueeze_661 = None
        unsqueeze_662: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_663: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_230: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_295, unsqueeze_663);  mul_295 = unsqueeze_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_231: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_230, 3)
        clamp_min_47: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_231, 0);  add_231 = None
        clamp_max_47: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_47, 6);  clamp_min_47 = None
        mul_296: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_230, clamp_max_47);  add_230 = clamp_max_47 = None
        div_47: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_296, 6);  mul_296 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_110: "f32[8, 672, 7, 7]" = torch.ops.aten.convolution.default(div_47, arg206_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672);  div_47 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_232: "f32[672]" = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_83: "f32[672]" = torch.ops.aten.sqrt.default(add_232);  add_232 = None
        reciprocal_83: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_297: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_665: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
        unsqueeze_667: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_665);  convolution_110 = unsqueeze_665 = None
        mul_298: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_669: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_299: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_669);  mul_298 = unsqueeze_669 = None
        unsqueeze_670: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_671: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_233: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_671);  mul_299 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_234: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(add_233, 3)
        clamp_min_48: "f32[8, 672, 7, 7]" = torch.ops.aten.clamp_min.default(add_234, 0);  add_234 = None
        clamp_max_48: "f32[8, 672, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_48, 6);  clamp_min_48 = None
        mul_300: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_233, clamp_max_48);  add_233 = clamp_max_48 = None
        div_48: "f32[8, 672, 7, 7]" = torch.ops.aten.div.Tensor(mul_300, 6);  mul_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(div_48, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_111: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_14, arg211_1, arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg211_1 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_35: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_111);  convolution_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_112: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_35, arg213_1, arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_35 = arg213_1 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_235: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_112, 3);  convolution_112 = None
        clamp_min_49: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_235, 0);  add_235 = None
        clamp_max_49: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_49, 6);  clamp_min_49 = None
        div_49: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_49, 6);  clamp_max_49 = None
        mul_301: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(div_48, div_49);  div_48 = div_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_113: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_301, arg215_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_301 = arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_236: "f32[160]" = torch.ops.aten.add.Tensor(arg217_1, 1e-05);  arg217_1 = None
        sqrt_84: "f32[160]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_84: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_302: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_673: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_302, -1);  mul_302 = None
        unsqueeze_675: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_673);  convolution_113 = unsqueeze_673 = None
        mul_303: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_677: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_304: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_303, unsqueeze_677);  mul_303 = unsqueeze_677 = None
        unsqueeze_678: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_679: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_237: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_304, unsqueeze_679);  mul_304 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_114: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(add_237, arg220_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg220_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_238: "f32[960]" = torch.ops.aten.add.Tensor(arg222_1, 1e-05);  arg222_1 = None
        sqrt_85: "f32[960]" = torch.ops.aten.sqrt.default(add_238);  add_238 = None
        reciprocal_85: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_305: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_681: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_305, -1);  mul_305 = None
        unsqueeze_683: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_681);  convolution_114 = unsqueeze_681 = None
        mul_306: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_685: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_307: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_306, unsqueeze_685);  mul_306 = unsqueeze_685 = None
        unsqueeze_686: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_687: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_239: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_687);  mul_307 = unsqueeze_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_240: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_239, 3)
        clamp_min_50: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_240, 0);  add_240 = None
        clamp_max_50: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_50, 6);  clamp_min_50 = None
        mul_308: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_239, clamp_max_50);  add_239 = clamp_max_50 = None
        div_50: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_308, 6);  mul_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_115: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(div_50, arg225_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960);  div_50 = arg225_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_241: "f32[960]" = torch.ops.aten.add.Tensor(arg227_1, 1e-05);  arg227_1 = None
        sqrt_86: "f32[960]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_86: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_309: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_689: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_691: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_689);  convolution_115 = unsqueeze_689 = None
        mul_310: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_693: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_311: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_693);  mul_310 = unsqueeze_693 = None
        unsqueeze_694: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_695: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_242: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_695);  mul_311 = unsqueeze_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_243: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_242, 3)
        clamp_min_51: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_243, 0);  add_243 = None
        clamp_max_51: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_51, 6);  clamp_min_51 = None
        mul_312: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_242, clamp_max_51);  add_242 = clamp_max_51 = None
        div_51: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_312, 6);  mul_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_15: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(div_51, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_116: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_15, arg230_1, arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg230_1 = arg231_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_36: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_116);  convolution_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_117: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_36, arg232_1, arg233_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_36 = arg232_1 = arg233_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_244: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_117, 3);  convolution_117 = None
        clamp_min_52: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_244, 0);  add_244 = None
        clamp_max_52: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_52, 6);  clamp_min_52 = None
        div_52: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_52, 6);  clamp_max_52 = None
        mul_313: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_51, div_52);  div_51 = div_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_118: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_313, arg234_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_313 = arg234_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_245: "f32[160]" = torch.ops.aten.add.Tensor(arg236_1, 1e-05);  arg236_1 = None
        sqrt_87: "f32[160]" = torch.ops.aten.sqrt.default(add_245);  add_245 = None
        reciprocal_87: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_314: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_697: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_314, -1);  mul_314 = None
        unsqueeze_699: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_697);  convolution_118 = unsqueeze_697 = None
        mul_315: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_701: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_316: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_701);  mul_315 = unsqueeze_701 = None
        unsqueeze_702: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_703: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_246: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_316, unsqueeze_703);  mul_316 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_247: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_246, add_237);  add_246 = add_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_119: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(add_247, arg239_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg239_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_248: "f32[960]" = torch.ops.aten.add.Tensor(arg241_1, 1e-05);  arg241_1 = None
        sqrt_88: "f32[960]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_88: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_317: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_705: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_317, -1);  mul_317 = None
        unsqueeze_707: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_705);  convolution_119 = unsqueeze_705 = None
        mul_318: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_709: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_319: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_709);  mul_318 = unsqueeze_709 = None
        unsqueeze_710: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_711: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_249: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_319, unsqueeze_711);  mul_319 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_250: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_249, 3)
        clamp_min_53: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_250, 0);  add_250 = None
        clamp_max_53: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_53, 6);  clamp_min_53 = None
        mul_320: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_249, clamp_max_53);  add_249 = clamp_max_53 = None
        div_53: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_320, 6);  mul_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_120: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(div_53, arg244_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960);  div_53 = arg244_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_251: "f32[960]" = torch.ops.aten.add.Tensor(arg246_1, 1e-05);  arg246_1 = None
        sqrt_89: "f32[960]" = torch.ops.aten.sqrt.default(add_251);  add_251 = None
        reciprocal_89: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_321: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_713: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
        unsqueeze_715: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_713);  convolution_120 = unsqueeze_713 = None
        mul_322: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_717: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_323: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_717);  mul_322 = unsqueeze_717 = None
        unsqueeze_718: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_719: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_252: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_719);  mul_323 = unsqueeze_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_253: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_252, 3)
        clamp_min_54: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_253, 0);  add_253 = None
        clamp_max_54: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_54, 6);  clamp_min_54 = None
        mul_324: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_252, clamp_max_54);  add_252 = clamp_max_54 = None
        div_54: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_324, 6);  mul_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_16: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(div_54, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_121: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_16, arg249_1, arg250_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_16 = arg249_1 = arg250_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_37: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_121);  convolution_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_122: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_37, arg251_1, arg252_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_37 = arg251_1 = arg252_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_254: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_122, 3);  convolution_122 = None
        clamp_min_55: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_254, 0);  add_254 = None
        clamp_max_55: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_55, 6);  clamp_min_55 = None
        div_55: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_55, 6);  clamp_max_55 = None
        mul_325: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_54, div_55);  div_54 = div_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_123: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_325, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_325 = arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_255: "f32[160]" = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_90: "f32[160]" = torch.ops.aten.sqrt.default(add_255);  add_255 = None
        reciprocal_90: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_326: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_721: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_326, -1);  mul_326 = None
        unsqueeze_723: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_721);  convolution_123 = unsqueeze_721 = None
        mul_327: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_725: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_328: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_327, unsqueeze_725);  mul_327 = unsqueeze_725 = None
        unsqueeze_726: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_727: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_256: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_727);  mul_328 = unsqueeze_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_257: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_256, add_247);  add_256 = add_247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:111 in forward, code: x = self.conv(x)
        convolution_124: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(add_257, arg258_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_257 = arg258_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_258: "f32[960]" = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
        sqrt_91: "f32[960]" = torch.ops.aten.sqrt.default(add_258);  add_258 = None
        reciprocal_91: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_329: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_729: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_329, -1);  mul_329 = None
        unsqueeze_731: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_729);  convolution_124 = unsqueeze_729 = None
        mul_330: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_733: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_331: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_330, unsqueeze_733);  mul_330 = unsqueeze_733 = None
        unsqueeze_734: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_735: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_259: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_331, unsqueeze_735);  mul_331 = unsqueeze_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_260: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_259, 3)
        clamp_min_56: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_260, 0);  add_260 = None
        clamp_max_56: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_56, 6);  clamp_min_56 = None
        mul_332: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_259, clamp_max_56);  add_259 = clamp_max_56 = None
        div_56: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_332, 6);  mul_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_17: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(div_56, [-1, -2], True);  div_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:261 in forward_head, code: x = self.conv_head(x)
        convolution_125: "f32[8, 1280, 1, 1]" = torch.ops.aten.convolution.default(mean_17, arg263_1, arg264_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg263_1 = arg264_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:263 in forward_head, code: x = self.act2(x)
        add_261: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convolution_125, 3)
        clamp_min_57: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add_261, 0);  add_261 = None
        clamp_max_57: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_57, 6);  clamp_min_57 = None
        mul_333: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_125, clamp_max_57);  convolution_125 = clamp_max_57 = None
        div_57: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul_333, 6);  mul_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_1: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        view_3: "f32[8, 1280]" = torch.ops.aten.view.default(div_57, [8, 1280]);  div_57 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg266_1, view_3, permute_1);  arg266_1 = view_3 = permute_1 = None
        return (addmm_1,)
        