class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[24, 32, 1, 1]", arg7_1: "f32[24]", arg8_1: "f32[24]", arg9_1: "f32[24]", arg10_1: "f32[24]", arg11_1: "f32[24, 8, 3, 3]", arg12_1: "f32[24]", arg13_1: "f32[24]", arg14_1: "f32[24]", arg15_1: "f32[24]", arg16_1: "f32[8, 24, 1, 1]", arg17_1: "f32[8]", arg18_1: "f32[24, 8, 1, 1]", arg19_1: "f32[24]", arg20_1: "f32[24, 24, 1, 1]", arg21_1: "f32[24]", arg22_1: "f32[24]", arg23_1: "f32[24]", arg24_1: "f32[24]", arg25_1: "f32[24, 32, 1, 1]", arg26_1: "f32[24]", arg27_1: "f32[24]", arg28_1: "f32[24]", arg29_1: "f32[24]", arg30_1: "f32[56, 24, 1, 1]", arg31_1: "f32[56]", arg32_1: "f32[56]", arg33_1: "f32[56]", arg34_1: "f32[56]", arg35_1: "f32[56, 8, 3, 3]", arg36_1: "f32[56]", arg37_1: "f32[56]", arg38_1: "f32[56]", arg39_1: "f32[56]", arg40_1: "f32[6, 56, 1, 1]", arg41_1: "f32[6]", arg42_1: "f32[56, 6, 1, 1]", arg43_1: "f32[56]", arg44_1: "f32[56, 56, 1, 1]", arg45_1: "f32[56]", arg46_1: "f32[56]", arg47_1: "f32[56]", arg48_1: "f32[56]", arg49_1: "f32[56, 24, 1, 1]", arg50_1: "f32[56]", arg51_1: "f32[56]", arg52_1: "f32[56]", arg53_1: "f32[56]", arg54_1: "f32[152, 56, 1, 1]", arg55_1: "f32[152]", arg56_1: "f32[152]", arg57_1: "f32[152]", arg58_1: "f32[152]", arg59_1: "f32[152, 8, 3, 3]", arg60_1: "f32[152]", arg61_1: "f32[152]", arg62_1: "f32[152]", arg63_1: "f32[152]", arg64_1: "f32[14, 152, 1, 1]", arg65_1: "f32[14]", arg66_1: "f32[152, 14, 1, 1]", arg67_1: "f32[152]", arg68_1: "f32[152, 152, 1, 1]", arg69_1: "f32[152]", arg70_1: "f32[152]", arg71_1: "f32[152]", arg72_1: "f32[152]", arg73_1: "f32[152, 56, 1, 1]", arg74_1: "f32[152]", arg75_1: "f32[152]", arg76_1: "f32[152]", arg77_1: "f32[152]", arg78_1: "f32[152, 152, 1, 1]", arg79_1: "f32[152]", arg80_1: "f32[152]", arg81_1: "f32[152]", arg82_1: "f32[152]", arg83_1: "f32[152, 8, 3, 3]", arg84_1: "f32[152]", arg85_1: "f32[152]", arg86_1: "f32[152]", arg87_1: "f32[152]", arg88_1: "f32[38, 152, 1, 1]", arg89_1: "f32[38]", arg90_1: "f32[152, 38, 1, 1]", arg91_1: "f32[152]", arg92_1: "f32[152, 152, 1, 1]", arg93_1: "f32[152]", arg94_1: "f32[152]", arg95_1: "f32[152]", arg96_1: "f32[152]", arg97_1: "f32[152, 152, 1, 1]", arg98_1: "f32[152]", arg99_1: "f32[152]", arg100_1: "f32[152]", arg101_1: "f32[152]", arg102_1: "f32[152, 8, 3, 3]", arg103_1: "f32[152]", arg104_1: "f32[152]", arg105_1: "f32[152]", arg106_1: "f32[152]", arg107_1: "f32[38, 152, 1, 1]", arg108_1: "f32[38]", arg109_1: "f32[152, 38, 1, 1]", arg110_1: "f32[152]", arg111_1: "f32[152, 152, 1, 1]", arg112_1: "f32[152]", arg113_1: "f32[152]", arg114_1: "f32[152]", arg115_1: "f32[152]", arg116_1: "f32[152, 152, 1, 1]", arg117_1: "f32[152]", arg118_1: "f32[152]", arg119_1: "f32[152]", arg120_1: "f32[152]", arg121_1: "f32[152, 8, 3, 3]", arg122_1: "f32[152]", arg123_1: "f32[152]", arg124_1: "f32[152]", arg125_1: "f32[152]", arg126_1: "f32[38, 152, 1, 1]", arg127_1: "f32[38]", arg128_1: "f32[152, 38, 1, 1]", arg129_1: "f32[152]", arg130_1: "f32[152, 152, 1, 1]", arg131_1: "f32[152]", arg132_1: "f32[152]", arg133_1: "f32[152]", arg134_1: "f32[152]", arg135_1: "f32[368, 152, 1, 1]", arg136_1: "f32[368]", arg137_1: "f32[368]", arg138_1: "f32[368]", arg139_1: "f32[368]", arg140_1: "f32[368, 8, 3, 3]", arg141_1: "f32[368]", arg142_1: "f32[368]", arg143_1: "f32[368]", arg144_1: "f32[368]", arg145_1: "f32[38, 368, 1, 1]", arg146_1: "f32[38]", arg147_1: "f32[368, 38, 1, 1]", arg148_1: "f32[368]", arg149_1: "f32[368, 368, 1, 1]", arg150_1: "f32[368]", arg151_1: "f32[368]", arg152_1: "f32[368]", arg153_1: "f32[368]", arg154_1: "f32[368, 152, 1, 1]", arg155_1: "f32[368]", arg156_1: "f32[368]", arg157_1: "f32[368]", arg158_1: "f32[368]", arg159_1: "f32[368, 368, 1, 1]", arg160_1: "f32[368]", arg161_1: "f32[368]", arg162_1: "f32[368]", arg163_1: "f32[368]", arg164_1: "f32[368, 8, 3, 3]", arg165_1: "f32[368]", arg166_1: "f32[368]", arg167_1: "f32[368]", arg168_1: "f32[368]", arg169_1: "f32[92, 368, 1, 1]", arg170_1: "f32[92]", arg171_1: "f32[368, 92, 1, 1]", arg172_1: "f32[368]", arg173_1: "f32[368, 368, 1, 1]", arg174_1: "f32[368]", arg175_1: "f32[368]", arg176_1: "f32[368]", arg177_1: "f32[368]", arg178_1: "f32[368, 368, 1, 1]", arg179_1: "f32[368]", arg180_1: "f32[368]", arg181_1: "f32[368]", arg182_1: "f32[368]", arg183_1: "f32[368, 8, 3, 3]", arg184_1: "f32[368]", arg185_1: "f32[368]", arg186_1: "f32[368]", arg187_1: "f32[368]", arg188_1: "f32[92, 368, 1, 1]", arg189_1: "f32[92]", arg190_1: "f32[368, 92, 1, 1]", arg191_1: "f32[368]", arg192_1: "f32[368, 368, 1, 1]", arg193_1: "f32[368]", arg194_1: "f32[368]", arg195_1: "f32[368]", arg196_1: "f32[368]", arg197_1: "f32[368, 368, 1, 1]", arg198_1: "f32[368]", arg199_1: "f32[368]", arg200_1: "f32[368]", arg201_1: "f32[368]", arg202_1: "f32[368, 8, 3, 3]", arg203_1: "f32[368]", arg204_1: "f32[368]", arg205_1: "f32[368]", arg206_1: "f32[368]", arg207_1: "f32[92, 368, 1, 1]", arg208_1: "f32[92]", arg209_1: "f32[368, 92, 1, 1]", arg210_1: "f32[368]", arg211_1: "f32[368, 368, 1, 1]", arg212_1: "f32[368]", arg213_1: "f32[368]", arg214_1: "f32[368]", arg215_1: "f32[368]", arg216_1: "f32[368, 368, 1, 1]", arg217_1: "f32[368]", arg218_1: "f32[368]", arg219_1: "f32[368]", arg220_1: "f32[368]", arg221_1: "f32[368, 8, 3, 3]", arg222_1: "f32[368]", arg223_1: "f32[368]", arg224_1: "f32[368]", arg225_1: "f32[368]", arg226_1: "f32[92, 368, 1, 1]", arg227_1: "f32[92]", arg228_1: "f32[368, 92, 1, 1]", arg229_1: "f32[368]", arg230_1: "f32[368, 368, 1, 1]", arg231_1: "f32[368]", arg232_1: "f32[368]", arg233_1: "f32[368]", arg234_1: "f32[368]", arg235_1: "f32[368, 368, 1, 1]", arg236_1: "f32[368]", arg237_1: "f32[368]", arg238_1: "f32[368]", arg239_1: "f32[368]", arg240_1: "f32[368, 8, 3, 3]", arg241_1: "f32[368]", arg242_1: "f32[368]", arg243_1: "f32[368]", arg244_1: "f32[368]", arg245_1: "f32[92, 368, 1, 1]", arg246_1: "f32[92]", arg247_1: "f32[368, 92, 1, 1]", arg248_1: "f32[368]", arg249_1: "f32[368, 368, 1, 1]", arg250_1: "f32[368]", arg251_1: "f32[368]", arg252_1: "f32[368]", arg253_1: "f32[368]", arg254_1: "f32[368, 368, 1, 1]", arg255_1: "f32[368]", arg256_1: "f32[368]", arg257_1: "f32[368]", arg258_1: "f32[368]", arg259_1: "f32[368, 8, 3, 3]", arg260_1: "f32[368]", arg261_1: "f32[368]", arg262_1: "f32[368]", arg263_1: "f32[368]", arg264_1: "f32[92, 368, 1, 1]", arg265_1: "f32[92]", arg266_1: "f32[368, 92, 1, 1]", arg267_1: "f32[368]", arg268_1: "f32[368, 368, 1, 1]", arg269_1: "f32[368]", arg270_1: "f32[368]", arg271_1: "f32[368]", arg272_1: "f32[368]", arg273_1: "f32[1000, 368]", arg274_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_70: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_101: "f32[32]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_44: "f32[32]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
        reciprocal_44: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_145: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_353: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_145, -1);  mul_145 = None
        unsqueeze_355: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_44: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_353);  convolution_70 = unsqueeze_353 = None
        mul_146: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_357: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_147: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_357);  mul_146 = unsqueeze_357 = None
        unsqueeze_358: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_359: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_102: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_147, unsqueeze_359);  mul_147 = unsqueeze_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_53: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_102);  add_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_71: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(relu_53, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_103: "f32[24]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_45: "f32[24]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
        reciprocal_45: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_148: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_361: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_148, -1);  mul_148 = None
        unsqueeze_363: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_45: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_361);  convolution_71 = unsqueeze_361 = None
        mul_149: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_365: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_150: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_149, unsqueeze_365);  mul_149 = unsqueeze_365 = None
        unsqueeze_366: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_367: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_104: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_150, unsqueeze_367);  mul_150 = unsqueeze_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_54: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_104);  add_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_72: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_54, arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 3);  relu_54 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_105: "f32[24]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_46: "f32[24]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
        reciprocal_46: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_151: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_369: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_151, -1);  mul_151 = None
        unsqueeze_371: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_46: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_369);  convolution_72 = unsqueeze_369 = None
        mul_152: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_373: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_153: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_152, unsqueeze_373);  mul_152 = unsqueeze_373 = None
        unsqueeze_374: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_375: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_106: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_375);  mul_153 = unsqueeze_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_55: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_106);  add_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "f32[8, 24, 1, 1]" = torch.ops.aten.mean.dim(relu_55, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_73: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_14, arg16_1, arg17_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg16_1 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_56: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_73);  convolution_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_74: "f32[8, 24, 1, 1]" = torch.ops.aten.convolution.default(relu_56, arg18_1, arg19_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_56 = arg18_1 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_13: "f32[8, 24, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_74);  convolution_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_154: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(relu_55, sigmoid_13);  relu_55 = sigmoid_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_75: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_154, arg20_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_154 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_107: "f32[24]" = torch.ops.aten.add.Tensor(arg22_1, 1e-05);  arg22_1 = None
        sqrt_47: "f32[24]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
        reciprocal_47: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_155: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_377: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_155, -1);  mul_155 = None
        unsqueeze_379: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_47: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_377);  convolution_75 = unsqueeze_377 = None
        mul_156: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_381: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_157: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_381);  mul_156 = unsqueeze_381 = None
        unsqueeze_382: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_383: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_108: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_383);  mul_157 = unsqueeze_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_76: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_53, arg25_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_53 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_109: "f32[24]" = torch.ops.aten.add.Tensor(arg27_1, 1e-05);  arg27_1 = None
        sqrt_48: "f32[24]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_48: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_158: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_385: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        unsqueeze_387: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_48: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_385);  convolution_76 = unsqueeze_385 = None
        mul_159: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_389: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_160: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_389);  mul_159 = unsqueeze_389 = None
        unsqueeze_390: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_391: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_110: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_391);  mul_160 = unsqueeze_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_111: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_108, add_110);  add_108 = add_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_57: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_111);  add_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_77: "f32[8, 56, 56, 56]" = torch.ops.aten.convolution.default(relu_57, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_112: "f32[56]" = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_49: "f32[56]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
        reciprocal_49: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_161: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_393: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_161, -1);  mul_161 = None
        unsqueeze_395: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_49: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_393);  convolution_77 = unsqueeze_393 = None
        mul_162: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_397: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_163: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_397);  mul_162 = unsqueeze_397 = None
        unsqueeze_398: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_399: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_113: "f32[8, 56, 56, 56]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_399);  mul_163 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_58: "f32[8, 56, 56, 56]" = torch.ops.aten.relu.default(add_113);  add_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_78: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(relu_58, arg35_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 7);  relu_58 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_114: "f32[56]" = torch.ops.aten.add.Tensor(arg37_1, 1e-05);  arg37_1 = None
        sqrt_50: "f32[56]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_50: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_164: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_401: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_164, -1);  mul_164 = None
        unsqueeze_403: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_50: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_401);  convolution_78 = unsqueeze_401 = None
        mul_165: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_405: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_166: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_405);  mul_165 = unsqueeze_405 = None
        unsqueeze_406: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_407: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_115: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_407);  mul_166 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_59: "f32[8, 56, 28, 28]" = torch.ops.aten.relu.default(add_115);  add_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_15: "f32[8, 56, 1, 1]" = torch.ops.aten.mean.dim(relu_59, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_79: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_15, arg40_1, arg41_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg40_1 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_60: "f32[8, 6, 1, 1]" = torch.ops.aten.relu.default(convolution_79);  convolution_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_80: "f32[8, 56, 1, 1]" = torch.ops.aten.convolution.default(relu_60, arg42_1, arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_60 = arg42_1 = arg43_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_14: "f32[8, 56, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_80);  convolution_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_167: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(relu_59, sigmoid_14);  relu_59 = sigmoid_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_81: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(mul_167, arg44_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_167 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_116: "f32[56]" = torch.ops.aten.add.Tensor(arg46_1, 1e-05);  arg46_1 = None
        sqrt_51: "f32[56]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_51: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_168: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_409: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
        unsqueeze_411: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_51: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_409);  convolution_81 = unsqueeze_409 = None
        mul_169: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_413: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_170: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_413);  mul_169 = unsqueeze_413 = None
        unsqueeze_414: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_415: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_117: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_415);  mul_170 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_82: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(relu_57, arg49_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_57 = arg49_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_118: "f32[56]" = torch.ops.aten.add.Tensor(arg51_1, 1e-05);  arg51_1 = None
        sqrt_52: "f32[56]" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_52: "f32[56]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_171: "f32[56]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_417: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_419: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_52: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_417);  convolution_82 = unsqueeze_417 = None
        mul_172: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_421: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_173: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_421);  mul_172 = unsqueeze_421 = None
        unsqueeze_422: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_423: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_119: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_423);  mul_173 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_120: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_117, add_119);  add_117 = add_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_61: "f32[8, 56, 28, 28]" = torch.ops.aten.relu.default(add_120);  add_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_83: "f32[8, 152, 28, 28]" = torch.ops.aten.convolution.default(relu_61, arg54_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_121: "f32[152]" = torch.ops.aten.add.Tensor(arg56_1, 1e-05);  arg56_1 = None
        sqrt_53: "f32[152]" = torch.ops.aten.sqrt.default(add_121);  add_121 = None
        reciprocal_53: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_174: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_425: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_427: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_53: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_425);  convolution_83 = unsqueeze_425 = None
        mul_175: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_429: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_176: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_429);  mul_175 = unsqueeze_429 = None
        unsqueeze_430: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_431: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_122: "f32[8, 152, 28, 28]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_431);  mul_176 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_62: "f32[8, 152, 28, 28]" = torch.ops.aten.relu.default(add_122);  add_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_84: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_62, arg59_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 19);  relu_62 = arg59_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_123: "f32[152]" = torch.ops.aten.add.Tensor(arg61_1, 1e-05);  arg61_1 = None
        sqrt_54: "f32[152]" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_54: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_177: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_433: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
        unsqueeze_435: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_54: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_433);  convolution_84 = unsqueeze_433 = None
        mul_178: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
        unsqueeze_436: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_437: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_179: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_437);  mul_178 = unsqueeze_437 = None
        unsqueeze_438: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_439: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_124: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_439);  mul_179 = unsqueeze_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_63: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_124);  add_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_16: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_63, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_85: "f32[8, 14, 1, 1]" = torch.ops.aten.convolution.default(mean_16, arg64_1, arg65_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_16 = arg64_1 = arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_64: "f32[8, 14, 1, 1]" = torch.ops.aten.relu.default(convolution_85);  convolution_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_86: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_64, arg66_1, arg67_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_64 = arg66_1 = arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_15: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_86);  convolution_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_180: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_63, sigmoid_15);  relu_63 = sigmoid_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_87: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_180, arg68_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_180 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_125: "f32[152]" = torch.ops.aten.add.Tensor(arg70_1, 1e-05);  arg70_1 = None
        sqrt_55: "f32[152]" = torch.ops.aten.sqrt.default(add_125);  add_125 = None
        reciprocal_55: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_181: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_441: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_181, -1);  mul_181 = None
        unsqueeze_443: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_55: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_441);  convolution_87 = unsqueeze_441 = None
        mul_182: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
        unsqueeze_444: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_445: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_183: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_445);  mul_182 = unsqueeze_445 = None
        unsqueeze_446: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_447: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_126: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_183, unsqueeze_447);  mul_183 = unsqueeze_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_88: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_61, arg73_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_61 = arg73_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_127: "f32[152]" = torch.ops.aten.add.Tensor(arg75_1, 1e-05);  arg75_1 = None
        sqrt_56: "f32[152]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_56: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_184: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_449: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_184, -1);  mul_184 = None
        unsqueeze_451: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_56: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_449);  convolution_88 = unsqueeze_449 = None
        mul_185: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
        unsqueeze_452: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_453: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_186: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_185, unsqueeze_453);  mul_185 = unsqueeze_453 = None
        unsqueeze_454: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_455: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_128: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_186, unsqueeze_455);  mul_186 = unsqueeze_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_129: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_126, add_128);  add_126 = add_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_65: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_129);  add_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_89: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_65, arg78_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_130: "f32[152]" = torch.ops.aten.add.Tensor(arg80_1, 1e-05);  arg80_1 = None
        sqrt_57: "f32[152]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_57: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_187: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_457: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_187, -1);  mul_187 = None
        unsqueeze_459: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_57: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_457);  convolution_89 = unsqueeze_457 = None
        mul_188: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
        unsqueeze_460: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_461: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_189: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_461);  mul_188 = unsqueeze_461 = None
        unsqueeze_462: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_463: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_131: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_463);  mul_189 = unsqueeze_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_66: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_131);  add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_90: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_66, arg83_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19);  relu_66 = arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_132: "f32[152]" = torch.ops.aten.add.Tensor(arg85_1, 1e-05);  arg85_1 = None
        sqrt_58: "f32[152]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_58: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_190: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_465: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_190, -1);  mul_190 = None
        unsqueeze_467: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_58: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_465);  convolution_90 = unsqueeze_465 = None
        mul_191: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_469: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_192: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_469);  mul_191 = unsqueeze_469 = None
        unsqueeze_470: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_471: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_133: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_192, unsqueeze_471);  mul_192 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_67: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_133);  add_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_17: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_67, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_91: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_17, arg88_1, arg89_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg88_1 = arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_68: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_91);  convolution_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_92: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_68, arg90_1, arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_68 = arg90_1 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_16: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_92);  convolution_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_193: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_67, sigmoid_16);  relu_67 = sigmoid_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_93: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_193, arg92_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_193 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_134: "f32[152]" = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_59: "f32[152]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_59: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_194: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_473: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_194, -1);  mul_194 = None
        unsqueeze_475: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_59: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_473);  convolution_93 = unsqueeze_473 = None
        mul_195: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_477: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_196: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_477);  mul_195 = unsqueeze_477 = None
        unsqueeze_478: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_479: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_135: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_196, unsqueeze_479);  mul_196 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_136: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_135, relu_65);  add_135 = relu_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_69: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_136);  add_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_94: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_69, arg97_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_137: "f32[152]" = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_60: "f32[152]" = torch.ops.aten.sqrt.default(add_137);  add_137 = None
        reciprocal_60: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_197: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_481: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_197, -1);  mul_197 = None
        unsqueeze_483: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_60: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_481);  convolution_94 = unsqueeze_481 = None
        mul_198: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_485: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_199: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_485);  mul_198 = unsqueeze_485 = None
        unsqueeze_486: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_487: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_138: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_199, unsqueeze_487);  mul_199 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_70: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_138);  add_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_95: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_70, arg102_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19);  relu_70 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_139: "f32[152]" = torch.ops.aten.add.Tensor(arg104_1, 1e-05);  arg104_1 = None
        sqrt_61: "f32[152]" = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_61: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_200: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_489: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_200, -1);  mul_200 = None
        unsqueeze_491: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_61: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_489);  convolution_95 = unsqueeze_489 = None
        mul_201: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_493: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_202: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_201, unsqueeze_493);  mul_201 = unsqueeze_493 = None
        unsqueeze_494: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_495: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_140: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_495);  mul_202 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_71: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_140);  add_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_18: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_71, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_96: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_18, arg107_1, arg108_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg107_1 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_72: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_96);  convolution_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_97: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_72, arg109_1, arg110_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_72 = arg109_1 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_17: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_97);  convolution_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_203: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_71, sigmoid_17);  relu_71 = sigmoid_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_98: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_203, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_203 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_141: "f32[152]" = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_62: "f32[152]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_62: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_204: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_497: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
        unsqueeze_499: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_497);  convolution_98 = unsqueeze_497 = None
        mul_205: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_501: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_206: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_501);  mul_205 = unsqueeze_501 = None
        unsqueeze_502: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_503: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_142: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_503);  mul_206 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_143: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_142, relu_69);  add_142 = relu_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_73: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_143);  add_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_99: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_73, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_144: "f32[152]" = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_63: "f32[152]" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_63: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_207: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_505: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_507: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_505);  convolution_99 = unsqueeze_505 = None
        mul_208: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_509: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_209: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_509);  mul_208 = unsqueeze_509 = None
        unsqueeze_510: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_511: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_145: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_511);  mul_209 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_74: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_145);  add_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_100: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_74, arg121_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19);  relu_74 = arg121_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_146: "f32[152]" = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_64: "f32[152]" = torch.ops.aten.sqrt.default(add_146);  add_146 = None
        reciprocal_64: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_210: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_513: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_515: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_513);  convolution_100 = unsqueeze_513 = None
        mul_211: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_517: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_212: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_517);  mul_211 = unsqueeze_517 = None
        unsqueeze_518: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_519: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_147: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_519);  mul_212 = unsqueeze_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_75: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_147);  add_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_19: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_75, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_101: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_19, arg126_1, arg127_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg126_1 = arg127_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_76: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_101);  convolution_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_102: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_76, arg128_1, arg129_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_76 = arg128_1 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_18: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_102);  convolution_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_213: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_75, sigmoid_18);  relu_75 = sigmoid_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_103: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_213, arg130_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_213 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_148: "f32[152]" = torch.ops.aten.add.Tensor(arg132_1, 1e-05);  arg132_1 = None
        sqrt_65: "f32[152]" = torch.ops.aten.sqrt.default(add_148);  add_148 = None
        reciprocal_65: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_214: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_521: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_214, -1);  mul_214 = None
        unsqueeze_523: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_521);  convolution_103 = unsqueeze_521 = None
        mul_215: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_525: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_216: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_215, unsqueeze_525);  mul_215 = unsqueeze_525 = None
        unsqueeze_526: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_527: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_149: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_527);  mul_216 = unsqueeze_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_150: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_149, relu_73);  add_149 = relu_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_77: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_150);  add_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_104: "f32[8, 368, 14, 14]" = torch.ops.aten.convolution.default(relu_77, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_151: "f32[368]" = torch.ops.aten.add.Tensor(arg137_1, 1e-05);  arg137_1 = None
        sqrt_66: "f32[368]" = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_66: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_217: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_529: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_217, -1);  mul_217 = None
        unsqueeze_531: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_529);  convolution_104 = unsqueeze_529 = None
        mul_218: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_533: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_219: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_533);  mul_218 = unsqueeze_533 = None
        unsqueeze_534: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_535: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_152: "f32[8, 368, 14, 14]" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_535);  mul_219 = unsqueeze_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_78: "f32[8, 368, 14, 14]" = torch.ops.aten.relu.default(add_152);  add_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_105: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_78, arg140_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 46);  relu_78 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_153: "f32[368]" = torch.ops.aten.add.Tensor(arg142_1, 1e-05);  arg142_1 = None
        sqrt_67: "f32[368]" = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_67: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_220: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_537: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_220, -1);  mul_220 = None
        unsqueeze_539: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_537);  convolution_105 = unsqueeze_537 = None
        mul_221: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_541: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_222: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_541);  mul_221 = unsqueeze_541 = None
        unsqueeze_542: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_543: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_154: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_543);  mul_222 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_79: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_154);  add_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_20: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_79, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_106: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_20, arg145_1, arg146_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg145_1 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_80: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_106);  convolution_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_107: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_80, arg147_1, arg148_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_80 = arg147_1 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_19: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_107);  convolution_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_223: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_79, sigmoid_19);  relu_79 = sigmoid_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_108: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_223, arg149_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_223 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_155: "f32[368]" = torch.ops.aten.add.Tensor(arg151_1, 1e-05);  arg151_1 = None
        sqrt_68: "f32[368]" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
        reciprocal_68: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_224: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_545: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_224, -1);  mul_224 = None
        unsqueeze_547: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_545);  convolution_108 = unsqueeze_545 = None
        mul_225: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_549: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_226: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_549);  mul_225 = unsqueeze_549 = None
        unsqueeze_550: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_551: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_156: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_551);  mul_226 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_109: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_77, arg154_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_77 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_157: "f32[368]" = torch.ops.aten.add.Tensor(arg156_1, 1e-05);  arg156_1 = None
        sqrt_69: "f32[368]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_69: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_227: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_553: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_227, -1);  mul_227 = None
        unsqueeze_555: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_553);  convolution_109 = unsqueeze_553 = None
        mul_228: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_557: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_229: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_228, unsqueeze_557);  mul_228 = unsqueeze_557 = None
        unsqueeze_558: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_559: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_158: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_229, unsqueeze_559);  mul_229 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_159: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_156, add_158);  add_156 = add_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_81: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_159);  add_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_110: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_81, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg159_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_160: "f32[368]" = torch.ops.aten.add.Tensor(arg161_1, 1e-05);  arg161_1 = None
        sqrt_70: "f32[368]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_70: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_230: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_561: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_230, -1);  mul_230 = None
        unsqueeze_563: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_561);  convolution_110 = unsqueeze_561 = None
        mul_231: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_565: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_232: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_565);  mul_231 = unsqueeze_565 = None
        unsqueeze_566: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_567: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_161: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_567);  mul_232 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_82: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_161);  add_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_111: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_82, arg164_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46);  relu_82 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_162: "f32[368]" = torch.ops.aten.add.Tensor(arg166_1, 1e-05);  arg166_1 = None
        sqrt_71: "f32[368]" = torch.ops.aten.sqrt.default(add_162);  add_162 = None
        reciprocal_71: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_233: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_569: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_233, -1);  mul_233 = None
        unsqueeze_571: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_569);  convolution_111 = unsqueeze_569 = None
        mul_234: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_573: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_235: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_573);  mul_234 = unsqueeze_573 = None
        unsqueeze_574: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_575: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_163: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_575);  mul_235 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_83: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_163);  add_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_21: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_83, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_112: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_21, arg169_1, arg170_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg169_1 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_84: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_112);  convolution_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_113: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_84, arg171_1, arg172_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_84 = arg171_1 = arg172_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_20: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_113);  convolution_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_236: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_83, sigmoid_20);  relu_83 = sigmoid_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_114: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_236, arg173_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_236 = arg173_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_164: "f32[368]" = torch.ops.aten.add.Tensor(arg175_1, 1e-05);  arg175_1 = None
        sqrt_72: "f32[368]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_72: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_237: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_577: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_579: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_577);  convolution_114 = unsqueeze_577 = None
        mul_238: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_581: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_239: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_581);  mul_238 = unsqueeze_581 = None
        unsqueeze_582: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_583: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_165: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_583);  mul_239 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_166: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_165, relu_81);  add_165 = relu_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_85: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_166);  add_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_115: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_85, arg178_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_167: "f32[368]" = torch.ops.aten.add.Tensor(arg180_1, 1e-05);  arg180_1 = None
        sqrt_73: "f32[368]" = torch.ops.aten.sqrt.default(add_167);  add_167 = None
        reciprocal_73: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_240: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_585: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_587: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_585);  convolution_115 = unsqueeze_585 = None
        mul_241: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_589: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_242: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_589);  mul_241 = unsqueeze_589 = None
        unsqueeze_590: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_591: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_168: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_591);  mul_242 = unsqueeze_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_86: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_168);  add_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_116: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_86, arg183_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46);  relu_86 = arg183_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_169: "f32[368]" = torch.ops.aten.add.Tensor(arg185_1, 1e-05);  arg185_1 = None
        sqrt_74: "f32[368]" = torch.ops.aten.sqrt.default(add_169);  add_169 = None
        reciprocal_74: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_243: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_593: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_595: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_593);  convolution_116 = unsqueeze_593 = None
        mul_244: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_597: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_245: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_597);  mul_244 = unsqueeze_597 = None
        unsqueeze_598: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_599: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_170: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_599);  mul_245 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_87: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_170);  add_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_22: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_87, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_117: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_22, arg188_1, arg189_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg188_1 = arg189_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_88: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_117);  convolution_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_118: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_88, arg190_1, arg191_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_88 = arg190_1 = arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_21: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_118);  convolution_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_246: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_87, sigmoid_21);  relu_87 = sigmoid_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_119: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_246, arg192_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_246 = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_171: "f32[368]" = torch.ops.aten.add.Tensor(arg194_1, 1e-05);  arg194_1 = None
        sqrt_75: "f32[368]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_75: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_247: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_601: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
        unsqueeze_603: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_601);  convolution_119 = unsqueeze_601 = None
        mul_248: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_605: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_249: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_605);  mul_248 = unsqueeze_605 = None
        unsqueeze_606: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_607: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_172: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_249, unsqueeze_607);  mul_249 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_173: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_172, relu_85);  add_172 = relu_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_89: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_173);  add_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_120: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_89, arg197_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg197_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_174: "f32[368]" = torch.ops.aten.add.Tensor(arg199_1, 1e-05);  arg199_1 = None
        sqrt_76: "f32[368]" = torch.ops.aten.sqrt.default(add_174);  add_174 = None
        reciprocal_76: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_250: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_609: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_611: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_609);  convolution_120 = unsqueeze_609 = None
        mul_251: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_613: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_252: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_613);  mul_251 = unsqueeze_613 = None
        unsqueeze_614: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
        unsqueeze_615: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_175: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_615);  mul_252 = unsqueeze_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_90: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_175);  add_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_121: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_90, arg202_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46);  relu_90 = arg202_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_176: "f32[368]" = torch.ops.aten.add.Tensor(arg204_1, 1e-05);  arg204_1 = None
        sqrt_77: "f32[368]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
        reciprocal_77: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_253: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_617: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_619: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_617);  convolution_121 = unsqueeze_617 = None
        mul_254: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_621: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_255: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_621);  mul_254 = unsqueeze_621 = None
        unsqueeze_622: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_623: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_177: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_623);  mul_255 = unsqueeze_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_91: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_177);  add_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_23: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_91, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_122: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_23, arg207_1, arg208_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg207_1 = arg208_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_92: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_122);  convolution_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_123: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_92, arg209_1, arg210_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_92 = arg209_1 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_22: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_123);  convolution_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_256: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_91, sigmoid_22);  relu_91 = sigmoid_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_124: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_256, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_256 = arg211_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_178: "f32[368]" = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_78: "f32[368]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_78: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_257: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_625: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_257, -1);  mul_257 = None
        unsqueeze_627: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_625);  convolution_124 = unsqueeze_625 = None
        mul_258: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_629: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_259: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_629);  mul_258 = unsqueeze_629 = None
        unsqueeze_630: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_631: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_179: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_259, unsqueeze_631);  mul_259 = unsqueeze_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_180: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_179, relu_89);  add_179 = relu_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_93: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_180);  add_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_125: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_93, arg216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg216_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_181: "f32[368]" = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_79: "f32[368]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_79: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_260: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_633: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_260, -1);  mul_260 = None
        unsqueeze_635: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_633);  convolution_125 = unsqueeze_633 = None
        mul_261: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_637: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_262: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_637);  mul_261 = unsqueeze_637 = None
        unsqueeze_638: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_639: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_182: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_639);  mul_262 = unsqueeze_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_94: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_182);  add_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_126: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_94, arg221_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46);  relu_94 = arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_183: "f32[368]" = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_80: "f32[368]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_80: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_263: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_641: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_263, -1);  mul_263 = None
        unsqueeze_643: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_641);  convolution_126 = unsqueeze_641 = None
        mul_264: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_645: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_265: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_264, unsqueeze_645);  mul_264 = unsqueeze_645 = None
        unsqueeze_646: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_647: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_184: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_647);  mul_265 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_95: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_184);  add_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_24: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_95, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_127: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_24, arg226_1, arg227_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg226_1 = arg227_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_96: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_127);  convolution_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_128: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_96, arg228_1, arg229_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_96 = arg228_1 = arg229_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_23: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_128);  convolution_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_266: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_95, sigmoid_23);  relu_95 = sigmoid_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_129: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_266, arg230_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_266 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_185: "f32[368]" = torch.ops.aten.add.Tensor(arg232_1, 1e-05);  arg232_1 = None
        sqrt_81: "f32[368]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_81: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_267: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_649: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_651: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_649);  convolution_129 = unsqueeze_649 = None
        mul_268: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_653: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_269: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_653);  mul_268 = unsqueeze_653 = None
        unsqueeze_654: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_655: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_186: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_655);  mul_269 = unsqueeze_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_187: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_186, relu_93);  add_186 = relu_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_97: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_187);  add_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_130: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_97, arg235_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg235_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_188: "f32[368]" = torch.ops.aten.add.Tensor(arg237_1, 1e-05);  arg237_1 = None
        sqrt_82: "f32[368]" = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_82: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_270: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_657: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_659: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_657);  convolution_130 = unsqueeze_657 = None
        mul_271: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_661: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_272: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_661);  mul_271 = unsqueeze_661 = None
        unsqueeze_662: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_663: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_189: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_663);  mul_272 = unsqueeze_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_98: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_189);  add_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_131: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_98, arg240_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46);  relu_98 = arg240_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_190: "f32[368]" = torch.ops.aten.add.Tensor(arg242_1, 1e-05);  arg242_1 = None
        sqrt_83: "f32[368]" = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_83: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_273: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_665: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_667: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_665);  convolution_131 = unsqueeze_665 = None
        mul_274: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_669: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_275: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_669);  mul_274 = unsqueeze_669 = None
        unsqueeze_670: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_671: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_191: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_671);  mul_275 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_99: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_191);  add_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_25: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_99, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_132: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_25, arg245_1, arg246_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg245_1 = arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_100: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_132);  convolution_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_133: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_100, arg247_1, arg248_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_100 = arg247_1 = arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_24: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_133);  convolution_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_276: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_99, sigmoid_24);  relu_99 = sigmoid_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_134: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_276, arg249_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_276 = arg249_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_192: "f32[368]" = torch.ops.aten.add.Tensor(arg251_1, 1e-05);  arg251_1 = None
        sqrt_84: "f32[368]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_84: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_277: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_673: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_675: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_673);  convolution_134 = unsqueeze_673 = None
        mul_278: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_677: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_279: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_677);  mul_278 = unsqueeze_677 = None
        unsqueeze_678: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_679: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_193: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_679);  mul_279 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_194: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_193, relu_97);  add_193 = relu_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_101: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_194);  add_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_135: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_101, arg254_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg254_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_195: "f32[368]" = torch.ops.aten.add.Tensor(arg256_1, 1e-05);  arg256_1 = None
        sqrt_85: "f32[368]" = torch.ops.aten.sqrt.default(add_195);  add_195 = None
        reciprocal_85: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_280: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_681: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_280, -1);  mul_280 = None
        unsqueeze_683: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_681);  convolution_135 = unsqueeze_681 = None
        mul_281: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_685: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_282: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_281, unsqueeze_685);  mul_281 = unsqueeze_685 = None
        unsqueeze_686: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_687: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_196: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_282, unsqueeze_687);  mul_282 = unsqueeze_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_102: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_196);  add_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_136: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_102, arg259_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46);  relu_102 = arg259_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_197: "f32[368]" = torch.ops.aten.add.Tensor(arg261_1, 1e-05);  arg261_1 = None
        sqrt_86: "f32[368]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_86: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_283: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_689: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_691: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_689);  convolution_136 = unsqueeze_689 = None
        mul_284: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_693: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_285: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_693);  mul_284 = unsqueeze_693 = None
        unsqueeze_694: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_695: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_198: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_285, unsqueeze_695);  mul_285 = unsqueeze_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_103: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_198);  add_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_26: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_103, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_137: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_26, arg264_1, arg265_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg264_1 = arg265_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_104: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_137);  convolution_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_138: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_104, arg266_1, arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_104 = arg266_1 = arg267_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_25: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_138);  convolution_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_286: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_103, sigmoid_25);  relu_103 = sigmoid_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_139: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_286, arg268_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_286 = arg268_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_199: "f32[368]" = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
        sqrt_87: "f32[368]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_87: "f32[368]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_287: "f32[368]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_697: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(mul_287, -1);  mul_287 = None
        unsqueeze_699: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_697);  convolution_139 = unsqueeze_697 = None
        mul_288: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_701: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_289: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_701);  mul_288 = unsqueeze_701 = None
        unsqueeze_702: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_703: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_200: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_703);  mul_289 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:245 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_201: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_200, relu_101);  add_200 = relu_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/regnet.py:246 in forward, code: x = self.act3(x)
        relu_105: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_201);  add_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_27: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_105, [-1, -2], True);  relu_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 368]" = torch.ops.aten.view.default(mean_27, [8, 368]);  mean_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_1: "f32[368, 1000]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg274_1, view_1, permute_1);  arg274_1 = view_1 = permute_1 = None
        return (addmm_1,)
        