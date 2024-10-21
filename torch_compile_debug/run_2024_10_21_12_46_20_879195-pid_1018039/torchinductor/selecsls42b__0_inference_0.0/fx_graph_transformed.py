class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[64, 32, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64, 64, 1, 1]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[32, 64, 3, 3]", arg17_1: "f32[32]", arg18_1: "f32[32]", arg19_1: "f32[32]", arg20_1: "f32[32]", arg21_1: "f32[64, 32, 1, 1]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[32, 64, 3, 3]", arg27_1: "f32[32]", arg28_1: "f32[32]", arg29_1: "f32[32]", arg30_1: "f32[32]", arg31_1: "f32[64, 128, 1, 1]", arg32_1: "f32[64]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[64, 64, 3, 3]", arg37_1: "f32[64]", arg38_1: "f32[64]", arg39_1: "f32[64]", arg40_1: "f32[64]", arg41_1: "f32[64, 64, 1, 1]", arg42_1: "f32[64]", arg43_1: "f32[64]", arg44_1: "f32[64]", arg45_1: "f32[64]", arg46_1: "f32[32, 64, 3, 3]", arg47_1: "f32[32]", arg48_1: "f32[32]", arg49_1: "f32[32]", arg50_1: "f32[32]", arg51_1: "f32[64, 32, 1, 1]", arg52_1: "f32[64]", arg53_1: "f32[64]", arg54_1: "f32[64]", arg55_1: "f32[64]", arg56_1: "f32[32, 64, 3, 3]", arg57_1: "f32[32]", arg58_1: "f32[32]", arg59_1: "f32[32]", arg60_1: "f32[32]", arg61_1: "f32[128, 192, 1, 1]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[128]", arg66_1: "f32[144, 128, 3, 3]", arg67_1: "f32[144]", arg68_1: "f32[144]", arg69_1: "f32[144]", arg70_1: "f32[144]", arg71_1: "f32[144, 144, 1, 1]", arg72_1: "f32[144]", arg73_1: "f32[144]", arg74_1: "f32[144]", arg75_1: "f32[144]", arg76_1: "f32[72, 144, 3, 3]", arg77_1: "f32[72]", arg78_1: "f32[72]", arg79_1: "f32[72]", arg80_1: "f32[72]", arg81_1: "f32[144, 72, 1, 1]", arg82_1: "f32[144]", arg83_1: "f32[144]", arg84_1: "f32[144]", arg85_1: "f32[144]", arg86_1: "f32[72, 144, 3, 3]", arg87_1: "f32[72]", arg88_1: "f32[72]", arg89_1: "f32[72]", arg90_1: "f32[72]", arg91_1: "f32[144, 288, 1, 1]", arg92_1: "f32[144]", arg93_1: "f32[144]", arg94_1: "f32[144]", arg95_1: "f32[144]", arg96_1: "f32[144, 144, 3, 3]", arg97_1: "f32[144]", arg98_1: "f32[144]", arg99_1: "f32[144]", arg100_1: "f32[144]", arg101_1: "f32[144, 144, 1, 1]", arg102_1: "f32[144]", arg103_1: "f32[144]", arg104_1: "f32[144]", arg105_1: "f32[144]", arg106_1: "f32[72, 144, 3, 3]", arg107_1: "f32[72]", arg108_1: "f32[72]", arg109_1: "f32[72]", arg110_1: "f32[72]", arg111_1: "f32[144, 72, 1, 1]", arg112_1: "f32[144]", arg113_1: "f32[144]", arg114_1: "f32[144]", arg115_1: "f32[144]", arg116_1: "f32[72, 144, 3, 3]", arg117_1: "f32[72]", arg118_1: "f32[72]", arg119_1: "f32[72]", arg120_1: "f32[72]", arg121_1: "f32[288, 432, 1, 1]", arg122_1: "f32[288]", arg123_1: "f32[288]", arg124_1: "f32[288]", arg125_1: "f32[288]", arg126_1: "f32[304, 288, 3, 3]", arg127_1: "f32[304]", arg128_1: "f32[304]", arg129_1: "f32[304]", arg130_1: "f32[304]", arg131_1: "f32[304, 304, 1, 1]", arg132_1: "f32[304]", arg133_1: "f32[304]", arg134_1: "f32[304]", arg135_1: "f32[304]", arg136_1: "f32[152, 304, 3, 3]", arg137_1: "f32[152]", arg138_1: "f32[152]", arg139_1: "f32[152]", arg140_1: "f32[152]", arg141_1: "f32[304, 152, 1, 1]", arg142_1: "f32[304]", arg143_1: "f32[304]", arg144_1: "f32[304]", arg145_1: "f32[304]", arg146_1: "f32[152, 304, 3, 3]", arg147_1: "f32[152]", arg148_1: "f32[152]", arg149_1: "f32[152]", arg150_1: "f32[152]", arg151_1: "f32[304, 608, 1, 1]", arg152_1: "f32[304]", arg153_1: "f32[304]", arg154_1: "f32[304]", arg155_1: "f32[304]", arg156_1: "f32[304, 304, 3, 3]", arg157_1: "f32[304]", arg158_1: "f32[304]", arg159_1: "f32[304]", arg160_1: "f32[304]", arg161_1: "f32[304, 304, 1, 1]", arg162_1: "f32[304]", arg163_1: "f32[304]", arg164_1: "f32[304]", arg165_1: "f32[304]", arg166_1: "f32[152, 304, 3, 3]", arg167_1: "f32[152]", arg168_1: "f32[152]", arg169_1: "f32[152]", arg170_1: "f32[152]", arg171_1: "f32[304, 152, 1, 1]", arg172_1: "f32[304]", arg173_1: "f32[304]", arg174_1: "f32[304]", arg175_1: "f32[304]", arg176_1: "f32[152, 304, 3, 3]", arg177_1: "f32[152]", arg178_1: "f32[152]", arg179_1: "f32[152]", arg180_1: "f32[152]", arg181_1: "f32[480, 912, 1, 1]", arg182_1: "f32[480]", arg183_1: "f32[480]", arg184_1: "f32[480]", arg185_1: "f32[480]", arg186_1: "f32[960, 480, 3, 3]", arg187_1: "f32[960]", arg188_1: "f32[960]", arg189_1: "f32[960]", arg190_1: "f32[960]", arg191_1: "f32[1024, 960, 3, 3]", arg192_1: "f32[1024]", arg193_1: "f32[1024]", arg194_1: "f32[1024]", arg195_1: "f32[1024]", arg196_1: "f32[1280, 1024, 3, 3]", arg197_1: "f32[1280]", arg198_1: "f32[1280]", arg199_1: "f32[1280]", arg200_1: "f32[1280]", arg201_1: "f32[1024, 1280, 1, 1]", arg202_1: "f32[1024]", arg203_1: "f32[1024]", arg204_1: "f32[1024]", arg205_1: "f32[1024]", arg206_1: "f32[1000, 1024]", arg207_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:169 in forward_features, code: x = self.stem(x)
        convolution_41: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        unsqueeze_328: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_329: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_41: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
        add_82: "f32[32]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_41: "f32[32]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        reciprocal_41: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_123: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_331: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_124: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_333: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_125: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
        unsqueeze_334: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_335: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_83: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
        relu_41: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_83);  add_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:100 in forward, code: d1 = self.conv1(x[0])
        convolution_42: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_41, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_41 = arg6_1 = None
        unsqueeze_336: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_337: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_42: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
        add_84: "f32[64]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_42: "f32[64]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_42: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_126: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
        unsqueeze_339: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_127: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_341: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_128: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
        unsqueeze_342: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_343: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_85: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
        relu_42: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_85);  add_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:101 in forward, code: d2 = self.conv3(self.conv2(d1))
        convolution_43: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_42, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg11_1 = None
        unsqueeze_344: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_345: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_43: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
        add_86: "f32[64]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_43: "f32[64]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_43: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_129: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
        unsqueeze_347: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_130: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_349: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_131: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
        unsqueeze_350: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_351: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_87: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
        relu_43: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_87);  add_87 = None
        convolution_44: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_43, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_43 = arg16_1 = None
        unsqueeze_352: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_353: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_44: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
        add_88: "f32[32]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_44: "f32[32]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
        reciprocal_44: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_132: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
        unsqueeze_355: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_133: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_357: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_134: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
        unsqueeze_358: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_359: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_89: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
        relu_44: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_89);  add_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:102 in forward, code: d3 = self.conv5(self.conv4(d2))
        convolution_45: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_44, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg21_1 = None
        unsqueeze_360: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_361: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_45: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
        add_90: "f32[64]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_45: "f32[64]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
        reciprocal_45: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_135: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_363: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_136: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_365: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_137: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
        unsqueeze_366: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_367: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_91: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
        relu_45: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_91);  add_91 = None
        convolution_46: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_45, arg26_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_45 = arg26_1 = None
        unsqueeze_368: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_369: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        sub_46: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
        add_92: "f32[32]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_46: "f32[32]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
        reciprocal_46: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_138: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_370: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
        unsqueeze_371: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        mul_139: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_373: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_140: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
        unsqueeze_374: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_375: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_93: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
        relu_46: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_93);  add_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:104 in forward, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
        cat_6: "f32[8, 128, 56, 56]" = torch.ops.aten.cat.default([relu_42, relu_44, relu_46], 1);  relu_42 = relu_44 = relu_46 = None
        convolution_47: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(cat_6, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg31_1 = None
        unsqueeze_376: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_377: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        sub_47: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
        add_94: "f32[64]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_47: "f32[64]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
        reciprocal_47: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_141: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_378: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
        unsqueeze_379: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        mul_142: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_381: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_143: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
        unsqueeze_382: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_383: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_95: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
        relu_47: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_95);  add_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:100 in forward, code: d1 = self.conv1(x[0])
        convolution_48: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_47, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg36_1 = None
        unsqueeze_384: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_385: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        sub_48: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
        add_96: "f32[64]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_48: "f32[64]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        reciprocal_48: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_144: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_386: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
        unsqueeze_387: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        mul_145: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_389: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_146: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
        unsqueeze_390: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_391: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_97: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
        relu_48: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_97);  add_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:101 in forward, code: d2 = self.conv3(self.conv2(d1))
        convolution_49: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_48, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg41_1 = None
        unsqueeze_392: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_393: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        sub_49: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
        add_98: "f32[64]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_49: "f32[64]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_49: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_147: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_394: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_395: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        mul_148: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_397: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_149: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
        unsqueeze_398: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_399: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_99: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
        relu_49: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_99);  add_99 = None
        convolution_50: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_49, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_49 = arg46_1 = None
        unsqueeze_400: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_401: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        sub_50: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
        add_100: "f32[32]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_50: "f32[32]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        reciprocal_50: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_150: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_402: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
        unsqueeze_403: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        mul_151: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_405: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_152: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
        unsqueeze_406: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_407: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_101: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
        relu_50: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_101);  add_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:102 in forward, code: d3 = self.conv5(self.conv4(d2))
        convolution_51: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_50, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg51_1 = None
        unsqueeze_408: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_409: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        sub_51: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
        add_102: "f32[64]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_51: "f32[64]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_51: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_153: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_410: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
        unsqueeze_411: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        mul_154: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_413: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_155: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
        unsqueeze_414: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_415: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_103: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
        relu_51: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_103);  add_103 = None
        convolution_52: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_51, arg56_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_51 = arg56_1 = None
        unsqueeze_416: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_417: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        sub_52: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
        add_104: "f32[32]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_52: "f32[32]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_52: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_156: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_418: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
        unsqueeze_419: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        mul_157: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_421: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_158: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
        unsqueeze_422: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_423: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_105: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
        relu_52: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_105);  add_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:107 in forward, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
        cat_7: "f32[8, 192, 56, 56]" = torch.ops.aten.cat.default([relu_48, relu_50, relu_52, relu_47], 1);  relu_48 = relu_50 = relu_52 = relu_47 = None
        convolution_53: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(cat_7, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg61_1 = None
        unsqueeze_424: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_425: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        sub_53: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  convolution_53 = unsqueeze_425 = None
        add_106: "f32[128]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_53: "f32[128]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_53: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_159: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_426: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_427: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        mul_160: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_429: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_161: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
        unsqueeze_430: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_431: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_107: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
        relu_53: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_107);  add_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:100 in forward, code: d1 = self.conv1(x[0])
        convolution_54: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_53, arg66_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_53 = arg66_1 = None
        unsqueeze_432: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_433: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        sub_54: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
        add_108: "f32[144]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_54: "f32[144]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
        reciprocal_54: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_162: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_434: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
        unsqueeze_435: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        mul_163: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
        unsqueeze_436: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_437: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_164: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
        unsqueeze_438: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_439: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_109: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
        relu_54: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_109);  add_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:101 in forward, code: d2 = self.conv3(self.conv2(d1))
        convolution_55: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_54, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg71_1 = None
        unsqueeze_440: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_441: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        sub_55: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
        add_110: "f32[144]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_55: "f32[144]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
        reciprocal_55: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_165: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_442: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
        unsqueeze_443: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        mul_166: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
        unsqueeze_444: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_445: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_167: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
        unsqueeze_446: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_447: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_111: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
        relu_55: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_111);  add_111 = None
        convolution_56: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_55, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_55 = arg76_1 = None
        unsqueeze_448: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_449: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        sub_56: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_449);  convolution_56 = unsqueeze_449 = None
        add_112: "f32[72]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_56: "f32[72]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
        reciprocal_56: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_168: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_450: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
        unsqueeze_451: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        mul_169: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
        unsqueeze_452: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_453: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_170: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
        unsqueeze_454: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_455: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_113: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
        relu_56: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_113);  add_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:102 in forward, code: d3 = self.conv5(self.conv4(d2))
        convolution_57: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_56, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg81_1 = None
        unsqueeze_456: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_457: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        sub_57: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_457);  convolution_57 = unsqueeze_457 = None
        add_114: "f32[144]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_57: "f32[144]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_57: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_171: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_458: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_459: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        mul_172: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
        unsqueeze_460: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_461: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_173: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
        unsqueeze_462: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_463: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_115: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
        relu_57: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_115);  add_115 = None
        convolution_58: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_57, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_57 = arg86_1 = None
        unsqueeze_464: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_465: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        sub_58: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_465);  convolution_58 = unsqueeze_465 = None
        add_116: "f32[72]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_58: "f32[72]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_58: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_174: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_466: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_467: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        mul_175: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_469: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_176: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
        unsqueeze_470: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_471: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_117: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
        relu_58: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_117);  add_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:104 in forward, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
        cat_8: "f32[8, 288, 28, 28]" = torch.ops.aten.cat.default([relu_54, relu_56, relu_58], 1);  relu_54 = relu_56 = relu_58 = None
        convolution_59: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(cat_8, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_8 = arg91_1 = None
        unsqueeze_472: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_473: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        sub_59: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_473);  convolution_59 = unsqueeze_473 = None
        add_118: "f32[144]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_59: "f32[144]" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_59: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_177: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_474: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
        unsqueeze_475: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        mul_178: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_477: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_179: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
        unsqueeze_478: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_479: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_119: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
        relu_59: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_119);  add_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:100 in forward, code: d1 = self.conv1(x[0])
        convolution_60: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_59, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg96_1 = None
        unsqueeze_480: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_481: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        sub_60: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_481);  convolution_60 = unsqueeze_481 = None
        add_120: "f32[144]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_60: "f32[144]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_60: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_180: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_482: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
        unsqueeze_483: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        mul_181: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_485: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_182: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
        unsqueeze_486: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_487: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_121: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
        relu_60: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_121);  add_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:101 in forward, code: d2 = self.conv3(self.conv2(d1))
        convolution_61: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_60, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg101_1 = None
        unsqueeze_488: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_489: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        sub_61: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_489);  convolution_61 = unsqueeze_489 = None
        add_122: "f32[144]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_61: "f32[144]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
        reciprocal_61: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_183: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_490: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_491: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        mul_184: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_493: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_185: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
        unsqueeze_494: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_495: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_123: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
        relu_61: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_123);  add_123 = None
        convolution_62: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_61, arg106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_61 = arg106_1 = None
        unsqueeze_496: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_497: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        sub_62: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_497);  convolution_62 = unsqueeze_497 = None
        add_124: "f32[72]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_62: "f32[72]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_62: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_186: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_498: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_499: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        mul_187: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_501: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_188: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
        unsqueeze_502: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_503: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_125: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
        relu_62: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_125);  add_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:102 in forward, code: d3 = self.conv5(self.conv4(d2))
        convolution_63: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_62, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg111_1 = None
        unsqueeze_504: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_505: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        sub_63: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_505);  convolution_63 = unsqueeze_505 = None
        add_126: "f32[144]" = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_63: "f32[144]" = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_63: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_189: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_506: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_507: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        mul_190: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_509: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_191: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
        unsqueeze_510: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_511: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_127: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
        relu_63: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_127);  add_127 = None
        convolution_64: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_63, arg116_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_63 = arg116_1 = None
        unsqueeze_512: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_513: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        sub_64: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_513);  convolution_64 = unsqueeze_513 = None
        add_128: "f32[72]" = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_64: "f32[72]" = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_64: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_192: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_514: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        unsqueeze_515: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        mul_193: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_517: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_194: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
        unsqueeze_518: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_519: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_129: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
        relu_64: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_129);  add_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:107 in forward, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
        cat_9: "f32[8, 432, 28, 28]" = torch.ops.aten.cat.default([relu_60, relu_62, relu_64, relu_59], 1);  relu_60 = relu_62 = relu_64 = relu_59 = None
        convolution_65: "f32[8, 288, 28, 28]" = torch.ops.aten.convolution.default(cat_9, arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_9 = arg121_1 = None
        unsqueeze_520: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_521: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        sub_65: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_521);  convolution_65 = unsqueeze_521 = None
        add_130: "f32[288]" = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_65: "f32[288]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_65: "f32[288]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_195: "f32[288]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_522: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_523: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        mul_196: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_525: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_197: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
        unsqueeze_526: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_527: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_131: "f32[8, 288, 28, 28]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
        relu_65: "f32[8, 288, 28, 28]" = torch.ops.aten.relu.default(add_131);  add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:100 in forward, code: d1 = self.conv1(x[0])
        convolution_66: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_65, arg126_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_65 = arg126_1 = None
        unsqueeze_528: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_529: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        sub_66: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_529);  convolution_66 = unsqueeze_529 = None
        add_132: "f32[304]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_66: "f32[304]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_66: "f32[304]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_198: "f32[304]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_530: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_531: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        mul_199: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_533: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_200: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
        unsqueeze_534: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_535: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_133: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
        relu_66: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_133);  add_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:101 in forward, code: d2 = self.conv3(self.conv2(d1))
        convolution_67: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_66, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg131_1 = None
        unsqueeze_536: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_537: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        sub_67: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_537);  convolution_67 = unsqueeze_537 = None
        add_134: "f32[304]" = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_67: "f32[304]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_67: "f32[304]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_201: "f32[304]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_538: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
        unsqueeze_539: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        mul_202: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_541: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_203: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
        unsqueeze_542: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_543: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_135: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
        relu_67: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_135);  add_135 = None
        convolution_68: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_67, arg136_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_67 = arg136_1 = None
        unsqueeze_544: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_545: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        sub_68: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_545);  convolution_68 = unsqueeze_545 = None
        add_136: "f32[152]" = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_68: "f32[152]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_68: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_204: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_546: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
        unsqueeze_547: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        mul_205: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_549: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_206: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
        unsqueeze_550: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_551: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_137: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
        relu_68: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_137);  add_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:102 in forward, code: d3 = self.conv5(self.conv4(d2))
        convolution_69: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_68, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg141_1 = None
        unsqueeze_552: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_553: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        sub_69: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_553);  convolution_69 = unsqueeze_553 = None
        add_138: "f32[304]" = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_69: "f32[304]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_69: "f32[304]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_207: "f32[304]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_554: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_555: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        mul_208: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_557: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_209: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
        unsqueeze_558: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_559: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_139: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
        relu_69: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_139);  add_139 = None
        convolution_70: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_69, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_69 = arg146_1 = None
        unsqueeze_560: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_561: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        sub_70: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_561);  convolution_70 = unsqueeze_561 = None
        add_140: "f32[152]" = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_70: "f32[152]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_70: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_210: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_562: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_563: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        mul_211: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_565: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_212: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
        unsqueeze_566: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_567: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_141: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
        relu_70: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_141);  add_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:104 in forward, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
        cat_10: "f32[8, 608, 14, 14]" = torch.ops.aten.cat.default([relu_66, relu_68, relu_70], 1);  relu_66 = relu_68 = relu_70 = None
        convolution_71: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(cat_10, arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_10 = arg151_1 = None
        unsqueeze_568: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_569: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        sub_71: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_569);  convolution_71 = unsqueeze_569 = None
        add_142: "f32[304]" = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_71: "f32[304]" = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_71: "f32[304]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_213: "f32[304]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_570: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
        unsqueeze_571: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        mul_214: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_573: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_215: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
        unsqueeze_574: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_575: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_143: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
        relu_71: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_143);  add_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:100 in forward, code: d1 = self.conv1(x[0])
        convolution_72: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_71, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg156_1 = None
        unsqueeze_576: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_577: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        sub_72: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_577);  convolution_72 = unsqueeze_577 = None
        add_144: "f32[304]" = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_72: "f32[304]" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_72: "f32[304]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_216: "f32[304]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_578: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_579: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        mul_217: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_581: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_218: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
        unsqueeze_582: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_583: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_145: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
        relu_72: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_145);  add_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:101 in forward, code: d2 = self.conv3(self.conv2(d1))
        convolution_73: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_72, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg161_1 = None
        unsqueeze_584: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_585: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        sub_73: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_585);  convolution_73 = unsqueeze_585 = None
        add_146: "f32[304]" = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_73: "f32[304]" = torch.ops.aten.sqrt.default(add_146);  add_146 = None
        reciprocal_73: "f32[304]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_219: "f32[304]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_586: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_587: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        mul_220: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_589: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_221: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
        unsqueeze_590: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_591: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_147: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
        relu_73: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_147);  add_147 = None
        convolution_74: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_73, arg166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_73 = arg166_1 = None
        unsqueeze_592: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_593: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        sub_74: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_593);  convolution_74 = unsqueeze_593 = None
        add_148: "f32[152]" = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_74: "f32[152]" = torch.ops.aten.sqrt.default(add_148);  add_148 = None
        reciprocal_74: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_222: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_594: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
        unsqueeze_595: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        mul_223: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_597: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_224: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
        unsqueeze_598: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_599: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_149: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
        relu_74: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_149);  add_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:102 in forward, code: d3 = self.conv5(self.conv4(d2))
        convolution_75: "f32[8, 304, 14, 14]" = torch.ops.aten.convolution.default(relu_74, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg171_1 = None
        unsqueeze_600: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_601: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        sub_75: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_601);  convolution_75 = unsqueeze_601 = None
        add_150: "f32[304]" = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_75: "f32[304]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_75: "f32[304]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_225: "f32[304]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_602: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
        unsqueeze_603: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        mul_226: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_605: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_227: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
        unsqueeze_606: "f32[304, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_607: "f32[304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_151: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
        relu_75: "f32[8, 304, 14, 14]" = torch.ops.aten.relu.default(add_151);  add_151 = None
        convolution_76: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_75, arg176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_75 = arg176_1 = None
        unsqueeze_608: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_609: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        sub_76: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_609);  convolution_76 = unsqueeze_609 = None
        add_152: "f32[152]" = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_76: "f32[152]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_76: "f32[152]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_228: "f32[152]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_610: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_611: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        mul_229: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_613: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_230: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
        unsqueeze_614: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_615: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_153: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
        relu_76: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_153);  add_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:107 in forward, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
        cat_11: "f32[8, 912, 14, 14]" = torch.ops.aten.cat.default([relu_72, relu_74, relu_76, relu_71], 1);  relu_72 = relu_74 = relu_76 = relu_71 = None
        convolution_77: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(cat_11, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_11 = arg181_1 = None
        unsqueeze_616: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_617: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        sub_77: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_617);  convolution_77 = unsqueeze_617 = None
        add_154: "f32[480]" = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_77: "f32[480]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_77: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_231: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_618: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
        unsqueeze_619: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        mul_232: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_621: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_233: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
        unsqueeze_622: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_623: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_155: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
        relu_77: "f32[8, 480, 14, 14]" = torch.ops.aten.relu.default(add_155);  add_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:171 in forward_features, code: x = self.head(self.from_seq(x))
        convolution_78: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(relu_77, arg186_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_77 = arg186_1 = None
        unsqueeze_624: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_625: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        sub_78: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_625);  convolution_78 = unsqueeze_625 = None
        add_156: "f32[960]" = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_78: "f32[960]" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_78: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_234: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_626: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
        unsqueeze_627: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        mul_235: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_629: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_236: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
        unsqueeze_630: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_631: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_157: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
        relu_78: "f32[8, 960, 7, 7]" = torch.ops.aten.relu.default(add_157);  add_157 = None
        convolution_79: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(relu_78, arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_78 = arg191_1 = None
        unsqueeze_632: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_633: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        sub_79: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_633);  convolution_79 = unsqueeze_633 = None
        add_158: "f32[1024]" = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_79: "f32[1024]" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_79: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_237: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_634: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_635: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        mul_238: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_637: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_239: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
        unsqueeze_638: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_639: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_159: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
        relu_79: "f32[8, 1024, 7, 7]" = torch.ops.aten.relu.default(add_159);  add_159 = None
        convolution_80: "f32[8, 1280, 4, 4]" = torch.ops.aten.convolution.default(relu_79, arg196_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_79 = arg196_1 = None
        unsqueeze_640: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_641: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        sub_80: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_641);  convolution_80 = unsqueeze_641 = None
        add_160: "f32[1280]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_80: "f32[1280]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_80: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_240: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_642: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_643: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        mul_241: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_645: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_242: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
        unsqueeze_646: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_647: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_161: "f32[8, 1280, 4, 4]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
        relu_80: "f32[8, 1280, 4, 4]" = torch.ops.aten.relu.default(add_161);  add_161 = None
        convolution_81: "f32[8, 1024, 4, 4]" = torch.ops.aten.convolution.default(relu_80, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_80 = arg201_1 = None
        unsqueeze_648: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_649: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        sub_81: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_649);  convolution_81 = unsqueeze_649 = None
        add_162: "f32[1024]" = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_81: "f32[1024]" = torch.ops.aten.sqrt.default(add_162);  add_162 = None
        reciprocal_81: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_243: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_650: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_651: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        mul_244: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_653: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_245: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
        unsqueeze_654: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_655: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_163: "f32[8, 1024, 4, 4]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
        relu_81: "f32[8, 1024, 4, 4]" = torch.ops.aten.relu.default(add_163);  add_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_81, [-1, -2], True);  relu_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 1024]" = torch.ops.aten.reshape.default(mean_1, [8, 1024]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/selecsls.py:177 in forward_head, code: return x if pre_logits else self.fc(x)
        permute_1: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg207_1, view_1, permute_1);  arg207_1 = view_1 = permute_1 = None
        return (addmm_1,)
        