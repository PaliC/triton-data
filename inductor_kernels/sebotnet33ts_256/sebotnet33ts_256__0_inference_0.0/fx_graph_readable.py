class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[24, 3, 3, 3]", arg1_1: "f32[8, 3, 256, 256]", arg2_1: "f32[24]", arg3_1: "f32[24]", arg4_1: "f32[24]", arg5_1: "f32[24]", arg6_1: "f32[32, 24, 3, 3]", arg7_1: "f32[32]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[64, 32, 3, 3]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[64, 64, 1, 1]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[64, 64, 3, 3]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[8, 64, 1, 1]", arg27_1: "f32[8]", arg28_1: "f32[64, 8, 1, 1]", arg29_1: "f32[64]", arg30_1: "f32[256, 64, 1, 1]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[256, 64, 1, 1]", arg36_1: "f32[256]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[256]", arg40_1: "f32[64, 256, 1, 1]", arg41_1: "f32[64]", arg42_1: "f32[64]", arg43_1: "f32[64]", arg44_1: "f32[64]", arg45_1: "f32[64, 64, 3, 3]", arg46_1: "f32[64]", arg47_1: "f32[64]", arg48_1: "f32[64]", arg49_1: "f32[64]", arg50_1: "f32[8, 64, 1, 1]", arg51_1: "f32[8]", arg52_1: "f32[64, 8, 1, 1]", arg53_1: "f32[64]", arg54_1: "f32[256, 64, 1, 1]", arg55_1: "f32[256]", arg56_1: "f32[256]", arg57_1: "f32[256]", arg58_1: "f32[256]", arg59_1: "f32[128, 256, 1, 1]", arg60_1: "f32[128]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128, 128, 3, 3]", arg65_1: "f32[128]", arg66_1: "f32[128]", arg67_1: "f32[128]", arg68_1: "f32[128]", arg69_1: "f32[8, 128, 1, 1]", arg70_1: "f32[8]", arg71_1: "f32[128, 8, 1, 1]", arg72_1: "f32[128]", arg73_1: "f32[512, 128, 1, 1]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[512]", arg78_1: "f32[512, 256, 1, 1]", arg79_1: "f32[512]", arg80_1: "f32[512]", arg81_1: "f32[512]", arg82_1: "f32[512]", arg83_1: "f32[128, 512, 1, 1]", arg84_1: "f32[128]", arg85_1: "f32[128]", arg86_1: "f32[128]", arg87_1: "f32[128]", arg88_1: "f32[128, 128, 3, 3]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[128]", arg92_1: "f32[128]", arg93_1: "f32[8, 128, 1, 1]", arg94_1: "f32[8]", arg95_1: "f32[128, 8, 1, 1]", arg96_1: "f32[128]", arg97_1: "f32[512, 128, 1, 1]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[512]", arg101_1: "f32[512]", arg102_1: "f32[128, 512, 1, 1]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128]", arg106_1: "f32[128]", arg107_1: "f32[384, 128, 1, 1]", arg108_1: "f32[63, 32]", arg109_1: "f32[63, 32]", arg110_1: "f32[128]", arg111_1: "f32[128]", arg112_1: "f32[128]", arg113_1: "f32[128]", arg114_1: "f32[512, 128, 1, 1]", arg115_1: "f32[512]", arg116_1: "f32[512]", arg117_1: "f32[512]", arg118_1: "f32[512]", arg119_1: "f32[256, 512, 1, 1]", arg120_1: "f32[256]", arg121_1: "f32[256]", arg122_1: "f32[256]", arg123_1: "f32[256]", arg124_1: "f32[256, 256, 3, 3]", arg125_1: "f32[256]", arg126_1: "f32[256]", arg127_1: "f32[256]", arg128_1: "f32[256]", arg129_1: "f32[16, 256, 1, 1]", arg130_1: "f32[16]", arg131_1: "f32[256, 16, 1, 1]", arg132_1: "f32[256]", arg133_1: "f32[1024, 256, 1, 1]", arg134_1: "f32[1024]", arg135_1: "f32[1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024]", arg138_1: "f32[1024, 512, 1, 1]", arg139_1: "f32[1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024]", arg142_1: "f32[1024]", arg143_1: "f32[256, 1024, 1, 1]", arg144_1: "f32[256]", arg145_1: "f32[256]", arg146_1: "f32[256]", arg147_1: "f32[256]", arg148_1: "f32[256, 256, 3, 3]", arg149_1: "f32[256]", arg150_1: "f32[256]", arg151_1: "f32[256]", arg152_1: "f32[256]", arg153_1: "f32[16, 256, 1, 1]", arg154_1: "f32[16]", arg155_1: "f32[256, 16, 1, 1]", arg156_1: "f32[256]", arg157_1: "f32[1024, 256, 1, 1]", arg158_1: "f32[1024]", arg159_1: "f32[1024]", arg160_1: "f32[1024]", arg161_1: "f32[1024]", arg162_1: "f32[256, 1024, 1, 1]", arg163_1: "f32[256]", arg164_1: "f32[256]", arg165_1: "f32[256]", arg166_1: "f32[256]", arg167_1: "f32[768, 256, 1, 1]", arg168_1: "f32[31, 64]", arg169_1: "f32[31, 64]", arg170_1: "f32[256]", arg171_1: "f32[256]", arg172_1: "f32[256]", arg173_1: "f32[256]", arg174_1: "f32[1024, 256, 1, 1]", arg175_1: "f32[1024]", arg176_1: "f32[1024]", arg177_1: "f32[1024]", arg178_1: "f32[1024]", arg179_1: "f32[512, 1024, 1, 1]", arg180_1: "f32[512]", arg181_1: "f32[512]", arg182_1: "f32[512]", arg183_1: "f32[512]", arg184_1: "f32[1536, 512, 1, 1]", arg185_1: "f32[31, 128]", arg186_1: "f32[31, 128]", arg187_1: "f32[512]", arg188_1: "f32[512]", arg189_1: "f32[512]", arg190_1: "f32[512]", arg191_1: "f32[1536, 512, 1, 1]", arg192_1: "f32[1536]", arg193_1: "f32[1536]", arg194_1: "f32[1536]", arg195_1: "f32[1536]", arg196_1: "f32[1536, 1024, 1, 1]", arg197_1: "f32[1536]", arg198_1: "f32[1536]", arg199_1: "f32[1536]", arg200_1: "f32[1536]", arg201_1: "f32[512, 1536, 1, 1]", arg202_1: "f32[512]", arg203_1: "f32[512]", arg204_1: "f32[512]", arg205_1: "f32[512]", arg206_1: "f32[1536, 512, 1, 1]", arg207_1: "f32[15, 128]", arg208_1: "f32[15, 128]", arg209_1: "f32[512]", arg210_1: "f32[512]", arg211_1: "f32[512]", arg212_1: "f32[512]", arg213_1: "f32[1536, 512, 1, 1]", arg214_1: "f32[1536]", arg215_1: "f32[1536]", arg216_1: "f32[1536]", arg217_1: "f32[1536]", arg218_1: "f32[1280, 1536, 1, 1]", arg219_1: "f32[1280]", arg220_1: "f32[1280]", arg221_1: "f32[1280]", arg222_1: "f32[1280]", arg223_1: "f32[1000, 1280]", arg224_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_50: "f32[8, 24, 128, 128]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_94: "f32[24]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_38: "f32[24]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
        reciprocal_38: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_158: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_305: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        unsqueeze_307: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_42: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_305);  convolution_50 = unsqueeze_305 = None
        mul_159: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_307);  sub_42 = unsqueeze_307 = None
        unsqueeze_308: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_309: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_160: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_309);  mul_159 = unsqueeze_309 = None
        unsqueeze_310: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_311: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_95: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_311);  mul_160 = unsqueeze_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_40: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(add_95)
        mul_161: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(add_95, sigmoid_40);  add_95 = sigmoid_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_51: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_161, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_161 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_96: "f32[32]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_39: "f32[32]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        reciprocal_39: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_162: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_313: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
        unsqueeze_315: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_43: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_313);  convolution_51 = unsqueeze_313 = None
        mul_163: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_315);  sub_43 = unsqueeze_315 = None
        unsqueeze_316: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_317: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_164: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_317);  mul_163 = unsqueeze_317 = None
        unsqueeze_318: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_319: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_97: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_319);  mul_164 = unsqueeze_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_41: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(add_97)
        mul_165: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_97, sigmoid_41);  add_97 = sigmoid_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_52: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_165, arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_165 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_98: "f32[64]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_40: "f32[64]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_40: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_166: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_321: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_166, -1);  mul_166 = None
        unsqueeze_323: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_44: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_321);  convolution_52 = unsqueeze_321 = None
        mul_167: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_323);  sub_44 = unsqueeze_323 = None
        unsqueeze_324: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_325: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_168: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_325);  mul_167 = unsqueeze_325 = None
        unsqueeze_326: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_327: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_99: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_168, unsqueeze_327);  mul_168 = unsqueeze_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_42: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_99)
        mul_169: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_99, sigmoid_42);  add_99 = sigmoid_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_53: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_169, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_100: "f32[64]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_41: "f32[64]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        reciprocal_41: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_170: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_329: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_170, -1);  mul_170 = None
        unsqueeze_331: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_45: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_329);  convolution_53 = unsqueeze_329 = None
        mul_171: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_331);  sub_45 = unsqueeze_331 = None
        unsqueeze_332: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_333: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_172: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_333);  mul_171 = unsqueeze_333 = None
        unsqueeze_334: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_335: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_101: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_172, unsqueeze_335);  mul_172 = unsqueeze_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_43: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_101)
        mul_173: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_101, sigmoid_43);  add_101 = sigmoid_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_54: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_173, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_173 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_102: "f32[64]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_42: "f32[64]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_42: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_174: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_337: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_339: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_46: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_337);  convolution_54 = unsqueeze_337 = None
        mul_175: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_339);  sub_46 = unsqueeze_339 = None
        unsqueeze_340: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_341: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_176: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_341);  mul_175 = unsqueeze_341 = None
        unsqueeze_342: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_343: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_103: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_343);  mul_176 = unsqueeze_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_44: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_103)
        mul_177: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_103, sigmoid_44);  add_103 = sigmoid_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(mul_177, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_55: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_7, arg26_1, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg26_1 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_6: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_55);  convolution_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_56: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(relu_6, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_6 = arg28_1 = arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_45: "f32[8, 64, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_56);  convolution_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_178: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_177, sigmoid_45);  mul_177 = sigmoid_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_57: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_178, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_178 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_104: "f32[256]" = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_43: "f32[256]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_43: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_179: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_345: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
        unsqueeze_347: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_47: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_345);  convolution_57 = unsqueeze_345 = None
        mul_180: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_347);  sub_47 = unsqueeze_347 = None
        unsqueeze_348: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_349: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_181: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_349);  mul_180 = unsqueeze_349 = None
        unsqueeze_350: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_351: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_105: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_351);  mul_181 = unsqueeze_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_58: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_169, arg35_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_169 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_106: "f32[256]" = torch.ops.aten.add.Tensor(arg37_1, 1e-05);  arg37_1 = None
        sqrt_44: "f32[256]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_44: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_182: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_353: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
        unsqueeze_355: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_48: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_353);  convolution_58 = unsqueeze_353 = None
        mul_183: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_355);  sub_48 = unsqueeze_355 = None
        unsqueeze_356: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_357: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_184: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_357);  mul_183 = unsqueeze_357 = None
        unsqueeze_358: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_359: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_107: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_359);  mul_184 = unsqueeze_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_108: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_105, add_107);  add_105 = add_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_46: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_108)
        mul_185: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_108, sigmoid_46);  add_108 = sigmoid_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_59: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_185, arg40_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_109: "f32[64]" = torch.ops.aten.add.Tensor(arg42_1, 1e-05);  arg42_1 = None
        sqrt_45: "f32[64]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_45: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_186: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_361: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_363: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_49: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_361);  convolution_59 = unsqueeze_361 = None
        mul_187: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_363);  sub_49 = unsqueeze_363 = None
        unsqueeze_364: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_365: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_188: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_365);  mul_187 = unsqueeze_365 = None
        unsqueeze_366: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_367: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_110: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_367);  mul_188 = unsqueeze_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_47: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_110)
        mul_189: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_110, sigmoid_47);  add_110 = sigmoid_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_60: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_189, arg45_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_189 = arg45_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_111: "f32[64]" = torch.ops.aten.add.Tensor(arg47_1, 1e-05);  arg47_1 = None
        sqrt_46: "f32[64]" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_46: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_190: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_369: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_190, -1);  mul_190 = None
        unsqueeze_371: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_50: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_369);  convolution_60 = unsqueeze_369 = None
        mul_191: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_371);  sub_50 = unsqueeze_371 = None
        unsqueeze_372: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_373: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_192: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_373);  mul_191 = unsqueeze_373 = None
        unsqueeze_374: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_375: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_112: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_192, unsqueeze_375);  mul_192 = unsqueeze_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_48: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_112)
        mul_193: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_112, sigmoid_48);  add_112 = sigmoid_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(mul_193, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_61: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_8, arg50_1, arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg50_1 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_7: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_61);  convolution_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_62: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(relu_7, arg52_1, arg53_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg52_1 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_49: "f32[8, 64, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62);  convolution_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_194: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_193, sigmoid_49);  mul_193 = sigmoid_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_63: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_194, arg54_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_194 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_113: "f32[256]" = torch.ops.aten.add.Tensor(arg56_1, 1e-05);  arg56_1 = None
        sqrt_47: "f32[256]" = torch.ops.aten.sqrt.default(add_113);  add_113 = None
        reciprocal_47: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_195: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_377: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_379: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_51: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_377);  convolution_63 = unsqueeze_377 = None
        mul_196: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_379);  sub_51 = unsqueeze_379 = None
        unsqueeze_380: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_381: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_197: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_381);  mul_196 = unsqueeze_381 = None
        unsqueeze_382: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_383: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_114: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_383);  mul_197 = unsqueeze_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_115: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_114, mul_185);  add_114 = mul_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_50: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_115)
        mul_198: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_115, sigmoid_50);  add_115 = sigmoid_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_64: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_198, arg59_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg59_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_116: "f32[128]" = torch.ops.aten.add.Tensor(arg61_1, 1e-05);  arg61_1 = None
        sqrt_48: "f32[128]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_48: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_199: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_385: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
        unsqueeze_387: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_52: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_385);  convolution_64 = unsqueeze_385 = None
        mul_200: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_387);  sub_52 = unsqueeze_387 = None
        unsqueeze_388: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_389: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_201: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_389);  mul_200 = unsqueeze_389 = None
        unsqueeze_390: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_391: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_117: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_391);  mul_201 = unsqueeze_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_51: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_117)
        mul_202: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_117, sigmoid_51);  add_117 = sigmoid_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_65: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_202, arg64_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_202 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_118: "f32[128]" = torch.ops.aten.add.Tensor(arg66_1, 1e-05);  arg66_1 = None
        sqrt_49: "f32[128]" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_49: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_203: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_393: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_203, -1);  mul_203 = None
        unsqueeze_395: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_53: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_393);  convolution_65 = unsqueeze_393 = None
        mul_204: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_395);  sub_53 = unsqueeze_395 = None
        unsqueeze_396: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_397: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_205: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_397);  mul_204 = unsqueeze_397 = None
        unsqueeze_398: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_399: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_119: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_399);  mul_205 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_52: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_119)
        mul_206: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_119, sigmoid_52);  add_119 = sigmoid_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(mul_206, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_66: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_9, arg69_1, arg70_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg69_1 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_8: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_67: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_8, arg71_1, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg71_1 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_53: "f32[8, 128, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_207: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_206, sigmoid_53);  mul_206 = sigmoid_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_68: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_207, arg73_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_207 = arg73_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_120: "f32[512]" = torch.ops.aten.add.Tensor(arg75_1, 1e-05);  arg75_1 = None
        sqrt_50: "f32[512]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_50: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_208: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_401: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
        unsqueeze_403: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_54: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_401);  convolution_68 = unsqueeze_401 = None
        mul_209: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_403);  sub_54 = unsqueeze_403 = None
        unsqueeze_404: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_405: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_210: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_405);  mul_209 = unsqueeze_405 = None
        unsqueeze_406: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_407: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_121: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_407);  mul_210 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_69: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_198, arg78_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_198 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_122: "f32[512]" = torch.ops.aten.add.Tensor(arg80_1, 1e-05);  arg80_1 = None
        sqrt_51: "f32[512]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
        reciprocal_51: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_211: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_409: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_211, -1);  mul_211 = None
        unsqueeze_411: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_55: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_409);  convolution_69 = unsqueeze_409 = None
        mul_212: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_411);  sub_55 = unsqueeze_411 = None
        unsqueeze_412: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_413: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_213: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_413);  mul_212 = unsqueeze_413 = None
        unsqueeze_414: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_415: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_123: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_213, unsqueeze_415);  mul_213 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_124: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_121, add_123);  add_121 = add_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_54: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_124)
        mul_214: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_124, sigmoid_54);  add_124 = sigmoid_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_70: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_214, arg83_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_125: "f32[128]" = torch.ops.aten.add.Tensor(arg85_1, 1e-05);  arg85_1 = None
        sqrt_52: "f32[128]" = torch.ops.aten.sqrt.default(add_125);  add_125 = None
        reciprocal_52: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_215: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_417: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_215, -1);  mul_215 = None
        unsqueeze_419: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_56: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_417);  convolution_70 = unsqueeze_417 = None
        mul_216: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_419);  sub_56 = unsqueeze_419 = None
        unsqueeze_420: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_421: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_217: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_216, unsqueeze_421);  mul_216 = unsqueeze_421 = None
        unsqueeze_422: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_423: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_126: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_217, unsqueeze_423);  mul_217 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_55: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_126)
        mul_218: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_126, sigmoid_55);  add_126 = sigmoid_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_71: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_218, arg88_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_218 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_127: "f32[128]" = torch.ops.aten.add.Tensor(arg90_1, 1e-05);  arg90_1 = None
        sqrt_53: "f32[128]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_53: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_219: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_425: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_427: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_57: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_425);  convolution_71 = unsqueeze_425 = None
        mul_220: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_427);  sub_57 = unsqueeze_427 = None
        unsqueeze_428: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_429: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_221: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_429);  mul_220 = unsqueeze_429 = None
        unsqueeze_430: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_431: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_128: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_431);  mul_221 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_56: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_128)
        mul_222: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_128, sigmoid_56);  add_128 = sigmoid_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(mul_222, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_72: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_10, arg93_1, arg94_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg93_1 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_9: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_72);  convolution_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_73: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_9, arg95_1, arg96_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg95_1 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_57: "f32[8, 128, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_223: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_222, sigmoid_57);  mul_222 = sigmoid_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_74: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_223, arg97_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_223 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_129: "f32[512]" = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_54: "f32[512]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_54: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_224: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_433: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_224, -1);  mul_224 = None
        unsqueeze_435: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_58: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_433);  convolution_74 = unsqueeze_433 = None
        mul_225: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_435);  sub_58 = unsqueeze_435 = None
        unsqueeze_436: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_437: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_226: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_437);  mul_225 = unsqueeze_437 = None
        unsqueeze_438: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_439: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_130: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_439);  mul_226 = unsqueeze_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_131: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_130, mul_214);  add_130 = mul_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_58: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_131)
        mul_227: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_131, sigmoid_58);  add_131 = sigmoid_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_75: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_227, arg102_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_132: "f32[128]" = torch.ops.aten.add.Tensor(arg104_1, 1e-05);  arg104_1 = None
        sqrt_55: "f32[128]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_55: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_228: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_441: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_443: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_59: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_441);  convolution_75 = unsqueeze_441 = None
        mul_229: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_443);  sub_59 = unsqueeze_443 = None
        unsqueeze_444: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_445: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_230: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_445);  mul_229 = unsqueeze_445 = None
        unsqueeze_446: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_447: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_133: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_447);  mul_230 = unsqueeze_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_59: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_133)
        mul_231: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_133, sigmoid_59);  add_133 = sigmoid_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:140 in forward, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_76: "f32[8, 384, 32, 32]" = torch.ops.aten.convolution.default(mul_231, arg107_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_231 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:144 in forward, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(convolution_76, [128, 128, 128], 1);  convolution_76 = None
        getitem_12: "f32[8, 128, 32, 32]" = split_with_sizes_4[0]
        getitem_13: "f32[8, 128, 32, 32]" = split_with_sizes_4[1]
        getitem_14: "f32[8, 128, 32, 32]" = split_with_sizes_4[2];  split_with_sizes_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:145 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        clone_29: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_12, memory_format = torch.contiguous_format);  getitem_12 = None
        view_97: "f32[32, 32, 1024]" = torch.ops.aten.view.default(clone_29, [32, 32, 1024]);  clone_29 = None
        permute_33: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:146 in forward, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        clone_30: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_13, memory_format = torch.contiguous_format);  getitem_13 = None
        view_98: "f32[32, 32, 1024]" = torch.ops.aten.view.default(clone_30, [32, 32, 1024]);  clone_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:147 in forward, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        clone_31: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_14, memory_format = torch.contiguous_format);  getitem_14 = None
        view_99: "f32[32, 32, 1024]" = torch.ops.aten.view.default(clone_31, [32, 32, 1024]);  clone_31 = None
        permute_34: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_24: "f32[32, 1024, 32]" = torch.ops.aten.expand.default(permute_33, [32, 1024, 32])
        expand_25: "f32[32, 32, 1024]" = torch.ops.aten.expand.default(view_98, [32, 32, 1024]);  view_98 = None
        bmm_8: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(expand_24, expand_25);  expand_24 = expand_25 = None
        mul_232: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(bmm_8, 0.1767766952966369);  bmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:72 in forward, code: q = q.reshape(B, self.height, self.width, -1)
        view_103: "f32[32, 32, 32, 32]" = torch.ops.aten.view.default(permute_33, [32, 32, 32, 32]);  permute_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        permute_35: "f32[32, 63]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        clone_32: "f32[32, 32, 32, 32]" = torch.ops.aten.clone.default(view_103, memory_format = torch.contiguous_format)
        view_104: "f32[32768, 32]" = torch.ops.aten.view.default(clone_32, [32768, 32]);  clone_32 = None
        mm_8: "f32[32768, 63]" = torch.ops.aten.mm.default(view_104, permute_35);  view_104 = permute_35 = None
        view_105: "f32[32, 32, 32, 63]" = torch.ops.aten.view.default(mm_8, [32, 32, 32, 63]);  mm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_106: "f32[1024, 32, 63]" = torch.ops.aten.view.default(view_105, [-1, 32, 63]);  view_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_16: "f32[1024, 32, 64]" = torch.ops.aten.constant_pad_nd.default(view_106, [0, 1], 0.0);  view_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_107: "f32[1024, 2048]" = torch.ops.aten.view.default(constant_pad_nd_16, [1024, 2048]);  constant_pad_nd_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_17: "f32[1024, 2079]" = torch.ops.aten.constant_pad_nd.default(view_107, [0, 31], 0.0);  view_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_108: "f32[1024, 33, 63]" = torch.ops.aten.view.default(constant_pad_nd_17, [-1, 33, 63]);  constant_pad_nd_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_26: "f32[1024, 32, 63]" = torch.ops.aten.slice.Tensor(view_108, 1, 0, 32);  view_108 = None
        slice_27: "f32[1024, 32, 32]" = torch.ops.aten.slice.Tensor(slice_26, 2, 31, 9223372036854775807);  slice_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_109: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.view.default(slice_27, [32, 32, 1, 32, 32]);  slice_27 = None
        expand_26: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.expand.default(view_109, [-1, -1, 32, -1, -1]);  view_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_36: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(expand_26, [0, 1, 3, 2, 4]);  expand_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:76 in forward, code: q = q.transpose(1, 2)
        permute_37: "f32[32, 32, 32, 32]" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        permute_38: "f32[32, 63]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        clone_33: "f32[32, 32, 32, 32]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_110: "f32[32768, 32]" = torch.ops.aten.view.default(clone_33, [32768, 32]);  clone_33 = None
        mm_9: "f32[32768, 63]" = torch.ops.aten.mm.default(view_110, permute_38);  view_110 = permute_38 = None
        view_111: "f32[32, 32, 32, 63]" = torch.ops.aten.view.default(mm_9, [32, 32, 32, 63]);  mm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_112: "f32[1024, 32, 63]" = torch.ops.aten.view.default(view_111, [-1, 32, 63]);  view_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_18: "f32[1024, 32, 64]" = torch.ops.aten.constant_pad_nd.default(view_112, [0, 1], 0.0);  view_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_113: "f32[1024, 2048]" = torch.ops.aten.view.default(constant_pad_nd_18, [1024, 2048]);  constant_pad_nd_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_19: "f32[1024, 2079]" = torch.ops.aten.constant_pad_nd.default(view_113, [0, 31], 0.0);  view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_114: "f32[1024, 33, 63]" = torch.ops.aten.view.default(constant_pad_nd_19, [-1, 33, 63]);  constant_pad_nd_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_29: "f32[1024, 32, 63]" = torch.ops.aten.slice.Tensor(view_114, 1, 0, 32);  view_114 = None
        slice_30: "f32[1024, 32, 32]" = torch.ops.aten.slice.Tensor(slice_29, 2, 31, 9223372036854775807);  slice_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_115: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.view.default(slice_30, [32, 32, 1, 32, 32]);  slice_30 = None
        expand_27: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.expand.default(view_115, [-1, -1, 32, -1, -1]);  view_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_39: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(expand_27, [0, 3, 1, 4, 2]);  expand_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:79 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_134: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.add.Tensor(permute_39, permute_36);  permute_39 = permute_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:80 in forward, code: rel_logits = rel_logits.reshape(B, HW, HW)
        clone_34: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format);  add_134 = None
        view_116: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(clone_34, [32, 1024, 1024]);  clone_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_135: "f32[32, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_232, view_116);  mul_232 = view_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:153 in forward, code: attn = attn.softmax(dim=-1)
        amax_4: "f32[32, 1024, 1]" = torch.ops.aten.amax.default(add_135, [-1], True)
        sub_60: "f32[32, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_135, amax_4);  add_135 = amax_4 = None
        exp_4: "f32[32, 1024, 1024]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_5: "f32[32, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_28: "f32[32, 1024, 1024]" = torch.ops.aten.expand.default(div_4, [32, 1024, 1024]);  div_4 = None
        expand_29: "f32[32, 1024, 32]" = torch.ops.aten.expand.default(permute_34, [32, 1024, 32]);  permute_34 = None
        bmm_9: "f32[32, 1024, 32]" = torch.ops.aten.bmm.default(expand_28, expand_29);  expand_28 = expand_29 = None
        permute_40: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(bmm_9, [0, 2, 1]);  bmm_9 = None
        clone_35: "f32[32, 32, 1024]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_120: "f32[8, 128, 32, 32]" = torch.ops.aten.view.default(clone_35, [8, 128, 32, 32]);  clone_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_136: "f32[128]" = torch.ops.aten.add.Tensor(arg111_1, 1e-05);  arg111_1 = None
        sqrt_56: "f32[128]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_56: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_233: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_449: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_233, -1);  mul_233 = None
        unsqueeze_451: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_61: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_449);  view_120 = unsqueeze_449 = None
        mul_234: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_451);  sub_61 = unsqueeze_451 = None
        unsqueeze_452: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_453: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_235: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_453);  mul_234 = unsqueeze_453 = None
        unsqueeze_454: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_455: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_137: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_455);  mul_235 = unsqueeze_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_60: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_137)
        mul_236: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_60);  add_137 = sigmoid_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_77: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_236, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_236 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_138: "f32[512]" = torch.ops.aten.add.Tensor(arg116_1, 1e-05);  arg116_1 = None
        sqrt_57: "f32[512]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_57: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_237: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_457: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_459: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_62: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_457);  convolution_77 = unsqueeze_457 = None
        mul_238: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_459);  sub_62 = unsqueeze_459 = None
        unsqueeze_460: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_461: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_239: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_461);  mul_238 = unsqueeze_461 = None
        unsqueeze_462: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_463: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_139: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_463);  mul_239 = unsqueeze_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_140: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_139, mul_227);  add_139 = mul_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        sigmoid_61: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_140)
        mul_240: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_140, sigmoid_61);  add_140 = sigmoid_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_78: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_240, arg119_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg119_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_141: "f32[256]" = torch.ops.aten.add.Tensor(arg121_1, 1e-05);  arg121_1 = None
        sqrt_58: "f32[256]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_58: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_241: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_465: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_241, -1);  mul_241 = None
        unsqueeze_467: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_63: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_465);  convolution_78 = unsqueeze_465 = None
        mul_242: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_467);  sub_63 = unsqueeze_467 = None
        unsqueeze_468: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_469: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_243: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_469);  mul_242 = unsqueeze_469 = None
        unsqueeze_470: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_471: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_142: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_243, unsqueeze_471);  mul_243 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_62: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_142)
        mul_244: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_142, sigmoid_62);  add_142 = sigmoid_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_79: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_244, arg124_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_244 = arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_143: "f32[256]" = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
        sqrt_59: "f32[256]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_59: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_245: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_473: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_245, -1);  mul_245 = None
        unsqueeze_475: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_64: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_473);  convolution_79 = unsqueeze_473 = None
        mul_246: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_475);  sub_64 = unsqueeze_475 = None
        unsqueeze_476: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_477: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_247: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_477);  mul_246 = unsqueeze_477 = None
        unsqueeze_478: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_479: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_144: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_247, unsqueeze_479);  mul_247 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_63: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_144)
        mul_248: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_144, sigmoid_63);  add_144 = sigmoid_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(mul_248, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_80: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_11, arg129_1, arg130_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg129_1 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_10: "f32[8, 16, 1, 1]" = torch.ops.aten.relu.default(convolution_80);  convolution_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_81: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_10, arg131_1, arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_10 = arg131_1 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_64: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_81);  convolution_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_249: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_248, sigmoid_64);  mul_248 = sigmoid_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_82: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_249, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_249 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_145: "f32[1024]" = torch.ops.aten.add.Tensor(arg135_1, 1e-05);  arg135_1 = None
        sqrt_60: "f32[1024]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_60: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_250: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_481: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_483: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_65: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_481);  convolution_82 = unsqueeze_481 = None
        mul_251: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_483);  sub_65 = unsqueeze_483 = None
        unsqueeze_484: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_485: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_252: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_485);  mul_251 = unsqueeze_485 = None
        unsqueeze_486: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_487: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_146: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_487);  mul_252 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_83: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_240, arg138_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_240 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_147: "f32[1024]" = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_61: "f32[1024]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_61: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_253: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_489: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_491: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_66: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_489);  convolution_83 = unsqueeze_489 = None
        mul_254: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_491);  sub_66 = unsqueeze_491 = None
        unsqueeze_492: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_493: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_255: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_493);  mul_254 = unsqueeze_493 = None
        unsqueeze_494: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_495: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_148: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_495);  mul_255 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_149: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_146, add_148);  add_146 = add_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_65: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_149)
        mul_256: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_149, sigmoid_65);  add_149 = sigmoid_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_84: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_256, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_150: "f32[256]" = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_62: "f32[256]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_62: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_257: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_497: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_257, -1);  mul_257 = None
        unsqueeze_499: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_67: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_497);  convolution_84 = unsqueeze_497 = None
        mul_258: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_499);  sub_67 = unsqueeze_499 = None
        unsqueeze_500: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_501: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_259: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_501);  mul_258 = unsqueeze_501 = None
        unsqueeze_502: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_503: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_151: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_259, unsqueeze_503);  mul_259 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_66: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_151)
        mul_260: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_151, sigmoid_66);  add_151 = sigmoid_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_85: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_260, arg148_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_260 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_152: "f32[256]" = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_63: "f32[256]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_63: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_261: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_505: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_507: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_68: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_505);  convolution_85 = unsqueeze_505 = None
        mul_262: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_507);  sub_68 = unsqueeze_507 = None
        unsqueeze_508: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_509: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_263: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_509);  mul_262 = unsqueeze_509 = None
        unsqueeze_510: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_511: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_153: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_511);  mul_263 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_67: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_153)
        mul_264: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_153, sigmoid_67);  add_153 = sigmoid_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_12: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(mul_264, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_86: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_12, arg153_1, arg154_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_12 = arg153_1 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_11: "f32[8, 16, 1, 1]" = torch.ops.aten.relu.default(convolution_86);  convolution_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_87: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_11, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg155_1 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_68: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_265: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_264, sigmoid_68);  mul_264 = sigmoid_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_88: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_265, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_265 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_154: "f32[1024]" = torch.ops.aten.add.Tensor(arg159_1, 1e-05);  arg159_1 = None
        sqrt_64: "f32[1024]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_64: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_266: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_513: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_266, -1);  mul_266 = None
        unsqueeze_515: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_69: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_513);  convolution_88 = unsqueeze_513 = None
        mul_267: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_515);  sub_69 = unsqueeze_515 = None
        unsqueeze_516: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_517: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_268: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_267, unsqueeze_517);  mul_267 = unsqueeze_517 = None
        unsqueeze_518: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_519: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_155: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_268, unsqueeze_519);  mul_268 = unsqueeze_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_156: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_155, mul_256);  add_155 = mul_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_69: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_156)
        mul_269: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_156, sigmoid_69);  add_156 = sigmoid_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_89: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_269, arg162_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_157: "f32[256]" = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_65: "f32[256]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_65: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_270: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_521: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_523: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_70: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_521);  convolution_89 = unsqueeze_521 = None
        mul_271: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_523);  sub_70 = unsqueeze_523 = None
        unsqueeze_524: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_525: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_272: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_525);  mul_271 = unsqueeze_525 = None
        unsqueeze_526: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_527: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_158: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_527);  mul_272 = unsqueeze_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_70: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_158)
        mul_273: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_158, sigmoid_70);  add_158 = sigmoid_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:140 in forward, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_90: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_273, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_273 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:144 in forward, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(convolution_90, [256, 256, 256], 1);  convolution_90 = None
        getitem_15: "f32[8, 256, 16, 16]" = split_with_sizes_5[0]
        getitem_16: "f32[8, 256, 16, 16]" = split_with_sizes_5[1]
        getitem_17: "f32[8, 256, 16, 16]" = split_with_sizes_5[2];  split_with_sizes_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:145 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        clone_36: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format);  getitem_15 = None
        view_121: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_36, [32, 64, 256]);  clone_36 = None
        permute_41: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:146 in forward, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        clone_37: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_16, memory_format = torch.contiguous_format);  getitem_16 = None
        view_122: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_37, [32, 64, 256]);  clone_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:147 in forward, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        clone_38: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_17, memory_format = torch.contiguous_format);  getitem_17 = None
        view_123: "f32[32, 64, 256]" = torch.ops.aten.view.default(clone_38, [32, 64, 256]);  clone_38 = None
        permute_42: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_30: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_41, [32, 256, 64])
        expand_31: "f32[32, 64, 256]" = torch.ops.aten.expand.default(view_122, [32, 64, 256]);  view_122 = None
        bmm_10: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(expand_30, expand_31);  expand_30 = expand_31 = None
        mul_274: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_10, 0.125);  bmm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:72 in forward, code: q = q.reshape(B, self.height, self.width, -1)
        view_127: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(permute_41, [32, 16, 16, 64]);  permute_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        permute_43: "f32[64, 31]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        clone_39: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(view_127, memory_format = torch.contiguous_format)
        view_128: "f32[8192, 64]" = torch.ops.aten.view.default(clone_39, [8192, 64]);  clone_39 = None
        mm_10: "f32[8192, 31]" = torch.ops.aten.mm.default(view_128, permute_43);  view_128 = permute_43 = None
        view_129: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_10, [32, 16, 16, 31]);  mm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_130: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_129, [-1, 16, 31]);  view_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_20: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_130, [0, 1], 0.0);  view_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_131: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_20, [512, 512]);  constant_pad_nd_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_21: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_131, [0, 15], 0.0);  view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_132: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_21, [-1, 17, 31]);  constant_pad_nd_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_32: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_132, 1, 0, 16);  view_132 = None
        slice_33: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_32, 2, 15, 9223372036854775807);  slice_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_133: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_33, [32, 16, 1, 16, 16]);  slice_33 = None
        expand_32: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_133, [-1, -1, 16, -1, -1]);  view_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_44: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_32, [0, 1, 3, 2, 4]);  expand_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:76 in forward, code: q = q.transpose(1, 2)
        permute_45: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        permute_46: "f32[64, 31]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        clone_40: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_134: "f32[8192, 64]" = torch.ops.aten.view.default(clone_40, [8192, 64]);  clone_40 = None
        mm_11: "f32[8192, 31]" = torch.ops.aten.mm.default(view_134, permute_46);  view_134 = permute_46 = None
        view_135: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_11, [32, 16, 16, 31]);  mm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_136: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_135, [-1, 16, 31]);  view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_22: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_136, [0, 1], 0.0);  view_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_137: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_22, [512, 512]);  constant_pad_nd_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_23: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_137, [0, 15], 0.0);  view_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_138: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_23, [-1, 17, 31]);  constant_pad_nd_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_35: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_138, 1, 0, 16);  view_138 = None
        slice_36: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_35, 2, 15, 9223372036854775807);  slice_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_139: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_36, [32, 16, 1, 16, 16]);  slice_36 = None
        expand_33: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_139, [-1, -1, 16, -1, -1]);  view_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_47: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_33, [0, 3, 1, 4, 2]);  expand_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:79 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_159: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_47, permute_44);  permute_47 = permute_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:80 in forward, code: rel_logits = rel_logits.reshape(B, HW, HW)
        clone_41: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_159, memory_format = torch.contiguous_format);  add_159 = None
        view_140: "f32[32, 256, 256]" = torch.ops.aten.view.default(clone_41, [32, 256, 256]);  clone_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_160: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_274, view_140);  mul_274 = view_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:153 in forward, code: attn = attn.softmax(dim=-1)
        amax_5: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_160, [-1], True)
        sub_71: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_160, amax_5);  add_160 = amax_5 = None
        exp_5: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        sum_6: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_34: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_5, [32, 256, 256]);  div_5 = None
        expand_35: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_42, [32, 256, 64]);  permute_42 = None
        bmm_11: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(expand_34, expand_35);  expand_34 = expand_35 = None
        permute_48: "f32[32, 64, 256]" = torch.ops.aten.permute.default(bmm_11, [0, 2, 1]);  bmm_11 = None
        clone_42: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_144: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_42, [8, 256, 16, 16]);  clone_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_161: "f32[256]" = torch.ops.aten.add.Tensor(arg171_1, 1e-05);  arg171_1 = None
        sqrt_66: "f32[256]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_66: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_275: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_529: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_275, -1);  mul_275 = None
        unsqueeze_531: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_72: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_529);  view_144 = unsqueeze_529 = None
        mul_276: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_531);  sub_72 = unsqueeze_531 = None
        unsqueeze_532: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_533: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_277: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_533);  mul_276 = unsqueeze_533 = None
        unsqueeze_534: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_535: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_162: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_277, unsqueeze_535);  mul_277 = unsqueeze_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_71: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_162)
        mul_278: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_162, sigmoid_71);  add_162 = sigmoid_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_91: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_278, arg174_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_278 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_163: "f32[1024]" = torch.ops.aten.add.Tensor(arg176_1, 1e-05);  arg176_1 = None
        sqrt_67: "f32[1024]" = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_67: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_279: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_537: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
        unsqueeze_539: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_73: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_537);  convolution_91 = unsqueeze_537 = None
        mul_280: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_539);  sub_73 = unsqueeze_539 = None
        unsqueeze_540: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_541: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_281: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_541);  mul_280 = unsqueeze_541 = None
        unsqueeze_542: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_543: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_164: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_543);  mul_281 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_165: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_164, mul_269);  add_164 = mul_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        sigmoid_72: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_165)
        mul_282: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_165, sigmoid_72);  add_165 = sigmoid_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_92: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_282, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg179_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_166: "f32[512]" = torch.ops.aten.add.Tensor(arg181_1, 1e-05);  arg181_1 = None
        sqrt_68: "f32[512]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_68: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_545: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_547: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_74: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_545);  convolution_92 = unsqueeze_545 = None
        mul_284: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_547);  sub_74 = unsqueeze_547 = None
        unsqueeze_548: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_549: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_285: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_549);  mul_284 = unsqueeze_549 = None
        unsqueeze_550: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_551: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_167: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_285, unsqueeze_551);  mul_285 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_73: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_167)
        mul_286: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_167, sigmoid_73);  add_167 = sigmoid_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:140 in forward, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_93: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_286, arg184_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_286 = arg184_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:144 in forward, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(convolution_93, [512, 512, 512], 1);  convolution_93 = None
        getitem_18: "f32[8, 512, 16, 16]" = split_with_sizes_6[0]
        getitem_19: "f32[8, 512, 16, 16]" = split_with_sizes_6[1]
        getitem_20: "f32[8, 512, 16, 16]" = split_with_sizes_6[2];  split_with_sizes_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:145 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        clone_43: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_18, memory_format = torch.contiguous_format);  getitem_18 = None
        view_145: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_43, [32, 128, 256]);  clone_43 = None
        permute_49: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:146 in forward, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        clone_44: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_19, memory_format = torch.contiguous_format);  getitem_19 = None
        view_146: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_44, [32, 128, 256]);  clone_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:147 in forward, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        clone_45: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_20, memory_format = torch.contiguous_format);  getitem_20 = None
        view_147: "f32[32, 128, 256]" = torch.ops.aten.view.default(clone_45, [32, 128, 256]);  clone_45 = None
        permute_50: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_36: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_49, [32, 256, 128])
        expand_37: "f32[32, 128, 256]" = torch.ops.aten.expand.default(view_146, [32, 128, 256]);  view_146 = None
        bmm_12: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(expand_36, expand_37);  expand_36 = expand_37 = None
        mul_287: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_12, 0.08838834764831845);  bmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:72 in forward, code: q = q.reshape(B, self.height, self.width, -1)
        view_151: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(permute_49, [32, 16, 16, 128]);  permute_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        permute_51: "f32[128, 31]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        clone_46: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(view_151, memory_format = torch.contiguous_format)
        view_152: "f32[8192, 128]" = torch.ops.aten.view.default(clone_46, [8192, 128]);  clone_46 = None
        mm_12: "f32[8192, 31]" = torch.ops.aten.mm.default(view_152, permute_51);  view_152 = permute_51 = None
        view_153: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_12, [32, 16, 16, 31]);  mm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_154: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_153, [-1, 16, 31]);  view_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_24: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_154, [0, 1], 0.0);  view_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_155: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_24, [512, 512]);  constant_pad_nd_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_25: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_155, [0, 15], 0.0);  view_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_156: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_25, [-1, 17, 31]);  constant_pad_nd_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_38: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_156, 1, 0, 16);  view_156 = None
        slice_39: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_38, 2, 15, 9223372036854775807);  slice_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_157: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_39, [32, 16, 1, 16, 16]);  slice_39 = None
        expand_38: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_157, [-1, -1, 16, -1, -1]);  view_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_52: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_38, [0, 1, 3, 2, 4]);  expand_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:76 in forward, code: q = q.transpose(1, 2)
        permute_53: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        permute_54: "f32[128, 31]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        clone_47: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
        view_158: "f32[8192, 128]" = torch.ops.aten.view.default(clone_47, [8192, 128]);  clone_47 = None
        mm_13: "f32[8192, 31]" = torch.ops.aten.mm.default(view_158, permute_54);  view_158 = permute_54 = None
        view_159: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(mm_13, [32, 16, 16, 31]);  mm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_160: "f32[512, 16, 31]" = torch.ops.aten.view.default(view_159, [-1, 16, 31]);  view_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_26: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_160, [0, 1], 0.0);  view_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_161: "f32[512, 512]" = torch.ops.aten.view.default(constant_pad_nd_26, [512, 512]);  constant_pad_nd_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_27: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_161, [0, 15], 0.0);  view_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_162: "f32[512, 17, 31]" = torch.ops.aten.view.default(constant_pad_nd_27, [-1, 17, 31]);  constant_pad_nd_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_41: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_162, 1, 0, 16);  view_162 = None
        slice_42: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_41, 2, 15, 9223372036854775807);  slice_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_163: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.view.default(slice_42, [32, 16, 1, 16, 16]);  slice_42 = None
        expand_39: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_163, [-1, -1, 16, -1, -1]);  view_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_55: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_39, [0, 3, 1, 4, 2]);  expand_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:79 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_168: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_55, permute_52);  permute_55 = permute_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:80 in forward, code: rel_logits = rel_logits.reshape(B, HW, HW)
        clone_48: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_168, memory_format = torch.contiguous_format);  add_168 = None
        view_164: "f32[32, 256, 256]" = torch.ops.aten.view.default(clone_48, [32, 256, 256]);  clone_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_169: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_287, view_164);  mul_287 = view_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:153 in forward, code: attn = attn.softmax(dim=-1)
        amax_6: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_169, [-1], True)
        sub_75: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_169, amax_6);  add_169 = amax_6 = None
        exp_6: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
        sum_7: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_40: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_6, [32, 256, 256]);  div_6 = None
        expand_41: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_50, [32, 256, 128]);  permute_50 = None
        bmm_13: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(expand_40, expand_41);  expand_40 = expand_41 = None
        permute_56: "f32[32, 128, 256]" = torch.ops.aten.permute.default(bmm_13, [0, 2, 1]);  bmm_13 = None
        clone_49: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_168: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_49, [8, 512, 16, 16]);  clone_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:156 in forward, code: out = self.pool(out)
        avg_pool2d_1: "f32[8, 512, 8, 8]" = torch.ops.aten.avg_pool2d.default(view_168, [2, 2], [2, 2]);  view_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_170: "f32[512]" = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_69: "f32[512]" = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_69: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_288: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_553: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_555: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_76: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d_1, unsqueeze_553);  avg_pool2d_1 = unsqueeze_553 = None
        mul_289: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_555);  sub_76 = unsqueeze_555 = None
        unsqueeze_556: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_557: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_290: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_557);  mul_289 = unsqueeze_557 = None
        unsqueeze_558: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_559: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_171: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_559);  mul_290 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_74: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_171)
        mul_291: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_171, sigmoid_74);  add_171 = sigmoid_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_94: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_291, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_291 = arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_172: "f32[1536]" = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_70: "f32[1536]" = torch.ops.aten.sqrt.default(add_172);  add_172 = None
        reciprocal_70: "f32[1536]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_292: "f32[1536]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_561: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_563: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_77: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_561);  convolution_94 = unsqueeze_561 = None
        mul_293: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_563);  sub_77 = unsqueeze_563 = None
        unsqueeze_564: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_565: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_294: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_565);  mul_293 = unsqueeze_565 = None
        unsqueeze_566: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_567: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_173: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_567);  mul_294 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_95: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_282, arg196_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_282 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_174: "f32[1536]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_71: "f32[1536]" = torch.ops.aten.sqrt.default(add_174);  add_174 = None
        reciprocal_71: "f32[1536]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_295: "f32[1536]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_569: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_571: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_78: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_569);  convolution_95 = unsqueeze_569 = None
        mul_296: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_571);  sub_78 = unsqueeze_571 = None
        unsqueeze_572: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_573: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_297: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_573);  mul_296 = unsqueeze_573 = None
        unsqueeze_574: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_575: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_175: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_297, unsqueeze_575);  mul_297 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_176: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(add_173, add_175);  add_173 = add_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        sigmoid_75: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(add_176)
        mul_298: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_176, sigmoid_75);  add_176 = sigmoid_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_96: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_298, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_177: "f32[512]" = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_72: "f32[512]" = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_72: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_299: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_577: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_299, -1);  mul_299 = None
        unsqueeze_579: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_79: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_577);  convolution_96 = unsqueeze_577 = None
        mul_300: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_579);  sub_79 = unsqueeze_579 = None
        unsqueeze_580: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_581: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_301: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_581);  mul_300 = unsqueeze_581 = None
        unsqueeze_582: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_583: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_178: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_301, unsqueeze_583);  mul_301 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_76: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_178)
        mul_302: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_178, sigmoid_76);  add_178 = sigmoid_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:140 in forward, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_97: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_302, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_302 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:144 in forward, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(convolution_97, [512, 512, 512], 1);  convolution_97 = None
        getitem_21: "f32[8, 512, 8, 8]" = split_with_sizes_7[0]
        getitem_22: "f32[8, 512, 8, 8]" = split_with_sizes_7[1]
        getitem_23: "f32[8, 512, 8, 8]" = split_with_sizes_7[2];  split_with_sizes_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:145 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        clone_50: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_21, memory_format = torch.contiguous_format);  getitem_21 = None
        view_169: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_50, [32, 128, 64]);  clone_50 = None
        permute_57: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:146 in forward, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        clone_51: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_22, memory_format = torch.contiguous_format);  getitem_22 = None
        view_170: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_51, [32, 128, 64]);  clone_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:147 in forward, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        clone_52: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_23, memory_format = torch.contiguous_format);  getitem_23 = None
        view_171: "f32[32, 128, 64]" = torch.ops.aten.view.default(clone_52, [32, 128, 64]);  clone_52 = None
        permute_58: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_42: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_57, [32, 64, 128])
        expand_43: "f32[32, 128, 64]" = torch.ops.aten.expand.default(view_170, [32, 128, 64]);  view_170 = None
        bmm_14: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(expand_42, expand_43);  expand_42 = expand_43 = None
        mul_303: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(bmm_14, 0.08838834764831845);  bmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:72 in forward, code: q = q.reshape(B, self.height, self.width, -1)
        view_175: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(permute_57, [32, 8, 8, 128]);  permute_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        permute_59: "f32[128, 15]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        clone_53: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(view_175, memory_format = torch.contiguous_format)
        view_176: "f32[2048, 128]" = torch.ops.aten.view.default(clone_53, [2048, 128]);  clone_53 = None
        mm_14: "f32[2048, 15]" = torch.ops.aten.mm.default(view_176, permute_59);  view_176 = permute_59 = None
        view_177: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(mm_14, [32, 8, 8, 15]);  mm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_178: "f32[256, 8, 15]" = torch.ops.aten.view.default(view_177, [-1, 8, 15]);  view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_28: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_178, [0, 1], 0.0);  view_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_179: "f32[256, 128]" = torch.ops.aten.view.default(constant_pad_nd_28, [256, 128]);  constant_pad_nd_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_29: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_179, [0, 7], 0.0);  view_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_180: "f32[256, 9, 15]" = torch.ops.aten.view.default(constant_pad_nd_29, [-1, 9, 15]);  constant_pad_nd_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_44: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(view_180, 1, 0, 8);  view_180 = None
        slice_45: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_44, 2, 7, 9223372036854775807);  slice_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_181: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.view.default(slice_45, [32, 8, 1, 8, 8]);  slice_45 = None
        expand_44: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_181, [-1, -1, 8, -1, -1]);  view_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_60: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_44, [0, 1, 3, 2, 4]);  expand_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:76 in forward, code: q = q.transpose(1, 2)
        permute_61: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        permute_62: "f32[128, 15]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        clone_54: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_182: "f32[2048, 128]" = torch.ops.aten.view.default(clone_54, [2048, 128]);  clone_54 = None
        mm_15: "f32[2048, 15]" = torch.ops.aten.mm.default(view_182, permute_62);  view_182 = permute_62 = None
        view_183: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(mm_15, [32, 8, 8, 15]);  mm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_184: "f32[256, 8, 15]" = torch.ops.aten.view.default(view_183, [-1, 8, 15]);  view_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_30: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_184, [0, 1], 0.0);  view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_185: "f32[256, 128]" = torch.ops.aten.view.default(constant_pad_nd_30, [256, 128]);  constant_pad_nd_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_31: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_185, [0, 7], 0.0);  view_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_186: "f32[256, 9, 15]" = torch.ops.aten.view.default(constant_pad_nd_31, [-1, 9, 15]);  constant_pad_nd_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_47: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(view_186, 1, 0, 8);  view_186 = None
        slice_48: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_47, 2, 7, 9223372036854775807);  slice_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_187: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.view.default(slice_48, [32, 8, 1, 8, 8]);  slice_48 = None
        expand_45: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_187, [-1, -1, 8, -1, -1]);  view_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_63: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_45, [0, 3, 1, 4, 2]);  expand_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:79 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_179: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.add.Tensor(permute_63, permute_60);  permute_63 = permute_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:80 in forward, code: rel_logits = rel_logits.reshape(B, HW, HW)
        clone_55: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.clone.default(add_179, memory_format = torch.contiguous_format);  add_179 = None
        view_188: "f32[32, 64, 64]" = torch.ops.aten.view.default(clone_55, [32, 64, 64]);  clone_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_180: "f32[32, 64, 64]" = torch.ops.aten.add.Tensor(mul_303, view_188);  mul_303 = view_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:153 in forward, code: attn = attn.softmax(dim=-1)
        amax_7: "f32[32, 64, 1]" = torch.ops.aten.amax.default(add_180, [-1], True)
        sub_80: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(add_180, amax_7);  add_180 = amax_7 = None
        exp_7: "f32[32, 64, 64]" = torch.ops.aten.exp.default(sub_80);  sub_80 = None
        sum_8: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 64, 64]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_46: "f32[32, 64, 64]" = torch.ops.aten.expand.default(div_7, [32, 64, 64]);  div_7 = None
        expand_47: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_58, [32, 64, 128]);  permute_58 = None
        bmm_15: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(expand_46, expand_47);  expand_46 = expand_47 = None
        permute_64: "f32[32, 128, 64]" = torch.ops.aten.permute.default(bmm_15, [0, 2, 1]);  bmm_15 = None
        clone_56: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_192: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_56, [8, 512, 8, 8]);  clone_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_181: "f32[512]" = torch.ops.aten.add.Tensor(arg210_1, 1e-05);  arg210_1 = None
        sqrt_73: "f32[512]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_73: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_304: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_585: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_304, -1);  mul_304 = None
        unsqueeze_587: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_81: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_192, unsqueeze_585);  view_192 = unsqueeze_585 = None
        mul_305: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_587);  sub_81 = unsqueeze_587 = None
        unsqueeze_588: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_589: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_306: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_305, unsqueeze_589);  mul_305 = unsqueeze_589 = None
        unsqueeze_590: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_591: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_182: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_306, unsqueeze_591);  mul_306 = unsqueeze_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_77: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_182)
        mul_307: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_182, sigmoid_77);  add_182 = sigmoid_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_98: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_307, arg213_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_307 = arg213_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_183: "f32[1536]" = torch.ops.aten.add.Tensor(arg215_1, 1e-05);  arg215_1 = None
        sqrt_74: "f32[1536]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_74: "f32[1536]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_308: "f32[1536]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_593: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_595: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_82: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_593);  convolution_98 = unsqueeze_593 = None
        mul_309: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_595);  sub_82 = unsqueeze_595 = None
        unsqueeze_596: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_597: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_310: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_597);  mul_309 = unsqueeze_597 = None
        unsqueeze_598: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_599: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_184: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_599);  mul_310 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_185: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(add_184, mul_298);  add_184 = mul_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        sigmoid_78: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(add_185)
        mul_311: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_185, sigmoid_78);  add_185 = sigmoid_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_99: "f32[8, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_311, arg218_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_311 = arg218_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_186: "f32[1280]" = torch.ops.aten.add.Tensor(arg220_1, 1e-05);  arg220_1 = None
        sqrt_75: "f32[1280]" = torch.ops.aten.sqrt.default(add_186);  add_186 = None
        reciprocal_75: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_312: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_601: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
        unsqueeze_603: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_83: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_601);  convolution_99 = unsqueeze_601 = None
        mul_313: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_603);  sub_83 = unsqueeze_603 = None
        unsqueeze_604: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_605: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_314: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_605);  mul_313 = unsqueeze_605 = None
        unsqueeze_606: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_607: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_187: "f32[8, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_607);  mul_314 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_79: "f32[8, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_187)
        mul_315: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_187, sigmoid_79);  add_187 = sigmoid_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_13: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_315, [-1, -2], True);  mul_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_193: "f32[8, 1280]" = torch.ops.aten.view.default(mean_13, [8, 1280]);  mean_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_65: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg224_1, view_193, permute_65);  arg224_1 = view_193 = permute_65 = None
        return (addmm_1,)
        