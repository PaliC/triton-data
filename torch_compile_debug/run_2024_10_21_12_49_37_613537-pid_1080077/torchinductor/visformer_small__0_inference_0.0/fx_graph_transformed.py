class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 7, 7]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[192, 32, 4, 4]", arg7_1: "f32[192]", arg8_1: "f32[192]", arg9_1: "f32[192]", arg10_1: "f32[192]", arg11_1: "f32[192]", arg12_1: "f32[1, 192, 28, 28]", arg13_1: "f32[192]", arg14_1: "f32[192]", arg15_1: "f32[192]", arg16_1: "f32[192]", arg17_1: "f32[384, 192, 1, 1]", arg18_1: "f32[384, 48, 3, 3]", arg19_1: "f32[192, 384, 1, 1]", arg20_1: "f32[192]", arg21_1: "f32[192]", arg22_1: "f32[192]", arg23_1: "f32[192]", arg24_1: "f32[384, 192, 1, 1]", arg25_1: "f32[384, 48, 3, 3]", arg26_1: "f32[192, 384, 1, 1]", arg27_1: "f32[192]", arg28_1: "f32[192]", arg29_1: "f32[192]", arg30_1: "f32[192]", arg31_1: "f32[384, 192, 1, 1]", arg32_1: "f32[384, 48, 3, 3]", arg33_1: "f32[192, 384, 1, 1]", arg34_1: "f32[192]", arg35_1: "f32[192]", arg36_1: "f32[192]", arg37_1: "f32[192]", arg38_1: "f32[384, 192, 1, 1]", arg39_1: "f32[384, 48, 3, 3]", arg40_1: "f32[192, 384, 1, 1]", arg41_1: "f32[192]", arg42_1: "f32[192]", arg43_1: "f32[192]", arg44_1: "f32[192]", arg45_1: "f32[384, 192, 1, 1]", arg46_1: "f32[384, 48, 3, 3]", arg47_1: "f32[192, 384, 1, 1]", arg48_1: "f32[192]", arg49_1: "f32[192]", arg50_1: "f32[192]", arg51_1: "f32[192]", arg52_1: "f32[384, 192, 1, 1]", arg53_1: "f32[384, 48, 3, 3]", arg54_1: "f32[192, 384, 1, 1]", arg55_1: "f32[192]", arg56_1: "f32[192]", arg57_1: "f32[192]", arg58_1: "f32[192]", arg59_1: "f32[384, 192, 1, 1]", arg60_1: "f32[384, 48, 3, 3]", arg61_1: "f32[192, 384, 1, 1]", arg62_1: "f32[384, 192, 2, 2]", arg63_1: "f32[384]", arg64_1: "f32[384]", arg65_1: "f32[384]", arg66_1: "f32[384]", arg67_1: "f32[384]", arg68_1: "f32[1, 384, 14, 14]", arg69_1: "f32[384]", arg70_1: "f32[384]", arg71_1: "f32[384]", arg72_1: "f32[384]", arg73_1: "f32[1152, 384, 1, 1]", arg74_1: "f32[384, 384, 1, 1]", arg75_1: "f32[384]", arg76_1: "f32[384]", arg77_1: "f32[384]", arg78_1: "f32[384]", arg79_1: "f32[1536, 384, 1, 1]", arg80_1: "f32[384, 1536, 1, 1]", arg81_1: "f32[384]", arg82_1: "f32[384]", arg83_1: "f32[384]", arg84_1: "f32[384]", arg85_1: "f32[1152, 384, 1, 1]", arg86_1: "f32[384, 384, 1, 1]", arg87_1: "f32[384]", arg88_1: "f32[384]", arg89_1: "f32[384]", arg90_1: "f32[384]", arg91_1: "f32[1536, 384, 1, 1]", arg92_1: "f32[384, 1536, 1, 1]", arg93_1: "f32[384]", arg94_1: "f32[384]", arg95_1: "f32[384]", arg96_1: "f32[384]", arg97_1: "f32[1152, 384, 1, 1]", arg98_1: "f32[384, 384, 1, 1]", arg99_1: "f32[384]", arg100_1: "f32[384]", arg101_1: "f32[384]", arg102_1: "f32[384]", arg103_1: "f32[1536, 384, 1, 1]", arg104_1: "f32[384, 1536, 1, 1]", arg105_1: "f32[384]", arg106_1: "f32[384]", arg107_1: "f32[384]", arg108_1: "f32[384]", arg109_1: "f32[1152, 384, 1, 1]", arg110_1: "f32[384, 384, 1, 1]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[384]", arg114_1: "f32[384]", arg115_1: "f32[1536, 384, 1, 1]", arg116_1: "f32[384, 1536, 1, 1]", arg117_1: "f32[768, 384, 2, 2]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[1, 768, 7, 7]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[2304, 768, 1, 1]", arg129_1: "f32[768, 768, 1, 1]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[768]", arg134_1: "f32[3072, 768, 1, 1]", arg135_1: "f32[768, 3072, 1, 1]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[2304, 768, 1, 1]", arg141_1: "f32[768, 768, 1, 1]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[3072, 768, 1, 1]", arg147_1: "f32[768, 3072, 1, 1]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[2304, 768, 1, 1]", arg153_1: "f32[768, 768, 1, 1]", arg154_1: "f32[768]", arg155_1: "f32[768]", arg156_1: "f32[768]", arg157_1: "f32[768]", arg158_1: "f32[3072, 768, 1, 1]", arg159_1: "f32[768, 3072, 1, 1]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[2304, 768, 1, 1]", arg165_1: "f32[768, 768, 1, 1]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[768]", arg170_1: "f32[3072, 768, 1, 1]", arg171_1: "f32[768, 3072, 1, 1]", arg172_1: "f32[768]", arg173_1: "f32[768]", arg174_1: "f32[768]", arg175_1: "f32[768]", arg176_1: "f32[1000, 768]", arg177_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:396 in forward_features, code: x = self.stem(x)
        convolution_57: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        unsqueeze_224: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_225: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_36: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_225);  convolution_57 = unsqueeze_225 = None
        add_104: "f32[32]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_28: "f32[32]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_28: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_158: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        unsqueeze_227: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_159: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_227);  sub_36 = unsqueeze_227 = None
        unsqueeze_228: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_229: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_160: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_229);  mul_159 = unsqueeze_229 = None
        unsqueeze_230: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_231: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_105: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_231);  mul_160 = unsqueeze_231 = None
        relu_1: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_105);  add_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_58: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_1, arg6_1, arg7_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg6_1 = arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        unsqueeze_232: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_233: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_37: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_233);  convolution_58 = unsqueeze_233 = None
        add_106: "f32[192]" = torch.ops.aten.add.Tensor(arg9_1, 1e-05);  arg9_1 = None
        sqrt_29: "f32[192]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_29: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_161: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_161, -1);  mul_161 = None
        unsqueeze_235: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_162: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_235);  sub_37 = unsqueeze_235 = None
        unsqueeze_236: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_237: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_163: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_237);  mul_162 = unsqueeze_237 = None
        unsqueeze_238: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_239: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_107: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_239);  mul_163 = unsqueeze_239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:401 in forward_features, code: x = self.pos_drop(x + self.pos_embed1)
        add_108: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_107, arg12_1);  add_107 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_240: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_241: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_38: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_108, unsqueeze_241);  unsqueeze_241 = None
        add_109: "f32[192]" = torch.ops.aten.add.Tensor(arg14_1, 1e-05);  arg14_1 = None
        sqrt_30: "f32[192]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_30: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_164: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_164, -1);  mul_164 = None
        unsqueeze_243: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_165: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_243);  sub_38 = unsqueeze_243 = None
        unsqueeze_244: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_245: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_166: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_245);  mul_165 = unsqueeze_245 = None
        unsqueeze_246: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_247: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_110: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_247);  mul_166 = unsqueeze_247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_59: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_110, arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_110 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_167: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_59, 0.5)
        mul_168: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_59, 0.7071067811865476);  convolution_59 = None
        erf_22: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_111: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_169: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_167, add_111);  mul_167 = add_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:66 in forward, code: x = self.conv2(x)
        convolution_60: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_169, arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_169 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.act2(x)
        mul_170: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_60, 0.5)
        mul_171: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_60, 0.7071067811865476);  convolution_60 = None
        erf_23: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
        add_112: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_172: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_170, add_112);  mul_170 = add_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_61: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_172, arg19_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_172 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_113: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_108, convolution_61);  add_108 = convolution_61 = None
        unsqueeze_248: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_249: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_39: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_113, unsqueeze_249);  unsqueeze_249 = None
        add_114: "f32[192]" = torch.ops.aten.add.Tensor(arg21_1, 1e-05);  arg21_1 = None
        sqrt_31: "f32[192]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_31: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_173: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_173, -1);  mul_173 = None
        unsqueeze_251: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_174: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_251);  sub_39 = unsqueeze_251 = None
        unsqueeze_252: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_253: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_175: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_253);  mul_174 = unsqueeze_253 = None
        unsqueeze_254: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_255: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_115: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_255);  mul_175 = unsqueeze_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_62: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_115, arg24_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_115 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_176: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_62, 0.5)
        mul_177: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_62, 0.7071067811865476);  convolution_62 = None
        erf_24: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
        add_116: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_178: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_176, add_116);  mul_176 = add_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:66 in forward, code: x = self.conv2(x)
        convolution_63: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_178, arg25_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_178 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.act2(x)
        mul_179: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_63, 0.5)
        mul_180: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_63, 0.7071067811865476);  convolution_63 = None
        erf_25: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
        add_117: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_181: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_179, add_117);  mul_179 = add_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_64: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_181, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_181 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_118: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_113, convolution_64);  add_113 = convolution_64 = None
        unsqueeze_256: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_257: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_40: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_118, unsqueeze_257);  unsqueeze_257 = None
        add_119: "f32[192]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_32: "f32[192]" = torch.ops.aten.sqrt.default(add_119);  add_119 = None
        reciprocal_32: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_182: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
        unsqueeze_259: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_183: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_259);  sub_40 = unsqueeze_259 = None
        unsqueeze_260: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_261: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_184: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_261);  mul_183 = unsqueeze_261 = None
        unsqueeze_262: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_263: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_120: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_263);  mul_184 = unsqueeze_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_65: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_120, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_120 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_185: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_65, 0.5)
        mul_186: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_65, 0.7071067811865476);  convolution_65 = None
        erf_26: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
        add_121: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_187: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_185, add_121);  mul_185 = add_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:66 in forward, code: x = self.conv2(x)
        convolution_66: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_187, arg32_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_187 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.act2(x)
        mul_188: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_66, 0.5)
        mul_189: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_66, 0.7071067811865476);  convolution_66 = None
        erf_27: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_189);  mul_189 = None
        add_122: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_190: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_188, add_122);  mul_188 = add_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_67: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_190, arg33_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_190 = arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_123: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_118, convolution_67);  add_118 = convolution_67 = None
        unsqueeze_264: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_265: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_41: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_123, unsqueeze_265);  unsqueeze_265 = None
        add_124: "f32[192]" = torch.ops.aten.add.Tensor(arg35_1, 1e-05);  arg35_1 = None
        sqrt_33: "f32[192]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_33: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_191: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_191, -1);  mul_191 = None
        unsqueeze_267: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_192: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_267);  sub_41 = unsqueeze_267 = None
        unsqueeze_268: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_269: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_193: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_269);  mul_192 = unsqueeze_269 = None
        unsqueeze_270: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_271: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_125: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_271);  mul_193 = unsqueeze_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_68: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_125, arg38_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_125 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_194: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_68, 0.5)
        mul_195: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476);  convolution_68 = None
        erf_28: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
        add_126: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_196: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_194, add_126);  mul_194 = add_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:66 in forward, code: x = self.conv2(x)
        convolution_69: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_196, arg39_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_196 = arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.act2(x)
        mul_197: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_69, 0.5)
        mul_198: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_69, 0.7071067811865476);  convolution_69 = None
        erf_29: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_198);  mul_198 = None
        add_127: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_199: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_197, add_127);  mul_197 = add_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_70: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_199, arg40_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_199 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_128: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_123, convolution_70);  add_123 = convolution_70 = None
        unsqueeze_272: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_273: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_42: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_128, unsqueeze_273);  unsqueeze_273 = None
        add_129: "f32[192]" = torch.ops.aten.add.Tensor(arg42_1, 1e-05);  arg42_1 = None
        sqrt_34: "f32[192]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_34: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_200: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_200, -1);  mul_200 = None
        unsqueeze_275: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_201: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_275);  sub_42 = unsqueeze_275 = None
        unsqueeze_276: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_277: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_202: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_201, unsqueeze_277);  mul_201 = unsqueeze_277 = None
        unsqueeze_278: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_279: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_130: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_279);  mul_202 = unsqueeze_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_71: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_130, arg45_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_130 = arg45_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_203: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_71, 0.5)
        mul_204: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_71, 0.7071067811865476);  convolution_71 = None
        erf_30: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_204);  mul_204 = None
        add_131: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_205: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_203, add_131);  mul_203 = add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:66 in forward, code: x = self.conv2(x)
        convolution_72: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_205, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_205 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.act2(x)
        mul_206: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_72, 0.5)
        mul_207: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_72, 0.7071067811865476);  convolution_72 = None
        erf_31: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_132: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_208: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_206, add_132);  mul_206 = add_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_73: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_208, arg47_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_208 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_133: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_128, convolution_73);  add_128 = convolution_73 = None
        unsqueeze_280: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_281: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_43: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_133, unsqueeze_281);  unsqueeze_281 = None
        add_134: "f32[192]" = torch.ops.aten.add.Tensor(arg49_1, 1e-05);  arg49_1 = None
        sqrt_35: "f32[192]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_35: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_209: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_209, -1);  mul_209 = None
        unsqueeze_283: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_210: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_283);  sub_43 = unsqueeze_283 = None
        unsqueeze_284: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_285: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_211: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_285);  mul_210 = unsqueeze_285 = None
        unsqueeze_286: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_287: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_135: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_287);  mul_211 = unsqueeze_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_74: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_135, arg52_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_135 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_212: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_74, 0.5)
        mul_213: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476);  convolution_74 = None
        erf_32: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_213);  mul_213 = None
        add_136: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_214: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_212, add_136);  mul_212 = add_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:66 in forward, code: x = self.conv2(x)
        convolution_75: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_214, arg53_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_214 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.act2(x)
        mul_215: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_75, 0.5)
        mul_216: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_75, 0.7071067811865476);  convolution_75 = None
        erf_33: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_216);  mul_216 = None
        add_137: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_217: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_215, add_137);  mul_215 = add_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_76: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_217, arg54_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_217 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_138: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_133, convolution_76);  add_133 = convolution_76 = None
        unsqueeze_288: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_289: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_44: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_138, unsqueeze_289);  unsqueeze_289 = None
        add_139: "f32[192]" = torch.ops.aten.add.Tensor(arg56_1, 1e-05);  arg56_1 = None
        sqrt_36: "f32[192]" = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_36: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_218: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_218, -1);  mul_218 = None
        unsqueeze_291: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_219: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_291);  sub_44 = unsqueeze_291 = None
        unsqueeze_292: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_293: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_220: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_219, unsqueeze_293);  mul_219 = unsqueeze_293 = None
        unsqueeze_294: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_295: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_140: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_220, unsqueeze_295);  mul_220 = unsqueeze_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_77: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_140, arg59_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_140 = arg59_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_221: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_77, 0.5)
        mul_222: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_77, 0.7071067811865476);  convolution_77 = None
        erf_34: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_222);  mul_222 = None
        add_141: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_223: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_221, add_141);  mul_221 = add_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:66 in forward, code: x = self.conv2(x)
        convolution_78: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_223, arg60_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_223 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.act2(x)
        mul_224: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_78, 0.5)
        mul_225: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_78, 0.7071067811865476);  convolution_78 = None
        erf_35: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_225);  mul_225 = None
        add_142: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_226: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_224, add_142);  mul_224 = add_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_79: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_226, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_226 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_143: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_138, convolution_79);  add_138 = convolution_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_80: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(add_143, arg62_1, arg63_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  add_143 = arg62_1 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        unsqueeze_296: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_297: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_45: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_297);  convolution_80 = unsqueeze_297 = None
        add_144: "f32[384]" = torch.ops.aten.add.Tensor(arg65_1, 1e-05);  arg65_1 = None
        sqrt_37: "f32[384]" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_37: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_227: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_227, -1);  mul_227 = None
        unsqueeze_299: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_228: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_299);  sub_45 = unsqueeze_299 = None
        unsqueeze_300: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_301: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_229: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_228, unsqueeze_301);  mul_228 = unsqueeze_301 = None
        unsqueeze_302: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_303: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_145: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_229, unsqueeze_303);  mul_229 = unsqueeze_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:411 in forward_features, code: x = self.pos_drop(x + self.pos_embed2)
        add_146: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_145, arg68_1);  add_145 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_304: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_305: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_46: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_146, unsqueeze_305);  unsqueeze_305 = None
        add_147: "f32[384]" = torch.ops.aten.add.Tensor(arg70_1, 1e-05);  arg70_1 = None
        sqrt_38: "f32[384]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_38: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_230: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_230, -1);  mul_230 = None
        unsqueeze_307: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_231: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_307);  sub_46 = unsqueeze_307 = None
        unsqueeze_308: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_309: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_232: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_309);  mul_231 = unsqueeze_309 = None
        unsqueeze_310: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_311: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_148: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_311);  mul_232 = unsqueeze_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:92 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convolution_81: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_148, arg73_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_148 = arg73_1 = None
        view_65: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.reshape.default(convolution_81, [8, 3, 6, 64, -1]);  convolution_81 = None
        permute_25: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_65, [1, 0, 2, 4, 3]);  view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:93 in forward, code: q, k, v = x.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_24: "f32[8, 6, 196, 64]" = unbind_8[0]
        getitem_25: "f32[8, 6, 196, 64]" = unbind_8[1]
        getitem_26: "f32[8, 6, 196, 64]" = unbind_8[2];  unbind_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:101 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_32: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_24, [8, 6, 196, 64]);  getitem_24 = None
        clone_98: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_66: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_98, [48, 196, 64]);  clone_98 = None
        permute_26: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_25, [0, 1, 3, 2]);  getitem_25 = None
        expand_33: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_26, [8, 6, 64, 196]);  permute_26 = None
        clone_99: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_67: "f32[48, 64, 196]" = torch.ops.aten.reshape.default(clone_99, [48, 64, 196]);  clone_99 = None
        bmm_16: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_66, view_67);  view_66 = view_67 = None
        view_68: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_16, [8, 6, 196, 196]);  bmm_16 = None
        
        # No stacktrace found for following nodes
        mul_tensor_14: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_68, 1);  view_68 = None
        amax_default_7: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_tensor_14, [-1], True)
        sub_tensor_7: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = amax_default_7 = None
        mul_tensor_15: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_tensor_7, 0.125);  sub_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:102 in forward, code: attn = attn.softmax(dim=-1)
        exp_8: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(mul_tensor_15);  mul_tensor_15 = None
        sum_9: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:104 in forward, code: x = attn @ v
        expand_34: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(div_8, [8, 6, 196, 196]);  div_8 = None
        view_69: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(expand_34, [48, 196, 196]);  expand_34 = None
        expand_35: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_26, [8, 6, 196, 64]);  getitem_26 = None
        clone_101: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_70: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_101, [48, 196, 64]);  clone_101 = None
        bmm_17: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_69, view_70);  view_69 = view_70 = None
        view_71: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_17, [8, 6, 196, 64]);  bmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:106 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_27: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_71, [0, 1, 3, 2]);  view_71 = None
        clone_102: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_72: "f32[8, 384, 14, 14]" = torch.ops.aten.reshape.default(clone_102, [8, 384, 14, 14]);  clone_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.proj(x)
        convolution_82: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_72, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_72 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_149: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_146, convolution_82);  add_146 = convolution_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_312: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_313: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_48: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_149, unsqueeze_313);  unsqueeze_313 = None
        add_150: "f32[384]" = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_39: "f32[384]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_39: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_234: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
        unsqueeze_315: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_235: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_315);  sub_48 = unsqueeze_315 = None
        unsqueeze_316: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_317: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_236: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_317);  mul_235 = unsqueeze_317 = None
        unsqueeze_318: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_319: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_151: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_319);  mul_236 = unsqueeze_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_83: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_151, arg79_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_151 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_237: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_83, 0.5)
        mul_238: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_83, 0.7071067811865476);  convolution_83 = None
        erf_36: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_152: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_239: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_237, add_152);  mul_237 = add_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_84: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_239, arg80_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_239 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_153: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_149, convolution_84);  add_149 = convolution_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_320: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_321: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_49: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_153, unsqueeze_321);  unsqueeze_321 = None
        add_154: "f32[384]" = torch.ops.aten.add.Tensor(arg82_1, 1e-05);  arg82_1 = None
        sqrt_40: "f32[384]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_40: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_240: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_323: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_241: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_323);  sub_49 = unsqueeze_323 = None
        unsqueeze_324: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_325: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_242: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_325);  mul_241 = unsqueeze_325 = None
        unsqueeze_326: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_327: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_155: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_327);  mul_242 = unsqueeze_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:92 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convolution_85: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_155, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_155 = arg85_1 = None
        view_73: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.reshape.default(convolution_85, [8, 3, 6, 64, -1]);  convolution_85 = None
        permute_28: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_73, [1, 0, 2, 4, 3]);  view_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:93 in forward, code: q, k, v = x.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_28);  permute_28 = None
        getitem_27: "f32[8, 6, 196, 64]" = unbind_9[0]
        getitem_28: "f32[8, 6, 196, 64]" = unbind_9[1]
        getitem_29: "f32[8, 6, 196, 64]" = unbind_9[2];  unbind_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:101 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_36: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_27, [8, 6, 196, 64]);  getitem_27 = None
        clone_106: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_74: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_106, [48, 196, 64]);  clone_106 = None
        permute_29: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_37: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_29, [8, 6, 64, 196]);  permute_29 = None
        clone_107: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_75: "f32[48, 64, 196]" = torch.ops.aten.reshape.default(clone_107, [48, 64, 196]);  clone_107 = None
        bmm_18: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_74, view_75);  view_74 = view_75 = None
        view_76: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_18, [8, 6, 196, 196]);  bmm_18 = None
        
        # No stacktrace found for following nodes
        mul_tensor_12: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_76, 1);  view_76 = None
        amax_default_6: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_tensor_12, [-1], True)
        sub_tensor_6: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = amax_default_6 = None
        mul_tensor_13: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.125);  sub_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:102 in forward, code: attn = attn.softmax(dim=-1)
        exp_9: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(mul_tensor_13);  mul_tensor_13 = None
        sum_10: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:104 in forward, code: x = attn @ v
        expand_38: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(div_9, [8, 6, 196, 196]);  div_9 = None
        view_77: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(expand_38, [48, 196, 196]);  expand_38 = None
        expand_39: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_29, [8, 6, 196, 64]);  getitem_29 = None
        clone_109: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_78: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_109, [48, 196, 64]);  clone_109 = None
        bmm_19: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
        view_79: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_19, [8, 6, 196, 64]);  bmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:106 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_30: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_79, [0, 1, 3, 2]);  view_79 = None
        clone_110: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_80: "f32[8, 384, 14, 14]" = torch.ops.aten.reshape.default(clone_110, [8, 384, 14, 14]);  clone_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.proj(x)
        convolution_86: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_80, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_80 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_156: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_153, convolution_86);  add_153 = convolution_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_328: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_329: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_51: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_156, unsqueeze_329);  unsqueeze_329 = None
        add_157: "f32[384]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_41: "f32[384]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_41: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_244: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_244, -1);  mul_244 = None
        unsqueeze_331: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_245: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_331);  sub_51 = unsqueeze_331 = None
        unsqueeze_332: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_333: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_246: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_333);  mul_245 = unsqueeze_333 = None
        unsqueeze_334: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_335: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_158: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_335);  mul_246 = unsqueeze_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_87: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_158, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_158 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_247: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_87, 0.5)
        mul_248: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_87, 0.7071067811865476);  convolution_87 = None
        erf_37: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_248);  mul_248 = None
        add_159: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_249: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_247, add_159);  mul_247 = add_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_88: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_249, arg92_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_249 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_160: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_156, convolution_88);  add_156 = convolution_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_336: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_337: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_52: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_160, unsqueeze_337);  unsqueeze_337 = None
        add_161: "f32[384]" = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_42: "f32[384]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_42: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_250: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_339: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_251: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_339);  sub_52 = unsqueeze_339 = None
        unsqueeze_340: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_341: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_252: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_341);  mul_251 = unsqueeze_341 = None
        unsqueeze_342: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_343: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_162: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_343);  mul_252 = unsqueeze_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:92 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convolution_89: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_162, arg97_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_162 = arg97_1 = None
        view_81: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.reshape.default(convolution_89, [8, 3, 6, 64, -1]);  convolution_89 = None
        permute_31: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_81, [1, 0, 2, 4, 3]);  view_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:93 in forward, code: q, k, v = x.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
        getitem_30: "f32[8, 6, 196, 64]" = unbind_10[0]
        getitem_31: "f32[8, 6, 196, 64]" = unbind_10[1]
        getitem_32: "f32[8, 6, 196, 64]" = unbind_10[2];  unbind_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:101 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_40: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_30, [8, 6, 196, 64]);  getitem_30 = None
        clone_114: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_82: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_114, [48, 196, 64]);  clone_114 = None
        permute_32: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
        expand_41: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_32, [8, 6, 64, 196]);  permute_32 = None
        clone_115: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_83: "f32[48, 64, 196]" = torch.ops.aten.reshape.default(clone_115, [48, 64, 196]);  clone_115 = None
        bmm_20: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_20, [8, 6, 196, 196]);  bmm_20 = None
        
        # No stacktrace found for following nodes
        mul_tensor_10: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_84, 1);  view_84 = None
        amax_default_5: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_tensor_10, [-1], True)
        sub_tensor_5: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_tensor_5, 0.125);  sub_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:102 in forward, code: attn = attn.softmax(dim=-1)
        exp_10: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        sum_11: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:104 in forward, code: x = attn @ v
        expand_42: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(div_10, [8, 6, 196, 196]);  div_10 = None
        view_85: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(expand_42, [48, 196, 196]);  expand_42 = None
        expand_43: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_32, [8, 6, 196, 64]);  getitem_32 = None
        clone_117: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_86: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_117, [48, 196, 64]);  clone_117 = None
        bmm_21: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_85, view_86);  view_85 = view_86 = None
        view_87: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_21, [8, 6, 196, 64]);  bmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:106 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_33: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_87, [0, 1, 3, 2]);  view_87 = None
        clone_118: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_88: "f32[8, 384, 14, 14]" = torch.ops.aten.reshape.default(clone_118, [8, 384, 14, 14]);  clone_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.proj(x)
        convolution_90: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_88, arg98_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_88 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_163: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_160, convolution_90);  add_160 = convolution_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_344: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_345: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_54: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_163, unsqueeze_345);  unsqueeze_345 = None
        add_164: "f32[384]" = torch.ops.aten.add.Tensor(arg100_1, 1e-05);  arg100_1 = None
        sqrt_43: "f32[384]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_43: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_254: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_254, -1);  mul_254 = None
        unsqueeze_347: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_255: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_347);  sub_54 = unsqueeze_347 = None
        unsqueeze_348: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_349: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_256: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_255, unsqueeze_349);  mul_255 = unsqueeze_349 = None
        unsqueeze_350: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_351: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_165: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_256, unsqueeze_351);  mul_256 = unsqueeze_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_91: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_165, arg103_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_165 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_257: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_91, 0.5)
        mul_258: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_91, 0.7071067811865476);  convolution_91 = None
        erf_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_258);  mul_258 = None
        add_166: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_259: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_257, add_166);  mul_257 = add_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_92: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_259, arg104_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_259 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_167: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_163, convolution_92);  add_163 = convolution_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_352: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_353: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_55: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_167, unsqueeze_353);  unsqueeze_353 = None
        add_168: "f32[384]" = torch.ops.aten.add.Tensor(arg106_1, 1e-05);  arg106_1 = None
        sqrt_44: "f32[384]" = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_44: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_260: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_260, -1);  mul_260 = None
        unsqueeze_355: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_261: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_355);  sub_55 = unsqueeze_355 = None
        unsqueeze_356: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_357: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_262: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_357);  mul_261 = unsqueeze_357 = None
        unsqueeze_358: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_359: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_169: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_359);  mul_262 = unsqueeze_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:92 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convolution_93: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_169, arg109_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_169 = arg109_1 = None
        view_89: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.reshape.default(convolution_93, [8, 3, 6, 64, -1]);  convolution_93 = None
        permute_34: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_89, [1, 0, 2, 4, 3]);  view_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:93 in forward, code: q, k, v = x.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_34);  permute_34 = None
        getitem_33: "f32[8, 6, 196, 64]" = unbind_11[0]
        getitem_34: "f32[8, 6, 196, 64]" = unbind_11[1]
        getitem_35: "f32[8, 6, 196, 64]" = unbind_11[2];  unbind_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:101 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_44: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_33, [8, 6, 196, 64]);  getitem_33 = None
        clone_122: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_90: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_122, [48, 196, 64]);  clone_122 = None
        permute_35: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_34, [0, 1, 3, 2]);  getitem_34 = None
        expand_45: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_35, [8, 6, 64, 196]);  permute_35 = None
        clone_123: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_91: "f32[48, 64, 196]" = torch.ops.aten.reshape.default(clone_123, [48, 64, 196]);  clone_123 = None
        bmm_22: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_90, view_91);  view_90 = view_91 = None
        view_92: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_22, [8, 6, 196, 196]);  bmm_22 = None
        
        # No stacktrace found for following nodes
        mul_tensor_8: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_92, 1);  view_92 = None
        amax_default_4: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_tensor_8, [-1], True)
        sub_tensor_4: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.125);  sub_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:102 in forward, code: attn = attn.softmax(dim=-1)
        exp_11: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        sum_12: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:104 in forward, code: x = attn @ v
        expand_46: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(div_11, [8, 6, 196, 196]);  div_11 = None
        view_93: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(expand_46, [48, 196, 196]);  expand_46 = None
        expand_47: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_35, [8, 6, 196, 64]);  getitem_35 = None
        clone_125: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_94: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_125, [48, 196, 64]);  clone_125 = None
        bmm_23: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
        view_95: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_23, [8, 6, 196, 64]);  bmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:106 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_36: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_95, [0, 1, 3, 2]);  view_95 = None
        clone_126: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_96: "f32[8, 384, 14, 14]" = torch.ops.aten.reshape.default(clone_126, [8, 384, 14, 14]);  clone_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.proj(x)
        convolution_94: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_96, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_96 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_170: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_167, convolution_94);  add_167 = convolution_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_360: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_361: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_57: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_170, unsqueeze_361);  unsqueeze_361 = None
        add_171: "f32[384]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_45: "f32[384]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_45: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_264: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_363: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_265: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_363);  sub_57 = unsqueeze_363 = None
        unsqueeze_364: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_365: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_266: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_365);  mul_265 = unsqueeze_365 = None
        unsqueeze_366: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_367: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_172: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_367);  mul_266 = unsqueeze_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_95: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_172, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_172 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_267: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_95, 0.5)
        mul_268: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_95, 0.7071067811865476);  convolution_95 = None
        erf_39: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_268);  mul_268 = None
        add_173: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_269: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_267, add_173);  mul_267 = add_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_96: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_269, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_269 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_174: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_170, convolution_96);  add_170 = convolution_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_97: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(add_174, arg117_1, arg118_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  add_174 = arg117_1 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        unsqueeze_368: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_369: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        sub_58: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_369);  convolution_97 = unsqueeze_369 = None
        add_175: "f32[768]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_46: "f32[768]" = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_46: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_270: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_370: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_371: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        mul_271: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_371);  sub_58 = unsqueeze_371 = None
        unsqueeze_372: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_373: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_272: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_373);  mul_271 = unsqueeze_373 = None
        unsqueeze_374: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_375: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_176: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_375);  mul_272 = unsqueeze_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:421 in forward_features, code: x = self.pos_drop(x + self.pos_embed3)
        add_177: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_176, arg123_1);  add_176 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_376: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_377: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        sub_59: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_177, unsqueeze_377);  unsqueeze_377 = None
        add_178: "f32[768]" = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_47: "f32[768]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_47: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_273: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_378: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_379: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        mul_274: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_379);  sub_59 = unsqueeze_379 = None
        unsqueeze_380: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_381: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_275: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_381);  mul_274 = unsqueeze_381 = None
        unsqueeze_382: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_383: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_179: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_383);  mul_275 = unsqueeze_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:92 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convolution_98: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_179, arg128_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_179 = arg128_1 = None
        view_97: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_98, [8, 3, 6, 128, -1]);  convolution_98 = None
        permute_37: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_97, [1, 0, 2, 4, 3]);  view_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:93 in forward, code: q, k, v = x.unbind(0)
        unbind_12 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
        getitem_36: "f32[8, 6, 49, 128]" = unbind_12[0]
        getitem_37: "f32[8, 6, 49, 128]" = unbind_12[1]
        getitem_38: "f32[8, 6, 49, 128]" = unbind_12[2];  unbind_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:101 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_48: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_36, [8, 6, 49, 128]);  getitem_36 = None
        clone_131: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_98: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_131, [48, 49, 128]);  clone_131 = None
        permute_38: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
        expand_49: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_38, [8, 6, 128, 49]);  permute_38 = None
        clone_132: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_99: "f32[48, 128, 49]" = torch.ops.aten.reshape.default(clone_132, [48, 128, 49]);  clone_132 = None
        bmm_24: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_98, view_99);  view_98 = view_99 = None
        view_100: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_24, [8, 6, 49, 49]);  bmm_24 = None
        
        # No stacktrace found for following nodes
        mul_tensor_6: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_100, 1);  view_100 = None
        amax_default_3: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_tensor_6, [-1], True)
        sub_tensor_3: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_tensor_3, 0.08838834764831845);  sub_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:102 in forward, code: attn = attn.softmax(dim=-1)
        exp_12: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        sum_13: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:104 in forward, code: x = attn @ v
        expand_50: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(div_12, [8, 6, 49, 49]);  div_12 = None
        view_101: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(expand_50, [48, 49, 49]);  expand_50 = None
        expand_51: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_38, [8, 6, 49, 128]);  getitem_38 = None
        clone_134: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_102: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_134, [48, 49, 128]);  clone_134 = None
        bmm_25: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_101, view_102);  view_101 = view_102 = None
        view_103: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_25, [8, 6, 49, 128]);  bmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:106 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_39: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2]);  view_103 = None
        clone_135: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_104: "f32[8, 768, 7, 7]" = torch.ops.aten.reshape.default(clone_135, [8, 768, 7, 7]);  clone_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.proj(x)
        convolution_99: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_104, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_104 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_180: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_177, convolution_99);  add_177 = convolution_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_384: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_385: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        sub_61: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_180, unsqueeze_385);  unsqueeze_385 = None
        add_181: "f32[768]" = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_48: "f32[768]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_48: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_277: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_386: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_387: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        mul_278: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_387);  sub_61 = unsqueeze_387 = None
        unsqueeze_388: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_389: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_279: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_389);  mul_278 = unsqueeze_389 = None
        unsqueeze_390: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_391: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_182: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_391);  mul_279 = unsqueeze_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_100: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_182, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_182 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_280: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_100, 0.5)
        mul_281: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_100, 0.7071067811865476);  convolution_100 = None
        erf_40: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_281);  mul_281 = None
        add_183: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_282: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_280, add_183);  mul_280 = add_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_101: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_282, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_282 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_184: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_180, convolution_101);  add_180 = convolution_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_392: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_393: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        sub_62: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_184, unsqueeze_393);  unsqueeze_393 = None
        add_185: "f32[768]" = torch.ops.aten.add.Tensor(arg137_1, 1e-05);  arg137_1 = None
        sqrt_49: "f32[768]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_49: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_283: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_394: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_395: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        mul_284: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_395);  sub_62 = unsqueeze_395 = None
        unsqueeze_396: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_397: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_285: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_397);  mul_284 = unsqueeze_397 = None
        unsqueeze_398: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_399: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_186: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_285, unsqueeze_399);  mul_285 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:92 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convolution_102: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_186, arg140_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_186 = arg140_1 = None
        view_105: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_102, [8, 3, 6, 128, -1]);  convolution_102 = None
        permute_40: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_105, [1, 0, 2, 4, 3]);  view_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:93 in forward, code: q, k, v = x.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_40);  permute_40 = None
        getitem_39: "f32[8, 6, 49, 128]" = unbind_13[0]
        getitem_40: "f32[8, 6, 49, 128]" = unbind_13[1]
        getitem_41: "f32[8, 6, 49, 128]" = unbind_13[2];  unbind_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:101 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_52: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_39, [8, 6, 49, 128]);  getitem_39 = None
        clone_139: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_106: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_139, [48, 49, 128]);  clone_139 = None
        permute_41: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_40, [0, 1, 3, 2]);  getitem_40 = None
        expand_53: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_41, [8, 6, 128, 49]);  permute_41 = None
        clone_140: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_107: "f32[48, 128, 49]" = torch.ops.aten.reshape.default(clone_140, [48, 128, 49]);  clone_140 = None
        bmm_26: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
        view_108: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_26, [8, 6, 49, 49]);  bmm_26 = None
        
        # No stacktrace found for following nodes
        mul_tensor_4: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_108, 1);  view_108 = None
        amax_default_2: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_tensor_4, [-1], True)
        sub_tensor_2: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.08838834764831845);  sub_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:102 in forward, code: attn = attn.softmax(dim=-1)
        exp_13: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        sum_14: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:104 in forward, code: x = attn @ v
        expand_54: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(div_13, [8, 6, 49, 49]);  div_13 = None
        view_109: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(expand_54, [48, 49, 49]);  expand_54 = None
        expand_55: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_41, [8, 6, 49, 128]);  getitem_41 = None
        clone_142: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_110: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_142, [48, 49, 128]);  clone_142 = None
        bmm_27: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_109, view_110);  view_109 = view_110 = None
        view_111: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_27, [8, 6, 49, 128]);  bmm_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:106 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_42: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_111, [0, 1, 3, 2]);  view_111 = None
        clone_143: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_112: "f32[8, 768, 7, 7]" = torch.ops.aten.reshape.default(clone_143, [8, 768, 7, 7]);  clone_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.proj(x)
        convolution_103: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_112, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_112 = arg141_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_187: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_184, convolution_103);  add_184 = convolution_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_400: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_401: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        sub_64: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_187, unsqueeze_401);  unsqueeze_401 = None
        add_188: "f32[768]" = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_50: "f32[768]" = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_50: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_287: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_402: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_287, -1);  mul_287 = None
        unsqueeze_403: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        mul_288: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_403);  sub_64 = unsqueeze_403 = None
        unsqueeze_404: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_405: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_289: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_405);  mul_288 = unsqueeze_405 = None
        unsqueeze_406: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_407: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_189: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_407);  mul_289 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_104: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_189, arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_189 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_290: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_104, 0.5)
        mul_291: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_104, 0.7071067811865476);  convolution_104 = None
        erf_41: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_291);  mul_291 = None
        add_190: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_292: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_290, add_190);  mul_290 = add_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_105: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_292, arg147_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_292 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_191: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_187, convolution_105);  add_187 = convolution_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_408: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_409: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        sub_65: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_191, unsqueeze_409);  unsqueeze_409 = None
        add_192: "f32[768]" = torch.ops.aten.add.Tensor(arg149_1, 1e-05);  arg149_1 = None
        sqrt_51: "f32[768]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_51: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_293: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_410: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_293, -1);  mul_293 = None
        unsqueeze_411: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        mul_294: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_411);  sub_65 = unsqueeze_411 = None
        unsqueeze_412: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_413: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_295: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_413);  mul_294 = unsqueeze_413 = None
        unsqueeze_414: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_415: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_193: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_295, unsqueeze_415);  mul_295 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:92 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convolution_106: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_193, arg152_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_193 = arg152_1 = None
        view_113: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_106, [8, 3, 6, 128, -1]);  convolution_106 = None
        permute_43: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_113, [1, 0, 2, 4, 3]);  view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:93 in forward, code: q, k, v = x.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_43);  permute_43 = None
        getitem_42: "f32[8, 6, 49, 128]" = unbind_14[0]
        getitem_43: "f32[8, 6, 49, 128]" = unbind_14[1]
        getitem_44: "f32[8, 6, 49, 128]" = unbind_14[2];  unbind_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:101 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_56: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_42, [8, 6, 49, 128]);  getitem_42 = None
        clone_147: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_114: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_147, [48, 49, 128]);  clone_147 = None
        permute_44: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_43, [0, 1, 3, 2]);  getitem_43 = None
        expand_57: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_44, [8, 6, 128, 49]);  permute_44 = None
        clone_148: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_115: "f32[48, 128, 49]" = torch.ops.aten.reshape.default(clone_148, [48, 128, 49]);  clone_148 = None
        bmm_28: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
        view_116: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_28, [8, 6, 49, 49]);  bmm_28 = None
        
        # No stacktrace found for following nodes
        mul_tensor_2: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_116, 1);  view_116 = None
        amax_default_1: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_tensor_2, [-1], True)
        sub_tensor_1: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_tensor_1, 0.08838834764831845);  sub_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:102 in forward, code: attn = attn.softmax(dim=-1)
        exp_14: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        sum_15: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:104 in forward, code: x = attn @ v
        expand_58: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(div_14, [8, 6, 49, 49]);  div_14 = None
        view_117: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(expand_58, [48, 49, 49]);  expand_58 = None
        expand_59: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_44, [8, 6, 49, 128]);  getitem_44 = None
        clone_150: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_118: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_150, [48, 49, 128]);  clone_150 = None
        bmm_29: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_117, view_118);  view_117 = view_118 = None
        view_119: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_29, [8, 6, 49, 128]);  bmm_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:106 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_45: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_119, [0, 1, 3, 2]);  view_119 = None
        clone_151: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_120: "f32[8, 768, 7, 7]" = torch.ops.aten.reshape.default(clone_151, [8, 768, 7, 7]);  clone_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.proj(x)
        convolution_107: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_120, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_120 = arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_194: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_191, convolution_107);  add_191 = convolution_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_416: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_417: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        sub_67: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_194, unsqueeze_417);  unsqueeze_417 = None
        add_195: "f32[768]" = torch.ops.aten.add.Tensor(arg155_1, 1e-05);  arg155_1 = None
        sqrt_52: "f32[768]" = torch.ops.aten.sqrt.default(add_195);  add_195 = None
        reciprocal_52: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_297: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_418: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
        unsqueeze_419: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        mul_298: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_419);  sub_67 = unsqueeze_419 = None
        unsqueeze_420: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_421: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_299: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_421);  mul_298 = unsqueeze_421 = None
        unsqueeze_422: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_423: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_196: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_423);  mul_299 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_108: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_196, arg158_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_196 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_300: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_108, 0.5)
        mul_301: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_108, 0.7071067811865476);  convolution_108 = None
        erf_42: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_301);  mul_301 = None
        add_197: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_302: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_300, add_197);  mul_300 = add_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_109: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_302, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_302 = arg159_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_198: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_194, convolution_109);  add_194 = convolution_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_424: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_425: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        sub_68: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_198, unsqueeze_425);  unsqueeze_425 = None
        add_199: "f32[768]" = torch.ops.aten.add.Tensor(arg161_1, 1e-05);  arg161_1 = None
        sqrt_53: "f32[768]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_53: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_303: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_426: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_427: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        mul_304: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_427);  sub_68 = unsqueeze_427 = None
        unsqueeze_428: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_429: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_305: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_429);  mul_304 = unsqueeze_429 = None
        unsqueeze_430: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_431: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_200: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_431);  mul_305 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:92 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convolution_110: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_200, arg164_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_200 = arg164_1 = None
        view_121: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_110, [8, 3, 6, 128, -1]);  convolution_110 = None
        permute_46: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_121, [1, 0, 2, 4, 3]);  view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:93 in forward, code: q, k, v = x.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_46);  permute_46 = None
        getitem_45: "f32[8, 6, 49, 128]" = unbind_15[0]
        getitem_46: "f32[8, 6, 49, 128]" = unbind_15[1]
        getitem_47: "f32[8, 6, 49, 128]" = unbind_15[2];  unbind_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:101 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_60: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_45, [8, 6, 49, 128]);  getitem_45 = None
        clone_155: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_122: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_155, [48, 49, 128]);  clone_155 = None
        permute_47: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_46, [0, 1, 3, 2]);  getitem_46 = None
        expand_61: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_47, [8, 6, 128, 49]);  permute_47 = None
        clone_156: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_123: "f32[48, 128, 49]" = torch.ops.aten.reshape.default(clone_156, [48, 128, 49]);  clone_156 = None
        bmm_30: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_30, [8, 6, 49, 49]);  bmm_30 = None
        
        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_124, 1);  view_124 = None
        amax_default: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.08838834764831845);  sub_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:102 in forward, code: attn = attn.softmax(dim=-1)
        exp_15: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_16: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:104 in forward, code: x = attn @ v
        expand_62: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(div_15, [8, 6, 49, 49]);  div_15 = None
        view_125: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(expand_62, [48, 49, 49]);  expand_62 = None
        expand_63: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_47, [8, 6, 49, 128]);  getitem_47 = None
        clone_158: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_126: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_158, [48, 49, 128]);  clone_158 = None
        bmm_31: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_125, view_126);  view_125 = view_126 = None
        view_127: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_31, [8, 6, 49, 128]);  bmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:106 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_48: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_127, [0, 1, 3, 2]);  view_127 = None
        clone_159: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_128: "f32[8, 768, 7, 7]" = torch.ops.aten.reshape.default(clone_159, [8, 768, 7, 7]);  clone_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.proj(x)
        convolution_111: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_128, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_128 = arg165_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:156 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_201: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_198, convolution_111);  add_198 = convolution_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_432: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_433: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        sub_70: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_201, unsqueeze_433);  unsqueeze_433 = None
        add_202: "f32[768]" = torch.ops.aten.add.Tensor(arg167_1, 1e-05);  arg167_1 = None
        sqrt_54: "f32[768]" = torch.ops.aten.sqrt.default(add_202);  add_202 = None
        reciprocal_54: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_307: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_434: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_307, -1);  mul_307 = None
        unsqueeze_435: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        mul_308: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_435);  sub_70 = unsqueeze_435 = None
        unsqueeze_436: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_437: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_309: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_437);  mul_308 = unsqueeze_437 = None
        unsqueeze_438: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_439: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_203: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_309, unsqueeze_439);  mul_309 = unsqueeze_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:62 in forward, code: x = self.conv1(x)
        convolution_112: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_203, arg170_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_203 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:63 in forward, code: x = self.act1(x)
        mul_310: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_112, 0.5)
        mul_311: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_112, 0.7071067811865476);  convolution_112 = None
        erf_43: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_311);  mul_311 = None
        add_204: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_312: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_310, add_204);  mul_310 = add_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.conv3(x)
        convolution_113: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_312, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_312 = arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:157 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_205: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_201, convolution_113);  add_201 = convolution_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:427 in forward_features, code: x = self.norm(x)
        unsqueeze_440: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_441: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        sub_71: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_205, unsqueeze_441);  add_205 = unsqueeze_441 = None
        add_206: "f32[768]" = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_55: "f32[768]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_55: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_313: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_442: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_313, -1);  mul_313 = None
        unsqueeze_443: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        mul_314: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_443);  sub_71 = unsqueeze_443 = None
        unsqueeze_444: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_445: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_315: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_445);  mul_314 = unsqueeze_445 = None
        unsqueeze_446: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_447: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_207: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_315, unsqueeze_447);  mul_315 = unsqueeze_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_207, [-1, -2], True);  add_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_129: "f32[8, 768]" = torch.ops.aten.reshape.default(mean_1, [8, 768]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/visformer.py:433 in forward_head, code: return x if pre_logits else self.head(x)
        permute_49: "f32[768, 1000]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg177_1, view_129, permute_49);  arg177_1 = view_129 = permute_49 = None
        return (addmm_1,)
        