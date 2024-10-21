class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[8]", arg3_1: "f32[8]", arg4_1: "f32[8]", arg5_1: "f32[8]", arg6_1: "f32[8, 1, 3, 3]", arg7_1: "f32[8]", arg8_1: "f32[8]", arg9_1: "f32[8]", arg10_1: "f32[8]", arg11_1: "f32[16, 8, 1, 1]", arg12_1: "f32[16]", arg13_1: "f32[16]", arg14_1: "f32[16]", arg15_1: "f32[16]", arg16_1: "f32[16, 1, 3, 3]", arg17_1: "f32[16]", arg18_1: "f32[16]", arg19_1: "f32[16]", arg20_1: "f32[16]", arg21_1: "f32[32, 16, 1, 1]", arg22_1: "f32[32]", arg23_1: "f32[32]", arg24_1: "f32[32]", arg25_1: "f32[32]", arg26_1: "f32[32, 1, 3, 3]", arg27_1: "f32[32]", arg28_1: "f32[32]", arg29_1: "f32[32]", arg30_1: "f32[32]", arg31_1: "f32[32, 32, 1, 1]", arg32_1: "f32[32]", arg33_1: "f32[32]", arg34_1: "f32[32]", arg35_1: "f32[32]", arg36_1: "f32[32, 1, 3, 3]", arg37_1: "f32[32]", arg38_1: "f32[32]", arg39_1: "f32[32]", arg40_1: "f32[32]", arg41_1: "f32[64, 32, 1, 1]", arg42_1: "f32[64]", arg43_1: "f32[64]", arg44_1: "f32[64]", arg45_1: "f32[64]", arg46_1: "f32[64, 1, 3, 3]", arg47_1: "f32[64]", arg48_1: "f32[64]", arg49_1: "f32[64]", arg50_1: "f32[64]", arg51_1: "f32[64, 64, 1, 1]", arg52_1: "f32[64]", arg53_1: "f32[64]", arg54_1: "f32[64]", arg55_1: "f32[64]", arg56_1: "f32[64, 1, 3, 3]", arg57_1: "f32[64]", arg58_1: "f32[64]", arg59_1: "f32[64]", arg60_1: "f32[64]", arg61_1: "f32[128, 64, 1, 1]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[128]", arg66_1: "f32[128, 1, 5, 5]", arg67_1: "f32[128]", arg68_1: "f32[128]", arg69_1: "f32[128]", arg70_1: "f32[128]", arg71_1: "f32[128, 128, 1, 1]", arg72_1: "f32[128]", arg73_1: "f32[128]", arg74_1: "f32[128]", arg75_1: "f32[128]", arg76_1: "f32[128, 1, 5, 5]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[128]", arg80_1: "f32[128]", arg81_1: "f32[128, 128, 1, 1]", arg82_1: "f32[128]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[128]", arg86_1: "f32[128, 1, 5, 5]", arg87_1: "f32[128]", arg88_1: "f32[128]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[128, 128, 1, 1]", arg92_1: "f32[128]", arg93_1: "f32[128]", arg94_1: "f32[128]", arg95_1: "f32[128]", arg96_1: "f32[128, 1, 5, 5]", arg97_1: "f32[128]", arg98_1: "f32[128]", arg99_1: "f32[128]", arg100_1: "f32[128]", arg101_1: "f32[128, 128, 1, 1]", arg102_1: "f32[128]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128]", arg106_1: "f32[128, 1, 5, 5]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "f32[128]", arg110_1: "f32[128]", arg111_1: "f32[128, 128, 1, 1]", arg112_1: "f32[128]", arg113_1: "f32[128]", arg114_1: "f32[128]", arg115_1: "f32[128]", arg116_1: "f32[128, 1, 5, 5]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[32, 128, 1, 1]", arg122_1: "f32[32]", arg123_1: "f32[128, 32, 1, 1]", arg124_1: "f32[128]", arg125_1: "f32[256, 128, 1, 1]", arg126_1: "f32[256]", arg127_1: "f32[256]", arg128_1: "f32[256]", arg129_1: "f32[256]", arg130_1: "f32[256, 1, 5, 5]", arg131_1: "f32[256]", arg132_1: "f32[256]", arg133_1: "f32[256]", arg134_1: "f32[256]", arg135_1: "f32[64, 256, 1, 1]", arg136_1: "f32[64]", arg137_1: "f32[256, 64, 1, 1]", arg138_1: "f32[256]", arg139_1: "f32[256, 256, 1, 1]", arg140_1: "f32[256]", arg141_1: "f32[256]", arg142_1: "f32[256]", arg143_1: "f32[256]", arg144_1: "f32[1280, 256, 1, 1]", arg145_1: "f32[1280]", arg146_1: "f32[1000, 1280]", arg147_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:251 in forward_features, code: x = self.conv_stem(x)
        convolution_32: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_216: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_217: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        sub_27: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_217);  convolution_32 = unsqueeze_217 = None
        add_84: "f32[8]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_27: "f32[8]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_27: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_111: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_218: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
        unsqueeze_219: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_112: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
        unsqueeze_220: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_221: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_113: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_221);  mul_112 = unsqueeze_221 = None
        unsqueeze_222: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_223: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_85: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_223);  mul_113 = unsqueeze_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_86: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(add_85, 3)
        clamp_min_30: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_min.default(add_86, 0);  add_86 = None
        clamp_max_30: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6);  clamp_min_30 = None
        mul_114: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_85, clamp_max_30);  add_85 = clamp_max_30 = None
        div_30: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(mul_114, 6);  mul_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_33: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(div_30, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  div_30 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_224: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_225: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_28: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_225);  convolution_33 = unsqueeze_225 = None
        add_87: "f32[8]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_28: "f32[8]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
        reciprocal_28: "f32[8]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_115: "f32[8]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(mul_115, -1);  mul_115 = None
        unsqueeze_227: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_116: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
        unsqueeze_228: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_229: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_117: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_229);  mul_116 = unsqueeze_229 = None
        unsqueeze_230: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_231: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_88: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_117, unsqueeze_231);  mul_117 = unsqueeze_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_89: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(add_88, 3)
        clamp_min_31: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_min.default(add_89, 0);  add_89 = None
        clamp_max_31: "f32[8, 8, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_31, 6);  clamp_min_31 = None
        mul_118: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_88, clamp_max_31);  add_88 = clamp_max_31 = None
        div_31: "f32[8, 8, 112, 112]" = torch.ops.aten.div.Tensor(mul_118, 6);  mul_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_34: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(div_31, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_31 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_232: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_233: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_29: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_233);  convolution_34 = unsqueeze_233 = None
        add_90: "f32[16]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_29: "f32[16]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
        reciprocal_29: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_119: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_119, -1);  mul_119 = None
        unsqueeze_235: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_120: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
        unsqueeze_236: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_237: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_121: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_237);  mul_120 = unsqueeze_237 = None
        unsqueeze_238: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_239: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_91: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_239);  mul_121 = unsqueeze_239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_92: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_91, 3)
        clamp_min_32: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_92, 0);  add_92 = None
        clamp_max_32: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_32, 6);  clamp_min_32 = None
        mul_122: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_91, clamp_max_32);  add_91 = clamp_max_32 = None
        div_32: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_122, 6);  mul_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_35: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(div_32, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  div_32 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_240: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_241: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_30: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_241);  convolution_35 = unsqueeze_241 = None
        add_93: "f32[16]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_30: "f32[16]" = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_30: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_123: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_243: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_124: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
        unsqueeze_244: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_245: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_125: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_245);  mul_124 = unsqueeze_245 = None
        unsqueeze_246: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_247: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_94: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_247);  mul_125 = unsqueeze_247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_95: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(add_94, 3)
        clamp_min_33: "f32[8, 16, 56, 56]" = torch.ops.aten.clamp_min.default(add_95, 0);  add_95 = None
        clamp_max_33: "f32[8, 16, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_33, 6);  clamp_min_33 = None
        mul_126: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(add_94, clamp_max_33);  add_94 = clamp_max_33 = None
        div_33: "f32[8, 16, 56, 56]" = torch.ops.aten.div.Tensor(mul_126, 6);  mul_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_36: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_33, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_33 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_248: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_249: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_31: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_249);  convolution_36 = unsqueeze_249 = None
        add_96: "f32[32]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_31: "f32[32]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        reciprocal_31: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_127: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_127, -1);  mul_127 = None
        unsqueeze_251: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_128: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
        unsqueeze_252: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_253: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_129: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_253);  mul_128 = unsqueeze_253 = None
        unsqueeze_254: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_255: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_97: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_255);  mul_129 = unsqueeze_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_98: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_97, 3)
        clamp_min_34: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_98, 0);  add_98 = None
        clamp_max_34: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_34, 6);  clamp_min_34 = None
        mul_130: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_97, clamp_max_34);  add_97 = clamp_max_34 = None
        div_34: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_130, 6);  mul_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_37: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_34, arg26_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  div_34 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_256: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_257: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_32: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_257);  convolution_37 = unsqueeze_257 = None
        add_99: "f32[32]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_32: "f32[32]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
        reciprocal_32: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_131: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_131, -1);  mul_131 = None
        unsqueeze_259: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_132: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
        unsqueeze_260: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_261: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_133: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_261);  mul_132 = unsqueeze_261 = None
        unsqueeze_262: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_263: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_100: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_133, unsqueeze_263);  mul_133 = unsqueeze_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_101: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_100, 3)
        clamp_min_35: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_101, 0);  add_101 = None
        clamp_max_35: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_35, 6);  clamp_min_35 = None
        mul_134: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_100, clamp_max_35);  add_100 = clamp_max_35 = None
        div_35: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_134, 6);  mul_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_38: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_35, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_35 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_264: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_265: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_33: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_265);  convolution_38 = unsqueeze_265 = None
        add_102: "f32[32]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_33: "f32[32]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_33: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_135: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_267: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_136: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
        unsqueeze_268: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_269: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_137: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_269);  mul_136 = unsqueeze_269 = None
        unsqueeze_270: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_271: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_103: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_271);  mul_137 = unsqueeze_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_104: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_103, 3)
        clamp_min_36: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_104, 0);  add_104 = None
        clamp_max_36: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_36, 6);  clamp_min_36 = None
        mul_138: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_103, clamp_max_36);  add_103 = clamp_max_36 = None
        div_36: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_138, 6);  mul_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_39: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(div_36, arg36_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32);  div_36 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_272: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_273: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_34: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_273);  convolution_39 = unsqueeze_273 = None
        add_105: "f32[32]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_34: "f32[32]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
        reciprocal_34: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_139: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
        unsqueeze_275: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_140: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
        unsqueeze_276: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_277: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_141: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_277);  mul_140 = unsqueeze_277 = None
        unsqueeze_278: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_279: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_106: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_141, unsqueeze_279);  mul_141 = unsqueeze_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_107: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_106, 3)
        clamp_min_37: "f32[8, 32, 28, 28]" = torch.ops.aten.clamp_min.default(add_107, 0);  add_107 = None
        clamp_max_37: "f32[8, 32, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_37, 6);  clamp_min_37 = None
        mul_142: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_106, clamp_max_37);  add_106 = clamp_max_37 = None
        div_37: "f32[8, 32, 28, 28]" = torch.ops.aten.div.Tensor(mul_142, 6);  mul_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_40: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_37, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_37 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_280: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_281: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_35: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_281);  convolution_40 = unsqueeze_281 = None
        add_108: "f32[64]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_35: "f32[64]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
        reciprocal_35: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_143: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_143, -1);  mul_143 = None
        unsqueeze_283: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_144: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
        unsqueeze_284: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_285: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_145: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_285);  mul_144 = unsqueeze_285 = None
        unsqueeze_286: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_287: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_109: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_287);  mul_145 = unsqueeze_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_110: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_109, 3)
        clamp_min_38: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_110, 0);  add_110 = None
        clamp_max_38: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_38, 6);  clamp_min_38 = None
        mul_146: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_109, clamp_max_38);  add_109 = clamp_max_38 = None
        div_38: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_146, 6);  mul_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_41: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_38, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  div_38 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_288: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_289: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_36: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_289);  convolution_41 = unsqueeze_289 = None
        add_111: "f32[64]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_36: "f32[64]" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_36: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_147: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_291: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_148: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
        unsqueeze_292: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_293: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_149: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_293);  mul_148 = unsqueeze_293 = None
        unsqueeze_294: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_295: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_112: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_295);  mul_149 = unsqueeze_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_113: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_112, 3)
        clamp_min_39: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_113, 0);  add_113 = None
        clamp_max_39: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_39, 6);  clamp_min_39 = None
        mul_150: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_112, clamp_max_39);  add_112 = clamp_max_39 = None
        div_39: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_150, 6);  mul_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_42: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_39, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_39 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_296: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_297: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_37: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_297);  convolution_42 = unsqueeze_297 = None
        add_114: "f32[64]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_37: "f32[64]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_37: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_151: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_151, -1);  mul_151 = None
        unsqueeze_299: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_152: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
        unsqueeze_300: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_301: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_153: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_152, unsqueeze_301);  mul_152 = unsqueeze_301 = None
        unsqueeze_302: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_303: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_115: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_303);  mul_153 = unsqueeze_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_116: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_115, 3)
        clamp_min_40: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_116, 0);  add_116 = None
        clamp_max_40: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_40, 6);  clamp_min_40 = None
        mul_154: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_115, clamp_max_40);  add_115 = clamp_max_40 = None
        div_40: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_154, 6);  mul_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_43: "f32[8, 64, 14, 14]" = torch.ops.aten.convolution.default(div_40, arg56_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  div_40 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_304: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_305: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_38: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_305);  convolution_43 = unsqueeze_305 = None
        add_117: "f32[64]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_38: "f32[64]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_38: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_155: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_155, -1);  mul_155 = None
        unsqueeze_307: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_156: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
        unsqueeze_308: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_309: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_157: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_309);  mul_156 = unsqueeze_309 = None
        unsqueeze_310: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_311: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_118: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_311);  mul_157 = unsqueeze_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_119: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_118, 3)
        clamp_min_41: "f32[8, 64, 14, 14]" = torch.ops.aten.clamp_min.default(add_119, 0);  add_119 = None
        clamp_max_41: "f32[8, 64, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_41, 6);  clamp_min_41 = None
        mul_158: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_118, clamp_max_41);  add_118 = clamp_max_41 = None
        div_41: "f32[8, 64, 14, 14]" = torch.ops.aten.div.Tensor(mul_158, 6);  mul_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_44: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_41, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_41 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_312: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_313: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_39: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_313);  convolution_44 = unsqueeze_313 = None
        add_120: "f32[128]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_39: "f32[128]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_39: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_159: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_315: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_160: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
        unsqueeze_316: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_317: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_161: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_317);  mul_160 = unsqueeze_317 = None
        unsqueeze_318: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_319: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_121: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_319);  mul_161 = unsqueeze_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_122: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_121, 3)
        clamp_min_42: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_122, 0);  add_122 = None
        clamp_max_42: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_42, 6);  clamp_min_42 = None
        mul_162: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_121, clamp_max_42);  add_121 = clamp_max_42 = None
        div_42: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_162, 6);  mul_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_45: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_42, arg66_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_42 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_320: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_321: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_40: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_321);  convolution_45 = unsqueeze_321 = None
        add_123: "f32[128]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_40: "f32[128]" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_40: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_163, -1);  mul_163 = None
        unsqueeze_323: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_164: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
        unsqueeze_324: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_325: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_165: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_325);  mul_164 = unsqueeze_325 = None
        unsqueeze_326: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_327: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_124: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_165, unsqueeze_327);  mul_165 = unsqueeze_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_125: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_124, 3)
        clamp_min_43: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_125, 0);  add_125 = None
        clamp_max_43: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_43, 6);  clamp_min_43 = None
        mul_166: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_124, clamp_max_43);  add_124 = clamp_max_43 = None
        div_43: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_166, 6);  mul_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_46: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_43, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_43 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_328: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_329: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_41: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_329);  convolution_46 = unsqueeze_329 = None
        add_126: "f32[128]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_41: "f32[128]" = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_41: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_167: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_167, -1);  mul_167 = None
        unsqueeze_331: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_168: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_333: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_169: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_333);  mul_168 = unsqueeze_333 = None
        unsqueeze_334: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_335: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_127: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_335);  mul_169 = unsqueeze_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_128: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_127, 3)
        clamp_min_44: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_128, 0);  add_128 = None
        clamp_max_44: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_44, 6);  clamp_min_44 = None
        mul_170: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_127, clamp_max_44);  add_127 = clamp_max_44 = None
        div_44: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_170, 6);  mul_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_47: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_44, arg76_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_44 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_336: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_337: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_42: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_337);  convolution_47 = unsqueeze_337 = None
        add_129: "f32[128]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_42: "f32[128]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_42: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_171: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_339: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_172: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_341: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_173: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_341);  mul_172 = unsqueeze_341 = None
        unsqueeze_342: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_343: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_130: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_343);  mul_173 = unsqueeze_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_131: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_130, 3)
        clamp_min_45: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_131, 0);  add_131 = None
        clamp_max_45: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_45, 6);  clamp_min_45 = None
        mul_174: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_130, clamp_max_45);  add_130 = clamp_max_45 = None
        div_45: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_174, 6);  mul_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_48: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_45, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_45 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_344: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_345: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_43: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_345);  convolution_48 = unsqueeze_345 = None
        add_132: "f32[128]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_43: "f32[128]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_43: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_175: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_175, -1);  mul_175 = None
        unsqueeze_347: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_176: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_349: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_177: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_349);  mul_176 = unsqueeze_349 = None
        unsqueeze_350: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_351: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_133: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_351);  mul_177 = unsqueeze_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_134: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_133, 3)
        clamp_min_46: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_134, 0);  add_134 = None
        clamp_max_46: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_46, 6);  clamp_min_46 = None
        mul_178: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_133, clamp_max_46);  add_133 = clamp_max_46 = None
        div_46: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_178, 6);  mul_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_49: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_46, arg86_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_46 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_352: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_353: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_44: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_353);  convolution_49 = unsqueeze_353 = None
        add_135: "f32[128]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_44: "f32[128]" = torch.ops.aten.sqrt.default(add_135);  add_135 = None
        reciprocal_44: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
        unsqueeze_355: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_180: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_357: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_181: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_357);  mul_180 = unsqueeze_357 = None
        unsqueeze_358: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_359: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_136: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_359);  mul_181 = unsqueeze_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_137: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_136, 3)
        clamp_min_47: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_137, 0);  add_137 = None
        clamp_max_47: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_47, 6);  clamp_min_47 = None
        mul_182: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_136, clamp_max_47);  add_136 = clamp_max_47 = None
        div_47: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_182, 6);  mul_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_50: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_47, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_47 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_360: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_361: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_45: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_361);  convolution_50 = unsqueeze_361 = None
        add_138: "f32[128]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_45: "f32[128]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_45: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_183: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_363: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_184: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_365: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_185: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_365);  mul_184 = unsqueeze_365 = None
        unsqueeze_366: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_367: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_139: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_367);  mul_185 = unsqueeze_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_140: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_139, 3)
        clamp_min_48: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_140, 0);  add_140 = None
        clamp_max_48: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_48, 6);  clamp_min_48 = None
        mul_186: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_139, clamp_max_48);  add_139 = clamp_max_48 = None
        div_48: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_186, 6);  mul_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_51: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_48, arg96_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_48 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_368: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_369: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        sub_46: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_369);  convolution_51 = unsqueeze_369 = None
        add_141: "f32[128]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_46: "f32[128]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_46: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_187: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_370: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_187, -1);  mul_187 = None
        unsqueeze_371: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        mul_188: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_373: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_189: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_373);  mul_188 = unsqueeze_373 = None
        unsqueeze_374: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_375: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_142: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_375);  mul_189 = unsqueeze_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_143: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_142, 3)
        clamp_min_49: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_143, 0);  add_143 = None
        clamp_max_49: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_49, 6);  clamp_min_49 = None
        mul_190: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_142, clamp_max_49);  add_142 = clamp_max_49 = None
        div_49: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_190, 6);  mul_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_52: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_49, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_49 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_376: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_377: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        sub_47: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_377);  convolution_52 = unsqueeze_377 = None
        add_144: "f32[128]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_47: "f32[128]" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_47: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_191: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_378: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_191, -1);  mul_191 = None
        unsqueeze_379: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        mul_192: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_381: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_193: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_381);  mul_192 = unsqueeze_381 = None
        unsqueeze_382: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_383: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_145: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_383);  mul_193 = unsqueeze_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_146: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_145, 3)
        clamp_min_50: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_146, 0);  add_146 = None
        clamp_max_50: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_50, 6);  clamp_min_50 = None
        mul_194: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_145, clamp_max_50);  add_145 = clamp_max_50 = None
        div_50: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_194, 6);  mul_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_53: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_50, arg106_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_50 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_384: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_385: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        sub_48: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_385);  convolution_53 = unsqueeze_385 = None
        add_147: "f32[128]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_48: "f32[128]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_48: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_195: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_386: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_387: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        mul_196: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_389: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_197: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_389);  mul_196 = unsqueeze_389 = None
        unsqueeze_390: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_391: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_148: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_391);  mul_197 = unsqueeze_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_149: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_148, 3)
        clamp_min_51: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_149, 0);  add_149 = None
        clamp_max_51: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_51, 6);  clamp_min_51 = None
        mul_198: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_148, clamp_max_51);  add_148 = clamp_max_51 = None
        div_51: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_198, 6);  mul_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_54: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_51, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_51 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_392: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_393: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        sub_49: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_393);  convolution_54 = unsqueeze_393 = None
        add_150: "f32[128]" = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_49: "f32[128]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_49: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_199: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_394: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
        unsqueeze_395: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        mul_200: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_397: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_201: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_397);  mul_200 = unsqueeze_397 = None
        unsqueeze_398: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_399: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_151: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_399);  mul_201 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_152: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(add_151, 3)
        clamp_min_52: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_min.default(add_152, 0);  add_152 = None
        clamp_max_52: "f32[8, 128, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_52, 6);  clamp_min_52 = None
        mul_202: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_151, clamp_max_52);  add_151 = clamp_max_52 = None
        div_52: "f32[8, 128, 14, 14]" = torch.ops.aten.div.Tensor(mul_202, 6);  mul_202 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_55: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(div_52, arg116_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 128);  div_52 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_400: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_401: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        sub_50: "f32[8, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_401);  convolution_55 = unsqueeze_401 = None
        add_153: "f32[128]" = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_50: "f32[128]" = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_50: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_203: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_402: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_203, -1);  mul_203 = None
        unsqueeze_403: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        mul_204: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_405: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_205: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_405);  mul_204 = unsqueeze_405 = None
        unsqueeze_406: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_407: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_154: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_407);  mul_205 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_155: "f32[8, 128, 7, 7]" = torch.ops.aten.add.Tensor(add_154, 3)
        clamp_min_53: "f32[8, 128, 7, 7]" = torch.ops.aten.clamp_min.default(add_155, 0);  add_155 = None
        clamp_max_53: "f32[8, 128, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_53, 6);  clamp_min_53 = None
        mul_206: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(add_154, clamp_max_53);  add_154 = clamp_max_53 = None
        div_53: "f32[8, 128, 7, 7]" = torch.ops.aten.div.Tensor(mul_206, 6);  mul_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_3: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(div_53, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_56: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_3, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg121_1 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_2: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_56);  convolution_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_57: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_2, arg123_1, arg124_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg123_1 = arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_156: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(convolution_57, 3);  convolution_57 = None
        clamp_min_54: "f32[8, 128, 1, 1]" = torch.ops.aten.clamp_min.default(add_156, 0);  add_156 = None
        clamp_max_54: "f32[8, 128, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_54, 6);  clamp_min_54 = None
        div_54: "f32[8, 128, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_54, 6);  clamp_max_54 = None
        mul_207: "f32[8, 128, 7, 7]" = torch.ops.aten.mul.Tensor(div_53, div_54);  div_53 = div_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_58: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(mul_207, arg125_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_207 = arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_408: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_409: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        sub_51: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_409);  convolution_58 = unsqueeze_409 = None
        add_157: "f32[256]" = torch.ops.aten.add.Tensor(arg127_1, 1e-05);  arg127_1 = None
        sqrt_51: "f32[256]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_51: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_208: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_410: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
        unsqueeze_411: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        mul_209: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_413: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_210: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_413);  mul_209 = unsqueeze_413 = None
        unsqueeze_414: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_415: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_158: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_415);  mul_210 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_159: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_158, 3)
        clamp_min_55: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_159, 0);  add_159 = None
        clamp_max_55: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_55, 6);  clamp_min_55 = None
        mul_211: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_158, clamp_max_55);  add_158 = clamp_max_55 = None
        div_55: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_211, 6);  mul_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_59: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(div_55, arg130_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 256);  div_55 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_416: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_417: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        sub_52: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_417);  convolution_59 = unsqueeze_417 = None
        add_160: "f32[256]" = torch.ops.aten.add.Tensor(arg132_1, 1e-05);  arg132_1 = None
        sqrt_52: "f32[256]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_52: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_212: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_418: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_212, -1);  mul_212 = None
        unsqueeze_419: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        mul_213: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_421: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_214: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_421);  mul_213 = unsqueeze_421 = None
        unsqueeze_422: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_423: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_161: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_423);  mul_214 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_162: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_161, 3)
        clamp_min_56: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_162, 0);  add_162 = None
        clamp_max_56: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_56, 6);  clamp_min_56 = None
        mul_215: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_161, clamp_max_56);  add_161 = clamp_max_56 = None
        div_56: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_215, 6);  mul_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_4: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(div_56, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_60: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_4, arg135_1, arg136_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg135_1 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        relu_3: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_60);  convolution_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_61: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_3, arg137_1, arg138_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_3 = arg137_1 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        add_163: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(convolution_61, 3);  convolution_61 = None
        clamp_min_57: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_min.default(add_163, 0);  add_163 = None
        clamp_max_57: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_57, 6);  clamp_min_57 = None
        div_57: "f32[8, 256, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_57, 6);  clamp_max_57 = None
        mul_216: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(div_56, div_57);  div_56 = div_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_62: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(mul_216, arg139_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_216 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_424: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_425: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        sub_53: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_425);  convolution_62 = unsqueeze_425 = None
        add_164: "f32[256]" = torch.ops.aten.add.Tensor(arg141_1, 1e-05);  arg141_1 = None
        sqrt_53: "f32[256]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_53: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_217: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_426: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_217, -1);  mul_217 = None
        unsqueeze_427: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        mul_218: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_429: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_219: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_429);  mul_218 = unsqueeze_429 = None
        unsqueeze_430: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_431: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_165: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_431);  mul_219 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        add_166: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(add_165, 3)
        clamp_min_58: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_min.default(add_166, 0);  add_166 = None
        clamp_max_58: "f32[8, 256, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_58, 6);  clamp_min_58 = None
        mul_220: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(add_165, clamp_max_58);  add_165 = clamp_max_58 = None
        div_58: "f32[8, 256, 7, 7]" = torch.ops.aten.div.Tensor(mul_220, 6);  mul_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_5: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(div_58, [-1, -2], True);  div_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:261 in forward_head, code: x = self.conv_head(x)
        convolution_63: "f32[8, 1280, 1, 1]" = torch.ops.aten.convolution.default(mean_5, arg144_1, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg144_1 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilenetv3.py:263 in forward_head, code: x = self.act2(x)
        add_167: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convolution_63, 3)
        clamp_min_59: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add_167, 0);  add_167 = None
        clamp_max_59: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_59, 6);  clamp_min_59 = None
        mul_221: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_63, clamp_max_59);  convolution_63 = clamp_max_59 = None
        div_59: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul_221, 6);  mul_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        view_3: "f32[8, 1280]" = torch.ops.aten.reshape.default(div_59, [8, 1280]);  div_59 = None
        permute_1: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg147_1, view_3, permute_1);  arg147_1 = view_3 = permute_1 = None
        return (addmm_1,)
        