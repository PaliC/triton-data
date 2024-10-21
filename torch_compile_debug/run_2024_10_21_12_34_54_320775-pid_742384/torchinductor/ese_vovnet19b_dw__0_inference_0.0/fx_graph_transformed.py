class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 3, 3]", arg1_1: "f32[8, 3, 288, 288]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64, 1, 3, 3]", arg7_1: "f32[64, 64, 1, 1]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64, 1, 3, 3]", arg13_1: "f32[64, 64, 1, 1]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[128, 64, 1, 1]", arg19_1: "f32[128]", arg20_1: "f32[128]", arg21_1: "f32[128]", arg22_1: "f32[128]", arg23_1: "f32[128, 1, 3, 3]", arg24_1: "f32[128, 128, 1, 1]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[128]", arg28_1: "f32[128]", arg29_1: "f32[128, 1, 3, 3]", arg30_1: "f32[128, 128, 1, 1]", arg31_1: "f32[128]", arg32_1: "f32[128]", arg33_1: "f32[128]", arg34_1: "f32[128]", arg35_1: "f32[128, 1, 3, 3]", arg36_1: "f32[128, 128, 1, 1]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[256, 448, 1, 1]", arg42_1: "f32[256]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[256]", arg46_1: "f32[256, 256, 1, 1]", arg47_1: "f32[256]", arg48_1: "f32[160, 256, 1, 1]", arg49_1: "f32[160]", arg50_1: "f32[160]", arg51_1: "f32[160]", arg52_1: "f32[160]", arg53_1: "f32[160, 1, 3, 3]", arg54_1: "f32[160, 160, 1, 1]", arg55_1: "f32[160]", arg56_1: "f32[160]", arg57_1: "f32[160]", arg58_1: "f32[160]", arg59_1: "f32[160, 1, 3, 3]", arg60_1: "f32[160, 160, 1, 1]", arg61_1: "f32[160]", arg62_1: "f32[160]", arg63_1: "f32[160]", arg64_1: "f32[160]", arg65_1: "f32[160, 1, 3, 3]", arg66_1: "f32[160, 160, 1, 1]", arg67_1: "f32[160]", arg68_1: "f32[160]", arg69_1: "f32[160]", arg70_1: "f32[160]", arg71_1: "f32[512, 736, 1, 1]", arg72_1: "f32[512]", arg73_1: "f32[512]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[512, 512, 1, 1]", arg77_1: "f32[512]", arg78_1: "f32[192, 512, 1, 1]", arg79_1: "f32[192]", arg80_1: "f32[192]", arg81_1: "f32[192]", arg82_1: "f32[192]", arg83_1: "f32[192, 1, 3, 3]", arg84_1: "f32[192, 192, 1, 1]", arg85_1: "f32[192]", arg86_1: "f32[192]", arg87_1: "f32[192]", arg88_1: "f32[192]", arg89_1: "f32[192, 1, 3, 3]", arg90_1: "f32[192, 192, 1, 1]", arg91_1: "f32[192]", arg92_1: "f32[192]", arg93_1: "f32[192]", arg94_1: "f32[192]", arg95_1: "f32[192, 1, 3, 3]", arg96_1: "f32[192, 192, 1, 1]", arg97_1: "f32[192]", arg98_1: "f32[192]", arg99_1: "f32[192]", arg100_1: "f32[192]", arg101_1: "f32[768, 1088, 1, 1]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[768, 768, 1, 1]", arg107_1: "f32[768]", arg108_1: "f32[224, 768, 1, 1]", arg109_1: "f32[224]", arg110_1: "f32[224]", arg111_1: "f32[224]", arg112_1: "f32[224]", arg113_1: "f32[224, 1, 3, 3]", arg114_1: "f32[224, 224, 1, 1]", arg115_1: "f32[224]", arg116_1: "f32[224]", arg117_1: "f32[224]", arg118_1: "f32[224]", arg119_1: "f32[224, 1, 3, 3]", arg120_1: "f32[224, 224, 1, 1]", arg121_1: "f32[224]", arg122_1: "f32[224]", arg123_1: "f32[224]", arg124_1: "f32[224]", arg125_1: "f32[224, 1, 3, 3]", arg126_1: "f32[224, 224, 1, 1]", arg127_1: "f32[224]", arg128_1: "f32[224]", arg129_1: "f32[224]", arg130_1: "f32[224]", arg131_1: "f32[1024, 1440, 1, 1]", arg132_1: "f32[1024]", arg133_1: "f32[1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024]", arg136_1: "f32[1024, 1024, 1, 1]", arg137_1: "f32[1024]", arg138_1: "f32[1000, 1024]", arg139_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_41: "f32[8, 64, 144, 144]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_184: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_185: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        sub_23: "f32[8, 64, 144, 144]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_185);  convolution_41 = unsqueeze_185 = None
        add_50: "f32[64]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_23: "f32[64]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
        reciprocal_23: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_186: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_73, -1);  mul_73 = None
        unsqueeze_187: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_74: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
        unsqueeze_188: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_189: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_75: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_189);  mul_74 = unsqueeze_189 = None
        unsqueeze_190: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_191: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_51: "f32[8, 64, 144, 144]" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_191);  mul_75 = unsqueeze_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_23: "f32[8, 64, 144, 144]" = torch.ops.aten.relu.default(add_51);  add_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_42: "f32[8, 64, 144, 144]" = torch.ops.aten.convolution.default(relu_23, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  relu_23 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_43: "f32[8, 64, 144, 144]" = torch.ops.aten.convolution.default(convolution_42, arg7_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_42 = arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_192: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_193: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        sub_24: "f32[8, 64, 144, 144]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_193);  convolution_43 = unsqueeze_193 = None
        add_52: "f32[64]" = torch.ops.aten.add.Tensor(arg9_1, 1e-05);  arg9_1 = None
        sqrt_24: "f32[64]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_24: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_76: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_194: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
        unsqueeze_195: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_77: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
        unsqueeze_196: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_197: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_78: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_197);  mul_77 = unsqueeze_197 = None
        unsqueeze_198: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_199: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_53: "f32[8, 64, 144, 144]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_199);  mul_78 = unsqueeze_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_24: "f32[8, 64, 144, 144]" = torch.ops.aten.relu.default(add_53);  add_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_44: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(relu_24, arg12_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  relu_24 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_45: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(convolution_44, arg13_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_44 = arg13_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_200: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_201: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        sub_25: "f32[8, 64, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_201);  convolution_45 = unsqueeze_201 = None
        add_54: "f32[64]" = torch.ops.aten.add.Tensor(arg15_1, 1e-05);  arg15_1 = None
        sqrt_25: "f32[64]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_25: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_79: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_202: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_79, -1);  mul_79 = None
        unsqueeze_203: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_80: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
        unsqueeze_204: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_205: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_81: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_205);  mul_80 = unsqueeze_205 = None
        unsqueeze_206: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_207: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_55: "f32[8, 64, 72, 72]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_207);  mul_81 = unsqueeze_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_25: "f32[8, 64, 72, 72]" = torch.ops.aten.relu.default(add_55);  add_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_46: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(relu_25, arg18_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_208: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_209: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        sub_26: "f32[8, 128, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_209);  convolution_46 = unsqueeze_209 = None
        add_56: "f32[128]" = torch.ops.aten.add.Tensor(arg20_1, 1e-05);  arg20_1 = None
        sqrt_26: "f32[128]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
        reciprocal_26: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_82: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_210: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_82, -1);  mul_82 = None
        unsqueeze_211: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_83: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
        unsqueeze_212: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_213: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_84: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_213);  mul_83 = unsqueeze_213 = None
        unsqueeze_214: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_215: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_57: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_84, unsqueeze_215);  mul_84 = unsqueeze_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_26: "f32[8, 128, 72, 72]" = torch.ops.aten.relu.default(add_57);  add_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_47: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(relu_26, arg23_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  relu_26 = arg23_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_48: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(convolution_47, arg24_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_47 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_216: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_217: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        sub_27: "f32[8, 128, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_217);  convolution_48 = unsqueeze_217 = None
        add_58: "f32[128]" = torch.ops.aten.add.Tensor(arg26_1, 1e-05);  arg26_1 = None
        sqrt_27: "f32[128]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
        reciprocal_27: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_218: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_85, -1);  mul_85 = None
        unsqueeze_219: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_86: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
        unsqueeze_220: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_221: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_87: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_86, unsqueeze_221);  mul_86 = unsqueeze_221 = None
        unsqueeze_222: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_223: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_59: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_223);  mul_87 = unsqueeze_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_27: "f32[8, 128, 72, 72]" = torch.ops.aten.relu.default(add_59);  add_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_49: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(relu_27, arg29_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_50: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(convolution_49, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_49 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_224: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_225: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_28: "f32[8, 128, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_225);  convolution_50 = unsqueeze_225 = None
        add_60: "f32[128]" = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_28: "f32[128]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
        reciprocal_28: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
        unsqueeze_227: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_89: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
        unsqueeze_228: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_229: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_90: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_229);  mul_89 = unsqueeze_229 = None
        unsqueeze_230: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_231: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_61: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_231);  mul_90 = unsqueeze_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_28: "f32[8, 128, 72, 72]" = torch.ops.aten.relu.default(add_61);  add_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_51: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(relu_28, arg35_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_52: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(convolution_51, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_51 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_232: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_233: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_29: "f32[8, 128, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_233);  convolution_52 = unsqueeze_233 = None
        add_62: "f32[128]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_29: "f32[128]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
        reciprocal_29: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_91: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
        unsqueeze_235: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_92: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
        unsqueeze_236: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_237: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_93: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_237);  mul_92 = unsqueeze_237 = None
        unsqueeze_238: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_239: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_63: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_239);  mul_93 = unsqueeze_239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_29: "f32[8, 128, 72, 72]" = torch.ops.aten.relu.default(add_63);  add_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vovnet.py:39 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_4: "f32[8, 448, 72, 72]" = torch.ops.aten.cat.default([relu_25, relu_27, relu_28, relu_29], 1);  relu_25 = relu_27 = relu_28 = relu_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_53: "f32[8, 256, 72, 72]" = torch.ops.aten.convolution.default(cat_4, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_240: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_241: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_30: "f32[8, 256, 72, 72]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_241);  convolution_53 = unsqueeze_241 = None
        add_64: "f32[256]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_30: "f32[256]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
        reciprocal_30: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_94: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_94, -1);  mul_94 = None
        unsqueeze_243: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_95: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
        unsqueeze_244: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_245: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_96: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_245);  mul_95 = unsqueeze_245 = None
        unsqueeze_246: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_247: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_65: "f32[8, 256, 72, 72]" = torch.ops.aten.add.Tensor(mul_96, unsqueeze_247);  mul_96 = unsqueeze_247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_30: "f32[8, 256, 72, 72]" = torch.ops.aten.relu.default(add_65);  add_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:66 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_5: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(relu_30, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:70 in forward, code: x_se = self.fc(x_se)
        convolution_54: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_5, arg46_1, arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg46_1 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:71 in forward, code: return x * self.gate(x_se)
        add_66: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(convolution_54, 3);  convolution_54 = None
        clamp_min_4: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
        clamp_max_4: "f32[8, 256, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
        div_4: "f32[8, 256, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_4, 6);  clamp_max_4 = None
        mul_97: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(relu_30, div_4);  relu_30 = div_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vovnet.py:153 in forward, code: x = self.pool(x)
        _low_memory_max_pool2d_with_offsets_3 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(mul_97, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_97 = None
        getitem_6: "f32[8, 256, 36, 36]" = _low_memory_max_pool2d_with_offsets_3[0];  _low_memory_max_pool2d_with_offsets_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_55: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(getitem_6, arg48_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_248: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_249: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_31: "f32[8, 160, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_249);  convolution_55 = unsqueeze_249 = None
        add_67: "f32[160]" = torch.ops.aten.add.Tensor(arg50_1, 1e-05);  arg50_1 = None
        sqrt_31: "f32[160]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
        reciprocal_31: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_98: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_98, -1);  mul_98 = None
        unsqueeze_251: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_99: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
        unsqueeze_252: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_253: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_100: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(mul_99, unsqueeze_253);  mul_99 = unsqueeze_253 = None
        unsqueeze_254: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_255: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_68: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_255);  mul_100 = unsqueeze_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_31: "f32[8, 160, 36, 36]" = torch.ops.aten.relu.default(add_68);  add_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_56: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(relu_31, arg53_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  relu_31 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_57: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(convolution_56, arg54_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_56 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_256: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_257: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_32: "f32[8, 160, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_257);  convolution_57 = unsqueeze_257 = None
        add_69: "f32[160]" = torch.ops.aten.add.Tensor(arg56_1, 1e-05);  arg56_1 = None
        sqrt_32: "f32[160]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
        reciprocal_32: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_101: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_101, -1);  mul_101 = None
        unsqueeze_259: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_102: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
        unsqueeze_260: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_261: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_103: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_261);  mul_102 = unsqueeze_261 = None
        unsqueeze_262: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_263: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_70: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_263);  mul_103 = unsqueeze_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_32: "f32[8, 160, 36, 36]" = torch.ops.aten.relu.default(add_70);  add_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_58: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(relu_32, arg59_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  arg59_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_59: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(convolution_58, arg60_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_58 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_264: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_265: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_33: "f32[8, 160, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_265);  convolution_59 = unsqueeze_265 = None
        add_71: "f32[160]" = torch.ops.aten.add.Tensor(arg62_1, 1e-05);  arg62_1 = None
        sqrt_33: "f32[160]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
        reciprocal_33: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_104: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_104, -1);  mul_104 = None
        unsqueeze_267: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_105: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
        unsqueeze_268: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_269: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_106: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_269);  mul_105 = unsqueeze_269 = None
        unsqueeze_270: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_271: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_72: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_271);  mul_106 = unsqueeze_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_33: "f32[8, 160, 36, 36]" = torch.ops.aten.relu.default(add_72);  add_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_60: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(relu_33, arg65_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 160);  arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_61: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(convolution_60, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_60 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_272: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_273: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_34: "f32[8, 160, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_273);  convolution_61 = unsqueeze_273 = None
        add_73: "f32[160]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_34: "f32[160]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
        reciprocal_34: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_107: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_107, -1);  mul_107 = None
        unsqueeze_275: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_108: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
        unsqueeze_276: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_277: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_109: "f32[8, 160, 36, 36]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_277);  mul_108 = unsqueeze_277 = None
        unsqueeze_278: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_279: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_74: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(mul_109, unsqueeze_279);  mul_109 = unsqueeze_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_34: "f32[8, 160, 36, 36]" = torch.ops.aten.relu.default(add_74);  add_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vovnet.py:39 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_5: "f32[8, 736, 36, 36]" = torch.ops.aten.cat.default([getitem_6, relu_32, relu_33, relu_34], 1);  getitem_6 = relu_32 = relu_33 = relu_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_62: "f32[8, 512, 36, 36]" = torch.ops.aten.convolution.default(cat_5, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_280: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_281: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_35: "f32[8, 512, 36, 36]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_281);  convolution_62 = unsqueeze_281 = None
        add_75: "f32[512]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_35: "f32[512]" = torch.ops.aten.sqrt.default(add_75);  add_75 = None
        reciprocal_35: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_110: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_110, -1);  mul_110 = None
        unsqueeze_283: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_111: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
        unsqueeze_284: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_285: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_112: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_285);  mul_111 = unsqueeze_285 = None
        unsqueeze_286: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_287: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_76: "f32[8, 512, 36, 36]" = torch.ops.aten.add.Tensor(mul_112, unsqueeze_287);  mul_112 = unsqueeze_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_35: "f32[8, 512, 36, 36]" = torch.ops.aten.relu.default(add_76);  add_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:66 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_6: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(relu_35, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:70 in forward, code: x_se = self.fc(x_se)
        convolution_63: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(mean_6, arg76_1, arg77_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_6 = arg76_1 = arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:71 in forward, code: return x * self.gate(x_se)
        add_77: "f32[8, 512, 1, 1]" = torch.ops.aten.add.Tensor(convolution_63, 3);  convolution_63 = None
        clamp_min_5: "f32[8, 512, 1, 1]" = torch.ops.aten.clamp_min.default(add_77, 0);  add_77 = None
        clamp_max_5: "f32[8, 512, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
        div_5: "f32[8, 512, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_5, 6);  clamp_max_5 = None
        mul_113: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(relu_35, div_5);  relu_35 = div_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vovnet.py:153 in forward, code: x = self.pool(x)
        _low_memory_max_pool2d_with_offsets_4 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(mul_113, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_113 = None
        getitem_8: "f32[8, 512, 18, 18]" = _low_memory_max_pool2d_with_offsets_4[0];  _low_memory_max_pool2d_with_offsets_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_64: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(getitem_8, arg78_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_288: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_289: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_36: "f32[8, 192, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_289);  convolution_64 = unsqueeze_289 = None
        add_78: "f32[192]" = torch.ops.aten.add.Tensor(arg80_1, 1e-05);  arg80_1 = None
        sqrt_36: "f32[192]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        reciprocal_36: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_114: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
        unsqueeze_291: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_115: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
        unsqueeze_292: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_293: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_116: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_293);  mul_115 = unsqueeze_293 = None
        unsqueeze_294: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_295: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_79: "f32[8, 192, 18, 18]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_295);  mul_116 = unsqueeze_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_36: "f32[8, 192, 18, 18]" = torch.ops.aten.relu.default(add_79);  add_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_65: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(relu_36, arg83_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  relu_36 = arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_66: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(convolution_65, arg84_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_65 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_296: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_297: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_37: "f32[8, 192, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_297);  convolution_66 = unsqueeze_297 = None
        add_80: "f32[192]" = torch.ops.aten.add.Tensor(arg86_1, 1e-05);  arg86_1 = None
        sqrt_37: "f32[192]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_37: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_117: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
        unsqueeze_299: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_118: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
        unsqueeze_300: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_301: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_119: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_301);  mul_118 = unsqueeze_301 = None
        unsqueeze_302: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_303: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_81: "f32[8, 192, 18, 18]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_303);  mul_119 = unsqueeze_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_37: "f32[8, 192, 18, 18]" = torch.ops.aten.relu.default(add_81);  add_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_67: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(relu_37, arg89_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_68: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(convolution_67, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_67 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_304: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_305: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_38: "f32[8, 192, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_305);  convolution_68 = unsqueeze_305 = None
        add_82: "f32[192]" = torch.ops.aten.add.Tensor(arg92_1, 1e-05);  arg92_1 = None
        sqrt_38: "f32[192]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        reciprocal_38: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_120: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
        unsqueeze_307: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_121: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
        unsqueeze_308: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_309: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_122: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_309);  mul_121 = unsqueeze_309 = None
        unsqueeze_310: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_311: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_83: "f32[8, 192, 18, 18]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_311);  mul_122 = unsqueeze_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_38: "f32[8, 192, 18, 18]" = torch.ops.aten.relu.default(add_83);  add_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_69: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(relu_38, arg95_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_70: "f32[8, 192, 18, 18]" = torch.ops.aten.convolution.default(convolution_69, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_69 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_312: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_313: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_39: "f32[8, 192, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_313);  convolution_70 = unsqueeze_313 = None
        add_84: "f32[192]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_39: "f32[192]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_39: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_123: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_315: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_124: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
        unsqueeze_316: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_317: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_125: "f32[8, 192, 18, 18]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_317);  mul_124 = unsqueeze_317 = None
        unsqueeze_318: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_319: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_85: "f32[8, 192, 18, 18]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_319);  mul_125 = unsqueeze_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_39: "f32[8, 192, 18, 18]" = torch.ops.aten.relu.default(add_85);  add_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vovnet.py:39 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_6: "f32[8, 1088, 18, 18]" = torch.ops.aten.cat.default([getitem_8, relu_37, relu_38, relu_39], 1);  getitem_8 = relu_37 = relu_38 = relu_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_71: "f32[8, 768, 18, 18]" = torch.ops.aten.convolution.default(cat_6, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_320: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_321: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_40: "f32[8, 768, 18, 18]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_321);  convolution_71 = unsqueeze_321 = None
        add_86: "f32[768]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_40: "f32[768]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_40: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_126: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
        unsqueeze_323: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_127: "f32[8, 768, 18, 18]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
        unsqueeze_324: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_325: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_128: "f32[8, 768, 18, 18]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_325);  mul_127 = unsqueeze_325 = None
        unsqueeze_326: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_327: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_87: "f32[8, 768, 18, 18]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_327);  mul_128 = unsqueeze_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_40: "f32[8, 768, 18, 18]" = torch.ops.aten.relu.default(add_87);  add_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:66 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(relu_40, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:70 in forward, code: x_se = self.fc(x_se)
        convolution_72: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(mean_7, arg106_1, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg106_1 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:71 in forward, code: return x * self.gate(x_se)
        add_88: "f32[8, 768, 1, 1]" = torch.ops.aten.add.Tensor(convolution_72, 3);  convolution_72 = None
        clamp_min_6: "f32[8, 768, 1, 1]" = torch.ops.aten.clamp_min.default(add_88, 0);  add_88 = None
        clamp_max_6: "f32[8, 768, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
        div_6: "f32[8, 768, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_6, 6);  clamp_max_6 = None
        mul_129: "f32[8, 768, 18, 18]" = torch.ops.aten.mul.Tensor(relu_40, div_6);  relu_40 = div_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vovnet.py:153 in forward, code: x = self.pool(x)
        _low_memory_max_pool2d_with_offsets_5 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(mul_129, [3, 3], [2, 2], [0, 0], [1, 1], True);  mul_129 = None
        getitem_10: "f32[8, 768, 9, 9]" = _low_memory_max_pool2d_with_offsets_5[0];  _low_memory_max_pool2d_with_offsets_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_73: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(getitem_10, arg108_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_328: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_329: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_41: "f32[8, 224, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_329);  convolution_73 = unsqueeze_329 = None
        add_89: "f32[224]" = torch.ops.aten.add.Tensor(arg110_1, 1e-05);  arg110_1 = None
        sqrt_41: "f32[224]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
        reciprocal_41: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_130: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_130, -1);  mul_130 = None
        unsqueeze_331: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_131: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_333: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_132: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_333);  mul_131 = unsqueeze_333 = None
        unsqueeze_334: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_335: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_90: "f32[8, 224, 9, 9]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_335);  mul_132 = unsqueeze_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_41: "f32[8, 224, 9, 9]" = torch.ops.aten.relu.default(add_90);  add_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_74: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(relu_41, arg113_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  relu_41 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_75: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(convolution_74, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_74 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_336: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_337: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_42: "f32[8, 224, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_337);  convolution_75 = unsqueeze_337 = None
        add_91: "f32[224]" = torch.ops.aten.add.Tensor(arg116_1, 1e-05);  arg116_1 = None
        sqrt_42: "f32[224]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
        reciprocal_42: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_133: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_133, -1);  mul_133 = None
        unsqueeze_339: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_134: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_341: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_135: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(mul_134, unsqueeze_341);  mul_134 = unsqueeze_341 = None
        unsqueeze_342: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_343: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_92: "f32[8, 224, 9, 9]" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_343);  mul_135 = unsqueeze_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_42: "f32[8, 224, 9, 9]" = torch.ops.aten.relu.default(add_92);  add_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_76: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(relu_42, arg119_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  arg119_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_77: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(convolution_76, arg120_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_76 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_344: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_345: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_43: "f32[8, 224, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_345);  convolution_77 = unsqueeze_345 = None
        add_93: "f32[224]" = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_43: "f32[224]" = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_43: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_136: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_136, -1);  mul_136 = None
        unsqueeze_347: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_137: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_349: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_138: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(mul_137, unsqueeze_349);  mul_137 = unsqueeze_349 = None
        unsqueeze_350: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_351: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_94: "f32[8, 224, 9, 9]" = torch.ops.aten.add.Tensor(mul_138, unsqueeze_351);  mul_138 = unsqueeze_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_43: "f32[8, 224, 9, 9]" = torch.ops.aten.relu.default(add_94);  add_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:42 in forward, code: x = self.conv_dw(x)
        convolution_78: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(relu_43, arg125_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 224);  arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/separable_conv.py:43 in forward, code: x = self.conv_pw(x)
        convolution_79: "f32[8, 224, 9, 9]" = torch.ops.aten.convolution.default(convolution_78, arg126_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_78 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_352: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_353: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_44: "f32[8, 224, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_353);  convolution_79 = unsqueeze_353 = None
        add_95: "f32[224]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_44: "f32[224]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_44: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_139: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
        unsqueeze_355: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_140: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_357: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_141: "f32[8, 224, 9, 9]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_357);  mul_140 = unsqueeze_357 = None
        unsqueeze_358: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_359: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_96: "f32[8, 224, 9, 9]" = torch.ops.aten.add.Tensor(mul_141, unsqueeze_359);  mul_141 = unsqueeze_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_44: "f32[8, 224, 9, 9]" = torch.ops.aten.relu.default(add_96);  add_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vovnet.py:39 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_7: "f32[8, 1440, 9, 9]" = torch.ops.aten.cat.default([getitem_10, relu_42, relu_43, relu_44], 1);  getitem_10 = relu_42 = relu_43 = relu_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_80: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(cat_7, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_360: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_361: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_45: "f32[8, 1024, 9, 9]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_361);  convolution_80 = unsqueeze_361 = None
        add_97: "f32[1024]" = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_45: "f32[1024]" = torch.ops.aten.sqrt.default(add_97);  add_97 = None
        reciprocal_45: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_142: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_142, -1);  mul_142 = None
        unsqueeze_363: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_143: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_365: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_144: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_365);  mul_143 = unsqueeze_365 = None
        unsqueeze_366: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_367: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_98: "f32[8, 1024, 9, 9]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_367);  mul_144 = unsqueeze_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_45: "f32[8, 1024, 9, 9]" = torch.ops.aten.relu.default(add_98);  add_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:66 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_45, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:70 in forward, code: x_se = self.fc(x_se)
        convolution_81: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(mean_8, arg136_1, arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg136_1 = arg137_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:71 in forward, code: return x * self.gate(x_se)
        add_99: "f32[8, 1024, 1, 1]" = torch.ops.aten.add.Tensor(convolution_81, 3);  convolution_81 = None
        clamp_min_7: "f32[8, 1024, 1, 1]" = torch.ops.aten.clamp_min.default(add_99, 0);  add_99 = None
        clamp_max_7: "f32[8, 1024, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
        div_7: "f32[8, 1024, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_7, 6);  clamp_max_7 = None
        mul_145: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(relu_45, div_7);  relu_45 = div_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_9: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(mul_145, [-1, -2], True);  mul_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 1024]" = torch.ops.aten.reshape.default(mean_9, [8, 1024]);  mean_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_1: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg139_1, view_1, permute_1);  arg139_1 = view_1 = permute_1 = None
        return (addmm_1,)
        