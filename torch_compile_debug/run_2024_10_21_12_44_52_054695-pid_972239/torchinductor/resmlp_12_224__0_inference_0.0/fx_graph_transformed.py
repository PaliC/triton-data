class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[384, 3, 16, 16]", arg2_1: "f32[384]", arg3_1: "f32[384]", arg4_1: "f32[1, 1, 384]", arg5_1: "f32[1, 1, 384]", arg6_1: "f32[196, 196]", arg7_1: "f32[196]", arg8_1: "f32[384]", arg9_1: "f32[1, 1, 384]", arg10_1: "f32[1, 1, 384]", arg11_1: "f32[1536, 384]", arg12_1: "f32[1536]", arg13_1: "f32[384, 1536]", arg14_1: "f32[384]", arg15_1: "f32[384]", arg16_1: "f32[1, 1, 384]", arg17_1: "f32[1, 1, 384]", arg18_1: "f32[196, 196]", arg19_1: "f32[196]", arg20_1: "f32[384]", arg21_1: "f32[1, 1, 384]", arg22_1: "f32[1, 1, 384]", arg23_1: "f32[1536, 384]", arg24_1: "f32[1536]", arg25_1: "f32[384, 1536]", arg26_1: "f32[384]", arg27_1: "f32[384]", arg28_1: "f32[1, 1, 384]", arg29_1: "f32[1, 1, 384]", arg30_1: "f32[196, 196]", arg31_1: "f32[196]", arg32_1: "f32[384]", arg33_1: "f32[1, 1, 384]", arg34_1: "f32[1, 1, 384]", arg35_1: "f32[1536, 384]", arg36_1: "f32[1536]", arg37_1: "f32[384, 1536]", arg38_1: "f32[384]", arg39_1: "f32[384]", arg40_1: "f32[1, 1, 384]", arg41_1: "f32[1, 1, 384]", arg42_1: "f32[196, 196]", arg43_1: "f32[196]", arg44_1: "f32[384]", arg45_1: "f32[1, 1, 384]", arg46_1: "f32[1, 1, 384]", arg47_1: "f32[1536, 384]", arg48_1: "f32[1536]", arg49_1: "f32[384, 1536]", arg50_1: "f32[384]", arg51_1: "f32[384]", arg52_1: "f32[1, 1, 384]", arg53_1: "f32[1, 1, 384]", arg54_1: "f32[196, 196]", arg55_1: "f32[196]", arg56_1: "f32[384]", arg57_1: "f32[1, 1, 384]", arg58_1: "f32[1, 1, 384]", arg59_1: "f32[1536, 384]", arg60_1: "f32[1536]", arg61_1: "f32[384, 1536]", arg62_1: "f32[384]", arg63_1: "f32[384]", arg64_1: "f32[1, 1, 384]", arg65_1: "f32[1, 1, 384]", arg66_1: "f32[196, 196]", arg67_1: "f32[196]", arg68_1: "f32[384]", arg69_1: "f32[1, 1, 384]", arg70_1: "f32[1, 1, 384]", arg71_1: "f32[1536, 384]", arg72_1: "f32[1536]", arg73_1: "f32[384, 1536]", arg74_1: "f32[384]", arg75_1: "f32[384]", arg76_1: "f32[1, 1, 384]", arg77_1: "f32[1, 1, 384]", arg78_1: "f32[196, 196]", arg79_1: "f32[196]", arg80_1: "f32[384]", arg81_1: "f32[1, 1, 384]", arg82_1: "f32[1, 1, 384]", arg83_1: "f32[1536, 384]", arg84_1: "f32[1536]", arg85_1: "f32[384, 1536]", arg86_1: "f32[384]", arg87_1: "f32[384]", arg88_1: "f32[1, 1, 384]", arg89_1: "f32[1, 1, 384]", arg90_1: "f32[196, 196]", arg91_1: "f32[196]", arg92_1: "f32[384]", arg93_1: "f32[1, 1, 384]", arg94_1: "f32[1, 1, 384]", arg95_1: "f32[1536, 384]", arg96_1: "f32[1536]", arg97_1: "f32[384, 1536]", arg98_1: "f32[384]", arg99_1: "f32[384]", arg100_1: "f32[1, 1, 384]", arg101_1: "f32[1, 1, 384]", arg102_1: "f32[196, 196]", arg103_1: "f32[196]", arg104_1: "f32[384]", arg105_1: "f32[1, 1, 384]", arg106_1: "f32[1, 1, 384]", arg107_1: "f32[1536, 384]", arg108_1: "f32[1536]", arg109_1: "f32[384, 1536]", arg110_1: "f32[384]", arg111_1: "f32[384]", arg112_1: "f32[1, 1, 384]", arg113_1: "f32[1, 1, 384]", arg114_1: "f32[196, 196]", arg115_1: "f32[196]", arg116_1: "f32[384]", arg117_1: "f32[1, 1, 384]", arg118_1: "f32[1, 1, 384]", arg119_1: "f32[1536, 384]", arg120_1: "f32[1536]", arg121_1: "f32[384, 1536]", arg122_1: "f32[384]", arg123_1: "f32[384]", arg124_1: "f32[1, 1, 384]", arg125_1: "f32[1, 1, 384]", arg126_1: "f32[196, 196]", arg127_1: "f32[196]", arg128_1: "f32[384]", arg129_1: "f32[1, 1, 384]", arg130_1: "f32[1, 1, 384]", arg131_1: "f32[1536, 384]", arg132_1: "f32[1536]", arg133_1: "f32[384, 1536]", arg134_1: "f32[384]", arg135_1: "f32[384]", arg136_1: "f32[1, 1, 384]", arg137_1: "f32[1, 1, 384]", arg138_1: "f32[196, 196]", arg139_1: "f32[196]", arg140_1: "f32[384]", arg141_1: "f32[1, 1, 384]", arg142_1: "f32[1, 1, 384]", arg143_1: "f32[1536, 384]", arg144_1: "f32[1536]", arg145_1: "f32[384, 1536]", arg146_1: "f32[384]", arg147_1: "f32[1, 1, 384]", arg148_1: "f32[1, 1, 384]", arg149_1: "f32[1000, 384]", arg150_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_218: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg148_1, 1);  arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_212: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg142_1, 1);  arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_209: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg137_1, 1);  arg137_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_203: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg130_1, 1);  arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_200: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg125_1, 1);  arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_194: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg118_1, 1);  arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_191: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg113_1, 1);  arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_185: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg106_1, 1);  arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_182: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg101_1, 1);  arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_176: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg94_1, 1);  arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_173: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg89_1, 1);  arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_167: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg82_1, 1);  arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_164: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg77_1, 1);  arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_158: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg70_1, 1);  arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_155: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg65_1, 1);  arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_149: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg58_1, 1);  arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_146: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg53_1, 1);  arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_140: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg46_1, 1);  arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_137: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg41_1, 1);  arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_131: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg34_1, 1);  arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_128: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg29_1, 1);  arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_122: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg22_1, 1);  arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_119: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg17_1, 1);  arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_113: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg10_1, 1);  arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_110: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(arg5_1, 1);  arg5_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_1: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_73: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(convolution_1, [8, 384, 196]);  convolution_1 = None
        permute_62: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_111: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_110, permute_62);  mul_110 = None
        add_73: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg4_1, mul_111);  arg4_1 = mul_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_63: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_73, [0, 2, 1]);  add_73 = None
        view_74: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_63, [3072, 196]);  permute_63 = None
        permute_64: "f32[196, 196]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[3072, 196]" = torch.ops.aten.mm.default(view_74, permute_64);  view_74 = permute_64 = None
        add_tensor_23: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_23, arg7_1);  mm_default_23 = arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_75: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 384, 196]);  add_tensor_23 = None
        permute_65: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        mul_112: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg3_1, permute_65);  arg3_1 = permute_65 = None
        add_74: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(permute_62, mul_112);  permute_62 = mul_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_114: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_113, add_74);  mul_113 = None
        add_75: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg9_1, mul_114);  arg9_1 = mul_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_37: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_75, memory_format = torch.contiguous_format);  add_75 = None
        view_76: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_37, [1568, 384]);  clone_37 = None
        permute_66: "f32[384, 1536]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_12: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_76, permute_66);  view_76 = permute_66 = None
        view_77: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_12, [8, 196, 1536]);  mm_12 = None
        add_76: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_77, arg12_1);  view_77 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_115: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_76, 0.5)
        mul_116: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_76, 0.7071067811865476);  add_76 = None
        erf_12: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_77: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_117: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_115, add_77);  mul_115 = add_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_78: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_117, [1568, 1536]);  mul_117 = None
        permute_67: "f32[1536, 384]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[1568, 384]" = torch.ops.aten.mm.default(view_78, permute_67);  view_78 = permute_67 = None
        add_tensor_22: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_22, arg14_1);  mm_default_22 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_79: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 196, 384]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_118: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg8_1, view_79);  arg8_1 = view_79 = None
        add_78: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_74, mul_118);  add_74 = mul_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_120: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_119, add_78);  mul_119 = None
        add_79: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg16_1, mul_120);  arg16_1 = mul_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_68: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_79, [0, 2, 1]);  add_79 = None
        view_80: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_68, [3072, 196]);  permute_68 = None
        permute_69: "f32[196, 196]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[3072, 196]" = torch.ops.aten.mm.default(view_80, permute_69);  view_80 = permute_69 = None
        add_tensor_21: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_21, arg19_1);  mm_default_21 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_81: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 384, 196]);  add_tensor_21 = None
        permute_70: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
        mul_121: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg15_1, permute_70);  arg15_1 = permute_70 = None
        add_80: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_78, mul_121);  add_78 = mul_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_123: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_122, add_80);  mul_122 = None
        add_81: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg21_1, mul_123);  arg21_1 = mul_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_40: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_81, memory_format = torch.contiguous_format);  add_81 = None
        view_82: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_40, [1568, 384]);  clone_40 = None
        permute_71: "f32[384, 1536]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_13: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_82, permute_71);  view_82 = permute_71 = None
        view_83: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_13, [8, 196, 1536]);  mm_13 = None
        add_82: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_83, arg24_1);  view_83 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_124: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_82, 0.5)
        mul_125: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_82, 0.7071067811865476);  add_82 = None
        erf_13: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_83: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_126: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_124, add_83);  mul_124 = add_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_84: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_126, [1568, 1536]);  mul_126 = None
        permute_72: "f32[1536, 384]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[1568, 384]" = torch.ops.aten.mm.default(view_84, permute_72);  view_84 = permute_72 = None
        add_tensor_20: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_20, arg26_1);  mm_default_20 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_85: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 196, 384]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_127: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg20_1, view_85);  arg20_1 = view_85 = None
        add_84: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_80, mul_127);  add_80 = mul_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_129: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_128, add_84);  mul_128 = None
        add_85: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg28_1, mul_129);  arg28_1 = mul_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_73: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_85, [0, 2, 1]);  add_85 = None
        view_86: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_73, [3072, 196]);  permute_73 = None
        permute_74: "f32[196, 196]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[3072, 196]" = torch.ops.aten.mm.default(view_86, permute_74);  view_86 = permute_74 = None
        add_tensor_19: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_19, arg31_1);  mm_default_19 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_87: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 384, 196]);  add_tensor_19 = None
        permute_75: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        mul_130: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg27_1, permute_75);  arg27_1 = permute_75 = None
        add_86: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_84, mul_130);  add_84 = mul_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_132: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_131, add_86);  mul_131 = None
        add_87: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg33_1, mul_132);  arg33_1 = mul_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_43: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format);  add_87 = None
        view_88: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_43, [1568, 384]);  clone_43 = None
        permute_76: "f32[384, 1536]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_14: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_88, permute_76);  view_88 = permute_76 = None
        view_89: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_14, [8, 196, 1536]);  mm_14 = None
        add_88: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_89, arg36_1);  view_89 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_133: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_88, 0.5)
        mul_134: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_88, 0.7071067811865476);  add_88 = None
        erf_14: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
        add_89: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_135: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_133, add_89);  mul_133 = add_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_90: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_135, [1568, 1536]);  mul_135 = None
        permute_77: "f32[1536, 384]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[1568, 384]" = torch.ops.aten.mm.default(view_90, permute_77);  view_90 = permute_77 = None
        add_tensor_18: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_18, arg38_1);  mm_default_18 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_91: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 196, 384]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_136: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg32_1, view_91);  arg32_1 = view_91 = None
        add_90: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_86, mul_136);  add_86 = mul_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_138: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_137, add_90);  mul_137 = None
        add_91: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg40_1, mul_138);  arg40_1 = mul_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_78: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_91, [0, 2, 1]);  add_91 = None
        view_92: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_78, [3072, 196]);  permute_78 = None
        permute_79: "f32[196, 196]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[3072, 196]" = torch.ops.aten.mm.default(view_92, permute_79);  view_92 = permute_79 = None
        add_tensor_17: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_17, arg43_1);  mm_default_17 = arg43_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_93: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 384, 196]);  add_tensor_17 = None
        permute_80: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
        mul_139: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg39_1, permute_80);  arg39_1 = permute_80 = None
        add_92: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_90, mul_139);  add_90 = mul_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_141: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_140, add_92);  mul_140 = None
        add_93: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg45_1, mul_141);  arg45_1 = mul_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_46: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_93, memory_format = torch.contiguous_format);  add_93 = None
        view_94: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_46, [1568, 384]);  clone_46 = None
        permute_81: "f32[384, 1536]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_15: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_94, permute_81);  view_94 = permute_81 = None
        view_95: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_15, [8, 196, 1536]);  mm_15 = None
        add_94: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_95, arg48_1);  view_95 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_142: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_94, 0.5)
        mul_143: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_94, 0.7071067811865476);  add_94 = None
        erf_15: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
        add_95: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_144: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_142, add_95);  mul_142 = add_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_96: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_144, [1568, 1536]);  mul_144 = None
        permute_82: "f32[1536, 384]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[1568, 384]" = torch.ops.aten.mm.default(view_96, permute_82);  view_96 = permute_82 = None
        add_tensor_16: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_16, arg50_1);  mm_default_16 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_97: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 196, 384]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_145: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg44_1, view_97);  arg44_1 = view_97 = None
        add_96: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_92, mul_145);  add_92 = mul_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_147: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_146, add_96);  mul_146 = None
        add_97: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg52_1, mul_147);  arg52_1 = mul_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_83: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_97, [0, 2, 1]);  add_97 = None
        view_98: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_83, [3072, 196]);  permute_83 = None
        permute_84: "f32[196, 196]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[3072, 196]" = torch.ops.aten.mm.default(view_98, permute_84);  view_98 = permute_84 = None
        add_tensor_15: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_15, arg55_1);  mm_default_15 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_99: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 384, 196]);  add_tensor_15 = None
        permute_85: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        mul_148: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg51_1, permute_85);  arg51_1 = permute_85 = None
        add_98: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_96, mul_148);  add_96 = mul_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_150: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_149, add_98);  mul_149 = None
        add_99: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg57_1, mul_150);  arg57_1 = mul_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_49: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_99, memory_format = torch.contiguous_format);  add_99 = None
        view_100: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_49, [1568, 384]);  clone_49 = None
        permute_86: "f32[384, 1536]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_16: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_100, permute_86);  view_100 = permute_86 = None
        view_101: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_16, [8, 196, 1536]);  mm_16 = None
        add_100: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_101, arg60_1);  view_101 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_151: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_100, 0.5)
        mul_152: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_100, 0.7071067811865476);  add_100 = None
        erf_16: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_101: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_153: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_151, add_101);  mul_151 = add_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_102: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_153, [1568, 1536]);  mul_153 = None
        permute_87: "f32[1536, 384]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[1568, 384]" = torch.ops.aten.mm.default(view_102, permute_87);  view_102 = permute_87 = None
        add_tensor_14: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_14, arg62_1);  mm_default_14 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_103: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 196, 384]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_154: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg56_1, view_103);  arg56_1 = view_103 = None
        add_102: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_98, mul_154);  add_98 = mul_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_156: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_155, add_102);  mul_155 = None
        add_103: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg64_1, mul_156);  arg64_1 = mul_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_88: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_103, [0, 2, 1]);  add_103 = None
        view_104: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_88, [3072, 196]);  permute_88 = None
        permute_89: "f32[196, 196]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[3072, 196]" = torch.ops.aten.mm.default(view_104, permute_89);  view_104 = permute_89 = None
        add_tensor_13: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_13, arg67_1);  mm_default_13 = arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_105: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 384, 196]);  add_tensor_13 = None
        permute_90: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_105, [0, 2, 1]);  view_105 = None
        mul_157: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg63_1, permute_90);  arg63_1 = permute_90 = None
        add_104: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_102, mul_157);  add_102 = mul_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_159: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_158, add_104);  mul_158 = None
        add_105: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg69_1, mul_159);  arg69_1 = mul_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_52: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_105, memory_format = torch.contiguous_format);  add_105 = None
        view_106: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_52, [1568, 384]);  clone_52 = None
        permute_91: "f32[384, 1536]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_17: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_106, permute_91);  view_106 = permute_91 = None
        view_107: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_17, [8, 196, 1536]);  mm_17 = None
        add_106: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_107, arg72_1);  view_107 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_160: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_106, 0.5)
        mul_161: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_106, 0.7071067811865476);  add_106 = None
        erf_17: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_107: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_162: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_160, add_107);  mul_160 = add_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_108: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_162, [1568, 1536]);  mul_162 = None
        permute_92: "f32[1536, 384]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[1568, 384]" = torch.ops.aten.mm.default(view_108, permute_92);  view_108 = permute_92 = None
        add_tensor_12: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_12, arg74_1);  mm_default_12 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_109: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 196, 384]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_163: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg68_1, view_109);  arg68_1 = view_109 = None
        add_108: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_104, mul_163);  add_104 = mul_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_165: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_164, add_108);  mul_164 = None
        add_109: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg76_1, mul_165);  arg76_1 = mul_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_93: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_109, [0, 2, 1]);  add_109 = None
        view_110: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_93, [3072, 196]);  permute_93 = None
        permute_94: "f32[196, 196]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[3072, 196]" = torch.ops.aten.mm.default(view_110, permute_94);  view_110 = permute_94 = None
        add_tensor_11: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_11, arg79_1);  mm_default_11 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_111: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 384, 196]);  add_tensor_11 = None
        permute_95: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        mul_166: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg75_1, permute_95);  arg75_1 = permute_95 = None
        add_110: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_108, mul_166);  add_108 = mul_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_168: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_167, add_110);  mul_167 = None
        add_111: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg81_1, mul_168);  arg81_1 = mul_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_55: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format);  add_111 = None
        view_112: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_55, [1568, 384]);  clone_55 = None
        permute_96: "f32[384, 1536]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_18: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_112, permute_96);  view_112 = permute_96 = None
        view_113: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_18, [8, 196, 1536]);  mm_18 = None
        add_112: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_113, arg84_1);  view_113 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_169: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_112, 0.5)
        mul_170: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_112, 0.7071067811865476);  add_112 = None
        erf_18: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_170);  mul_170 = None
        add_113: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_171: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_169, add_113);  mul_169 = add_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_114: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_171, [1568, 1536]);  mul_171 = None
        permute_97: "f32[1536, 384]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[1568, 384]" = torch.ops.aten.mm.default(view_114, permute_97);  view_114 = permute_97 = None
        add_tensor_10: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_10, arg86_1);  mm_default_10 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_115: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 384]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_172: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg80_1, view_115);  arg80_1 = view_115 = None
        add_114: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_110, mul_172);  add_110 = mul_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_174: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_173, add_114);  mul_173 = None
        add_115: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg88_1, mul_174);  arg88_1 = mul_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_98: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_115, [0, 2, 1]);  add_115 = None
        view_116: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_98, [3072, 196]);  permute_98 = None
        permute_99: "f32[196, 196]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[3072, 196]" = torch.ops.aten.mm.default(view_116, permute_99);  view_116 = permute_99 = None
        add_tensor_9: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_9, arg91_1);  mm_default_9 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_117: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 384, 196]);  add_tensor_9 = None
        permute_100: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
        mul_175: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg87_1, permute_100);  arg87_1 = permute_100 = None
        add_116: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_114, mul_175);  add_114 = mul_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_177: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_176, add_116);  mul_176 = None
        add_117: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg93_1, mul_177);  arg93_1 = mul_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_58: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_117, memory_format = torch.contiguous_format);  add_117 = None
        view_118: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_58, [1568, 384]);  clone_58 = None
        permute_101: "f32[384, 1536]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_19: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_118, permute_101);  view_118 = permute_101 = None
        view_119: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_19, [8, 196, 1536]);  mm_19 = None
        add_118: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_119, arg96_1);  view_119 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_178: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_118, 0.5)
        mul_179: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_118, 0.7071067811865476);  add_118 = None
        erf_19: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
        add_119: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_180: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_178, add_119);  mul_178 = add_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_120: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_180, [1568, 1536]);  mul_180 = None
        permute_102: "f32[1536, 384]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[1568, 384]" = torch.ops.aten.mm.default(view_120, permute_102);  view_120 = permute_102 = None
        add_tensor_8: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_8, arg98_1);  mm_default_8 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_121: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 196, 384]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_181: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg92_1, view_121);  arg92_1 = view_121 = None
        add_120: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_116, mul_181);  add_116 = mul_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_183: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_182, add_120);  mul_182 = None
        add_121: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg100_1, mul_183);  arg100_1 = mul_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_103: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_121, [0, 2, 1]);  add_121 = None
        view_122: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_103, [3072, 196]);  permute_103 = None
        permute_104: "f32[196, 196]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[3072, 196]" = torch.ops.aten.mm.default(view_122, permute_104);  view_122 = permute_104 = None
        add_tensor_7: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_7, arg103_1);  mm_default_7 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_123: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 384, 196]);  add_tensor_7 = None
        permute_105: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        mul_184: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg99_1, permute_105);  arg99_1 = permute_105 = None
        add_122: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_120, mul_184);  add_120 = mul_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_186: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_185, add_122);  mul_185 = None
        add_123: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg105_1, mul_186);  arg105_1 = mul_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_61: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_123, memory_format = torch.contiguous_format);  add_123 = None
        view_124: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_61, [1568, 384]);  clone_61 = None
        permute_106: "f32[384, 1536]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_20: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_124, permute_106);  view_124 = permute_106 = None
        view_125: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_20, [8, 196, 1536]);  mm_20 = None
        add_124: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_125, arg108_1);  view_125 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_187: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_124, 0.5)
        mul_188: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_124, 0.7071067811865476);  add_124 = None
        erf_20: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
        add_125: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_189: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_187, add_125);  mul_187 = add_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_126: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_189, [1568, 1536]);  mul_189 = None
        permute_107: "f32[1536, 384]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[1568, 384]" = torch.ops.aten.mm.default(view_126, permute_107);  view_126 = permute_107 = None
        add_tensor_6: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_6, arg110_1);  mm_default_6 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_127: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 196, 384]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_190: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg104_1, view_127);  arg104_1 = view_127 = None
        add_126: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_122, mul_190);  add_122 = mul_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_192: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_191, add_126);  mul_191 = None
        add_127: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg112_1, mul_192);  arg112_1 = mul_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_108: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_127, [0, 2, 1]);  add_127 = None
        view_128: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_108, [3072, 196]);  permute_108 = None
        permute_109: "f32[196, 196]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[3072, 196]" = torch.ops.aten.mm.default(view_128, permute_109);  view_128 = permute_109 = None
        add_tensor_5: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_5, arg115_1);  mm_default_5 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_129: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 384, 196]);  add_tensor_5 = None
        permute_110: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
        mul_193: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg111_1, permute_110);  arg111_1 = permute_110 = None
        add_128: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_126, mul_193);  add_126 = mul_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_195: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_194, add_128);  mul_194 = None
        add_129: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg117_1, mul_195);  arg117_1 = mul_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_64: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format);  add_129 = None
        view_130: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_64, [1568, 384]);  clone_64 = None
        permute_111: "f32[384, 1536]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_21: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_130, permute_111);  view_130 = permute_111 = None
        view_131: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_21, [8, 196, 1536]);  mm_21 = None
        add_130: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_131, arg120_1);  view_131 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_196: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_130, 0.5)
        mul_197: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_130, 0.7071067811865476);  add_130 = None
        erf_21: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
        add_131: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_198: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_196, add_131);  mul_196 = add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_132: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_198, [1568, 1536]);  mul_198 = None
        permute_112: "f32[1536, 384]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[1568, 384]" = torch.ops.aten.mm.default(view_132, permute_112);  view_132 = permute_112 = None
        add_tensor_4: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_4, arg122_1);  mm_default_4 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_133: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 196, 384]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_199: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg116_1, view_133);  arg116_1 = view_133 = None
        add_132: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_128, mul_199);  add_128 = mul_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_201: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_200, add_132);  mul_200 = None
        add_133: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg124_1, mul_201);  arg124_1 = mul_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_113: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_133, [0, 2, 1]);  add_133 = None
        view_134: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_113, [3072, 196]);  permute_113 = None
        permute_114: "f32[196, 196]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[3072, 196]" = torch.ops.aten.mm.default(view_134, permute_114);  view_134 = permute_114 = None
        add_tensor_3: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_3, arg127_1);  mm_default_3 = arg127_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_135: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 384, 196]);  add_tensor_3 = None
        permute_115: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        mul_202: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg123_1, permute_115);  arg123_1 = permute_115 = None
        add_134: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_132, mul_202);  add_132 = mul_202 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_204: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_203, add_134);  mul_203 = None
        add_135: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg129_1, mul_204);  arg129_1 = mul_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_67: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_135, memory_format = torch.contiguous_format);  add_135 = None
        view_136: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_67, [1568, 384]);  clone_67 = None
        permute_116: "f32[384, 1536]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_22: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_136, permute_116);  view_136 = permute_116 = None
        view_137: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_22, [8, 196, 1536]);  mm_22 = None
        add_136: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_137, arg132_1);  view_137 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_205: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_136, 0.5)
        mul_206: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_136, 0.7071067811865476);  add_136 = None
        erf_22: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_206);  mul_206 = None
        add_137: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_207: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_205, add_137);  mul_205 = add_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_138: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_207, [1568, 1536]);  mul_207 = None
        permute_117: "f32[1536, 384]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[1568, 384]" = torch.ops.aten.mm.default(view_138, permute_117);  view_138 = permute_117 = None
        add_tensor_2: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_2, arg134_1);  mm_default_2 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_139: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 196, 384]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_208: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg128_1, view_139);  arg128_1 = view_139 = None
        add_138: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_134, mul_208);  add_134 = mul_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_210: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_209, add_138);  mul_209 = None
        add_139: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg136_1, mul_210);  arg136_1 = mul_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_118: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_139, [0, 2, 1]);  add_139 = None
        view_140: "f32[3072, 196]" = torch.ops.aten.reshape.default(permute_118, [3072, 196]);  permute_118 = None
        permute_119: "f32[196, 196]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[3072, 196]" = torch.ops.aten.mm.default(view_140, permute_119);  view_140 = permute_119 = None
        add_tensor_1: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_1, arg139_1);  mm_default_1 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:125 in forward, code: x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        view_141: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 384, 196]);  add_tensor_1 = None
        permute_120: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        mul_211: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg135_1, permute_120);  arg135_1 = permute_120 = None
        add_140: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_138, mul_211);  add_138 = mul_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_213: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_212, add_140);  mul_212 = None
        add_141: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg141_1, mul_213);  arg141_1 = mul_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        clone_70: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_141, memory_format = torch.contiguous_format);  add_141 = None
        view_142: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_70, [1568, 384]);  clone_70 = None
        permute_121: "f32[384, 1536]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_23: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_142, permute_121);  view_142 = permute_121 = None
        view_143: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(mm_23, [8, 196, 1536]);  mm_23 = None
        add_142: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(view_143, arg144_1);  view_143 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_214: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_142, 0.5)
        mul_215: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_142, 0.7071067811865476);  add_142 = None
        erf_23: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_215);  mul_215 = None
        add_143: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_216: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_214, add_143);  mul_214 = add_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_144: "f32[1568, 1536]" = torch.ops.aten.reshape.default(mul_216, [1568, 1536]);  mul_216 = None
        permute_122: "f32[1536, 384]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[1568, 384]" = torch.ops.aten.mm.default(view_144, permute_122);  view_144 = permute_122 = None
        add_tensor: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default, arg146_1);  mm_default = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_145: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor, [8, 196, 384]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:126 in forward, code: x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        mul_217: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(arg140_1, view_145);  arg140_1 = view_145 = None
        add_144: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_140, mul_217);  add_140 = mul_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:94 in forward, code: return torch.addcmul(self.beta, self.alpha, x)
        mul_219: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_218, add_144);  mul_218 = add_144 = None
        add_145: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(arg147_1, mul_219);  arg147_1 = mul_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:346 in forward_head, code: x = x.mean(dim=1)
        mean_1: "f32[8, 384]" = torch.ops.aten.mean.dim(add_145, [1]);  add_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:348 in forward_head, code: return x if pre_logits else self.head(x)
        permute_123: "f32[384, 1000]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_49: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg150_1, mean_1, permute_123);  arg150_1 = mean_1 = permute_123 = None
        return (addmm_49,)
        