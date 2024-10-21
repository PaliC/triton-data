class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[768, 3, 16, 16]", arg2_1: "f32[768]", arg3_1: "f32[1, 197, 768]", arg4_1: "f32[1, 1, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[2304, 768]", arg8_1: "f32[2304]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[3072, 768]", arg14_1: "f32[3072]", arg15_1: "f32[768, 3072]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[2304, 768]", arg20_1: "f32[2304]", arg21_1: "f32[768, 768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[3072, 768]", arg26_1: "f32[3072]", arg27_1: "f32[768, 3072]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[2304, 768]", arg32_1: "f32[2304]", arg33_1: "f32[768, 768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[3072, 768]", arg38_1: "f32[3072]", arg39_1: "f32[768, 3072]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[2304, 768]", arg44_1: "f32[2304]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[3072, 768]", arg50_1: "f32[3072]", arg51_1: "f32[768, 3072]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[2304, 768]", arg56_1: "f32[2304]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[3072, 768]", arg62_1: "f32[3072]", arg63_1: "f32[768, 3072]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[2304, 768]", arg68_1: "f32[2304]", arg69_1: "f32[768, 768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[3072, 768]", arg74_1: "f32[3072]", arg75_1: "f32[768, 3072]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[2304, 768]", arg80_1: "f32[2304]", arg81_1: "f32[768, 768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[3072, 768]", arg86_1: "f32[3072]", arg87_1: "f32[768, 3072]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[768]", arg91_1: "f32[2304, 768]", arg92_1: "f32[2304]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[3072, 768]", arg98_1: "f32[3072]", arg99_1: "f32[768, 3072]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[2304, 768]", arg104_1: "f32[2304]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[3072, 768]", arg110_1: "f32[3072]", arg111_1: "f32[768, 3072]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[2304, 768]", arg116_1: "f32[2304]", arg117_1: "f32[768, 768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[3072, 768]", arg122_1: "f32[3072]", arg123_1: "f32[768, 3072]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[2304, 768]", arg128_1: "f32[2304]", arg129_1: "f32[768, 768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[3072, 768]", arg134_1: "f32[3072]", arg135_1: "f32[768, 3072]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[2304, 768]", arg140_1: "f32[2304]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[3072, 768]", arg146_1: "f32[3072]", arg147_1: "f32[768, 3072]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[1000, 768]", arg152_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_1: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_121: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution_1, [8, 768, 196]);  convolution_1 = None
        permute_74: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:682 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        expand_1: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg4_1, [8, -1, -1]);  arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        cat_1: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand_1, permute_74], 1);  expand_1 = permute_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:697 in _pos_embed, code: x = x + pos_embed
        add_87: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(cat_1, arg3_1);  cat_1 = arg3_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_25 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_134: "f32[8, 197, 1]" = var_mean_25[0]
        getitem_135: "f32[8, 197, 1]" = var_mean_25[1];  var_mean_25 = None
        add_88: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
        rsqrt_25: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_25: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_135);  getitem_135 = None
        mul_86: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_87: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_86, arg5_1);  mul_86 = arg5_1 = None
        add_89: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_87, arg6_1);  mul_87 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_122: "f32[1576, 768]" = torch.ops.aten.view.default(add_89, [1576, 768]);  add_89 = None
        permute_75: "f32[768, 2304]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_49: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg8_1, view_122, permute_75);  arg8_1 = view_122 = permute_75 = None
        view_123: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_49, [8, 197, 2304]);  addmm_49 = None
        view_124: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_123, [8, 197, 3, 12, 64]);  view_123 = None
        permute_76: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_124, [2, 0, 3, 1, 4]);  view_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_12 = torch.ops.aten.unbind.int(permute_76);  permute_76 = None
        getitem_136: "f32[8, 12, 197, 64]" = unbind_12[0]
        getitem_137: "f32[8, 12, 197, 64]" = unbind_12[1]
        getitem_138: "f32[8, 12, 197, 64]" = unbind_12[2];  unbind_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_136, getitem_137, getitem_138, None, False);  getitem_136 = getitem_137 = getitem_138 = None
        getitem_139: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_77: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
        view_125: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_77, [8, 197, 768]);  permute_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_126: "f32[1576, 768]" = torch.ops.aten.view.default(view_125, [1576, 768]);  view_125 = None
        permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_50: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg10_1, view_126, permute_78);  arg10_1 = view_126 = permute_78 = None
        view_127: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_50, [8, 197, 768]);  addmm_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_90: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_87, view_127);  add_87 = view_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_26 = torch.ops.aten.var_mean.correction(add_90, [2], correction = 0, keepdim = True)
        getitem_143: "f32[8, 197, 1]" = var_mean_26[0]
        getitem_144: "f32[8, 197, 1]" = var_mean_26[1];  var_mean_26 = None
        add_91: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-06);  getitem_143 = None
        rsqrt_26: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_26: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_90, getitem_144);  getitem_144 = None
        mul_88: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        mul_89: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg11_1);  mul_88 = arg11_1 = None
        add_92: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_89, arg12_1);  mul_89 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_128: "f32[1576, 768]" = torch.ops.aten.view.default(add_92, [1576, 768]);  add_92 = None
        permute_79: "f32[768, 3072]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_51: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg14_1, view_128, permute_79);  arg14_1 = view_128 = permute_79 = None
        view_129: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_51, [8, 197, 3072]);  addmm_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_90: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_91: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_12: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_93: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_92: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_90, add_93);  mul_90 = add_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_130: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_92, [1576, 3072]);  mul_92 = None
        permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_52: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg16_1, view_130, permute_80);  arg16_1 = view_130 = permute_80 = None
        view_131: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_52, [8, 197, 768]);  addmm_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_94: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_90, view_131);  add_90 = view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_27 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_145: "f32[8, 197, 1]" = var_mean_27[0]
        getitem_146: "f32[8, 197, 1]" = var_mean_27[1];  var_mean_27 = None
        add_95: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-06);  getitem_145 = None
        rsqrt_27: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_27: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_94, getitem_146);  getitem_146 = None
        mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, arg17_1);  mul_93 = arg17_1 = None
        add_96: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_94, arg18_1);  mul_94 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_132: "f32[1576, 768]" = torch.ops.aten.view.default(add_96, [1576, 768]);  add_96 = None
        permute_81: "f32[768, 2304]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_53: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg20_1, view_132, permute_81);  arg20_1 = view_132 = permute_81 = None
        view_133: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_53, [8, 197, 2304]);  addmm_53 = None
        view_134: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_133, [8, 197, 3, 12, 64]);  view_133 = None
        permute_82: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_134, [2, 0, 3, 1, 4]);  view_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_82);  permute_82 = None
        getitem_147: "f32[8, 12, 197, 64]" = unbind_13[0]
        getitem_148: "f32[8, 12, 197, 64]" = unbind_13[1]
        getitem_149: "f32[8, 12, 197, 64]" = unbind_13[2];  unbind_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_147, getitem_148, getitem_149, None, False);  getitem_147 = getitem_148 = getitem_149 = None
        getitem_150: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_83: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        view_135: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_83, [8, 197, 768]);  permute_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_136: "f32[1576, 768]" = torch.ops.aten.view.default(view_135, [1576, 768]);  view_135 = None
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_54: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg22_1, view_136, permute_84);  arg22_1 = view_136 = permute_84 = None
        view_137: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_54, [8, 197, 768]);  addmm_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_97: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_94, view_137);  add_94 = view_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_28 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_154: "f32[8, 197, 1]" = var_mean_28[0]
        getitem_155: "f32[8, 197, 1]" = var_mean_28[1];  var_mean_28 = None
        add_98: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_28: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_28: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_155);  getitem_155 = None
        mul_95: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_96: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_95, arg23_1);  mul_95 = arg23_1 = None
        add_99: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_96, arg24_1);  mul_96 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_138: "f32[1576, 768]" = torch.ops.aten.view.default(add_99, [1576, 768]);  add_99 = None
        permute_85: "f32[768, 3072]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_55: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg26_1, view_138, permute_85);  arg26_1 = view_138 = permute_85 = None
        view_139: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_55, [8, 197, 3072]);  addmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_97: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
        mul_98: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_139, 0.7071067811865476);  view_139 = None
        erf_13: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_100: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_99: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_97, add_100);  mul_97 = add_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_140: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_99, [1576, 3072]);  mul_99 = None
        permute_86: "f32[3072, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_56: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg28_1, view_140, permute_86);  arg28_1 = view_140 = permute_86 = None
        view_141: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_56, [8, 197, 768]);  addmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_101: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_97, view_141);  add_97 = view_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_29 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_156: "f32[8, 197, 1]" = var_mean_29[0]
        getitem_157: "f32[8, 197, 1]" = var_mean_29[1];  var_mean_29 = None
        add_102: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
        rsqrt_29: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_29: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_101, getitem_157);  getitem_157 = None
        mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, arg29_1);  mul_100 = arg29_1 = None
        add_103: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_101, arg30_1);  mul_101 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_142: "f32[1576, 768]" = torch.ops.aten.view.default(add_103, [1576, 768]);  add_103 = None
        permute_87: "f32[768, 2304]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_57: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg32_1, view_142, permute_87);  arg32_1 = view_142 = permute_87 = None
        view_143: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_57, [8, 197, 2304]);  addmm_57 = None
        view_144: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_143, [8, 197, 3, 12, 64]);  view_143 = None
        permute_88: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_144, [2, 0, 3, 1, 4]);  view_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_88);  permute_88 = None
        getitem_158: "f32[8, 12, 197, 64]" = unbind_14[0]
        getitem_159: "f32[8, 12, 197, 64]" = unbind_14[1]
        getitem_160: "f32[8, 12, 197, 64]" = unbind_14[2];  unbind_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_158, getitem_159, getitem_160, None, False);  getitem_158 = getitem_159 = getitem_160 = None
        getitem_161: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_89: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None
        view_145: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_89, [8, 197, 768]);  permute_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_146: "f32[1576, 768]" = torch.ops.aten.view.default(view_145, [1576, 768]);  view_145 = None
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_58: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg34_1, view_146, permute_90);  arg34_1 = view_146 = permute_90 = None
        view_147: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_58, [8, 197, 768]);  addmm_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_104: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_101, view_147);  add_101 = view_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_30 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
        getitem_165: "f32[8, 197, 1]" = var_mean_30[0]
        getitem_166: "f32[8, 197, 1]" = var_mean_30[1];  var_mean_30 = None
        add_105: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_165, 1e-06);  getitem_165 = None
        rsqrt_30: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_30: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_104, getitem_166);  getitem_166 = None
        mul_102: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg35_1);  mul_102 = arg35_1 = None
        add_106: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_103, arg36_1);  mul_103 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_148: "f32[1576, 768]" = torch.ops.aten.view.default(add_106, [1576, 768]);  add_106 = None
        permute_91: "f32[768, 3072]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_59: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg38_1, view_148, permute_91);  arg38_1 = view_148 = permute_91 = None
        view_149: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_59, [8, 197, 3072]);  addmm_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_104: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
        mul_105: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
        erf_14: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_107: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_107);  mul_104 = add_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_150: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_106, [1576, 3072]);  mul_106 = None
        permute_92: "f32[3072, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_60: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg40_1, view_150, permute_92);  arg40_1 = view_150 = permute_92 = None
        view_151: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_60, [8, 197, 768]);  addmm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_108: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_104, view_151);  add_104 = view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_31 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_167: "f32[8, 197, 1]" = var_mean_31[0]
        getitem_168: "f32[8, 197, 1]" = var_mean_31[1];  var_mean_31 = None
        add_109: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
        rsqrt_31: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_31: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_108, getitem_168);  getitem_168 = None
        mul_107: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_107, arg41_1);  mul_107 = arg41_1 = None
        add_110: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_108, arg42_1);  mul_108 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_152: "f32[1576, 768]" = torch.ops.aten.view.default(add_110, [1576, 768]);  add_110 = None
        permute_93: "f32[768, 2304]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_61: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg44_1, view_152, permute_93);  arg44_1 = view_152 = permute_93 = None
        view_153: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_61, [8, 197, 2304]);  addmm_61 = None
        view_154: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_153, [8, 197, 3, 12, 64]);  view_153 = None
        permute_94: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_154, [2, 0, 3, 1, 4]);  view_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_94);  permute_94 = None
        getitem_169: "f32[8, 12, 197, 64]" = unbind_15[0]
        getitem_170: "f32[8, 12, 197, 64]" = unbind_15[1]
        getitem_171: "f32[8, 12, 197, 64]" = unbind_15[2];  unbind_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_169, getitem_170, getitem_171, None, False);  getitem_169 = getitem_170 = getitem_171 = None
        getitem_172: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_95: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_155: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_95, [8, 197, 768]);  permute_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_156: "f32[1576, 768]" = torch.ops.aten.view.default(view_155, [1576, 768]);  view_155 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_62: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg46_1, view_156, permute_96);  arg46_1 = view_156 = permute_96 = None
        view_157: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_62, [8, 197, 768]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_111: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_108, view_157);  add_108 = view_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_32 = torch.ops.aten.var_mean.correction(add_111, [2], correction = 0, keepdim = True)
        getitem_176: "f32[8, 197, 1]" = var_mean_32[0]
        getitem_177: "f32[8, 197, 1]" = var_mean_32[1];  var_mean_32 = None
        add_112: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_32: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_32: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_111, getitem_177);  getitem_177 = None
        mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_110: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_109, arg47_1);  mul_109 = arg47_1 = None
        add_113: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_110, arg48_1);  mul_110 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_158: "f32[1576, 768]" = torch.ops.aten.view.default(add_113, [1576, 768]);  add_113 = None
        permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_63: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg50_1, view_158, permute_97);  arg50_1 = view_158 = permute_97 = None
        view_159: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_63, [8, 197, 3072]);  addmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_111: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
        mul_112: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476);  view_159 = None
        erf_15: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_114: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_113: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_111, add_114);  mul_111 = add_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_160: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_113, [1576, 3072]);  mul_113 = None
        permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_64: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg52_1, view_160, permute_98);  arg52_1 = view_160 = permute_98 = None
        view_161: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_64, [8, 197, 768]);  addmm_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_115: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_111, view_161);  add_111 = view_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_33 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
        getitem_178: "f32[8, 197, 1]" = var_mean_33[0]
        getitem_179: "f32[8, 197, 1]" = var_mean_33[1];  var_mean_33 = None
        add_116: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_33: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_33: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_115, getitem_179);  getitem_179 = None
        mul_114: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_115: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_114, arg53_1);  mul_114 = arg53_1 = None
        add_117: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_115, arg54_1);  mul_115 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_162: "f32[1576, 768]" = torch.ops.aten.view.default(add_117, [1576, 768]);  add_117 = None
        permute_99: "f32[768, 2304]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_65: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg56_1, view_162, permute_99);  arg56_1 = view_162 = permute_99 = None
        view_163: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_65, [8, 197, 2304]);  addmm_65 = None
        view_164: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_163, [8, 197, 3, 12, 64]);  view_163 = None
        permute_100: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_164, [2, 0, 3, 1, 4]);  view_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_100);  permute_100 = None
        getitem_180: "f32[8, 12, 197, 64]" = unbind_16[0]
        getitem_181: "f32[8, 12, 197, 64]" = unbind_16[1]
        getitem_182: "f32[8, 12, 197, 64]" = unbind_16[2];  unbind_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_180, getitem_181, getitem_182, None, False);  getitem_180 = getitem_181 = getitem_182 = None
        getitem_183: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_101: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
        view_165: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_101, [8, 197, 768]);  permute_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_166: "f32[1576, 768]" = torch.ops.aten.view.default(view_165, [1576, 768]);  view_165 = None
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_66: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg58_1, view_166, permute_102);  arg58_1 = view_166 = permute_102 = None
        view_167: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_66, [8, 197, 768]);  addmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_118: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_115, view_167);  add_115 = view_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_34 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
        getitem_187: "f32[8, 197, 1]" = var_mean_34[0]
        getitem_188: "f32[8, 197, 1]" = var_mean_34[1];  var_mean_34 = None
        add_119: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_187, 1e-06);  getitem_187 = None
        rsqrt_34: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_34: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_118, getitem_188);  getitem_188 = None
        mul_116: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_117: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_116, arg59_1);  mul_116 = arg59_1 = None
        add_120: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_117, arg60_1);  mul_117 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_168: "f32[1576, 768]" = torch.ops.aten.view.default(add_120, [1576, 768]);  add_120 = None
        permute_103: "f32[768, 3072]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_67: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg62_1, view_168, permute_103);  arg62_1 = view_168 = permute_103 = None
        view_169: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_67, [8, 197, 3072]);  addmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_118: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_169, 0.5)
        mul_119: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_169, 0.7071067811865476);  view_169 = None
        erf_16: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_121: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_120: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_118, add_121);  mul_118 = add_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_170: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_120, [1576, 3072]);  mul_120 = None
        permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_68: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg64_1, view_170, permute_104);  arg64_1 = view_170 = permute_104 = None
        view_171: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_68, [8, 197, 768]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_122: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_118, view_171);  add_118 = view_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_35 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
        getitem_189: "f32[8, 197, 1]" = var_mean_35[0]
        getitem_190: "f32[8, 197, 1]" = var_mean_35[1];  var_mean_35 = None
        add_123: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_189, 1e-06);  getitem_189 = None
        rsqrt_35: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_35: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_122, getitem_190);  getitem_190 = None
        mul_121: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_122: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_121, arg65_1);  mul_121 = arg65_1 = None
        add_124: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_122, arg66_1);  mul_122 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_172: "f32[1576, 768]" = torch.ops.aten.view.default(add_124, [1576, 768]);  add_124 = None
        permute_105: "f32[768, 2304]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_69: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg68_1, view_172, permute_105);  arg68_1 = view_172 = permute_105 = None
        view_173: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_69, [8, 197, 2304]);  addmm_69 = None
        view_174: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_173, [8, 197, 3, 12, 64]);  view_173 = None
        permute_106: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_174, [2, 0, 3, 1, 4]);  view_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_106);  permute_106 = None
        getitem_191: "f32[8, 12, 197, 64]" = unbind_17[0]
        getitem_192: "f32[8, 12, 197, 64]" = unbind_17[1]
        getitem_193: "f32[8, 12, 197, 64]" = unbind_17[2];  unbind_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_191, getitem_192, getitem_193, None, False);  getitem_191 = getitem_192 = getitem_193 = None
        getitem_194: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_107: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
        view_175: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_107, [8, 197, 768]);  permute_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_176: "f32[1576, 768]" = torch.ops.aten.view.default(view_175, [1576, 768]);  view_175 = None
        permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_70: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg70_1, view_176, permute_108);  arg70_1 = view_176 = permute_108 = None
        view_177: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_70, [8, 197, 768]);  addmm_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_125: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_122, view_177);  add_122 = view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_36 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_198: "f32[8, 197, 1]" = var_mean_36[0]
        getitem_199: "f32[8, 197, 1]" = var_mean_36[1];  var_mean_36 = None
        add_126: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_36: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_36: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_125, getitem_199);  getitem_199 = None
        mul_123: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_124: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_123, arg71_1);  mul_123 = arg71_1 = None
        add_127: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_124, arg72_1);  mul_124 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_178: "f32[1576, 768]" = torch.ops.aten.view.default(add_127, [1576, 768]);  add_127 = None
        permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_71: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg74_1, view_178, permute_109);  arg74_1 = view_178 = permute_109 = None
        view_179: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_71, [8, 197, 3072]);  addmm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_125: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.5)
        mul_126: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.7071067811865476);  view_179 = None
        erf_17: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_128: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_127: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_125, add_128);  mul_125 = add_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_180: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_127, [1576, 3072]);  mul_127 = None
        permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_72: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg76_1, view_180, permute_110);  arg76_1 = view_180 = permute_110 = None
        view_181: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_72, [8, 197, 768]);  addmm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_129: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_125, view_181);  add_125 = view_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_37 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_200: "f32[8, 197, 1]" = var_mean_37[0]
        getitem_201: "f32[8, 197, 1]" = var_mean_37[1];  var_mean_37 = None
        add_130: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_37: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_37: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_129, getitem_201);  getitem_201 = None
        mul_128: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_129: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_128, arg77_1);  mul_128 = arg77_1 = None
        add_131: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_129, arg78_1);  mul_129 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_182: "f32[1576, 768]" = torch.ops.aten.view.default(add_131, [1576, 768]);  add_131 = None
        permute_111: "f32[768, 2304]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_73: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg80_1, view_182, permute_111);  arg80_1 = view_182 = permute_111 = None
        view_183: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_73, [8, 197, 2304]);  addmm_73 = None
        view_184: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_183, [8, 197, 3, 12, 64]);  view_183 = None
        permute_112: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_184, [2, 0, 3, 1, 4]);  view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_18 = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
        getitem_202: "f32[8, 12, 197, 64]" = unbind_18[0]
        getitem_203: "f32[8, 12, 197, 64]" = unbind_18[1]
        getitem_204: "f32[8, 12, 197, 64]" = unbind_18[2];  unbind_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_202, getitem_203, getitem_204, None, False);  getitem_202 = getitem_203 = getitem_204 = None
        getitem_205: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_113: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
        view_185: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_113, [8, 197, 768]);  permute_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_186: "f32[1576, 768]" = torch.ops.aten.view.default(view_185, [1576, 768]);  view_185 = None
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_74: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg82_1, view_186, permute_114);  arg82_1 = view_186 = permute_114 = None
        view_187: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_74, [8, 197, 768]);  addmm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_132: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_129, view_187);  add_129 = view_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_38 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
        getitem_209: "f32[8, 197, 1]" = var_mean_38[0]
        getitem_210: "f32[8, 197, 1]" = var_mean_38[1];  var_mean_38 = None
        add_133: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_209, 1e-06);  getitem_209 = None
        rsqrt_38: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_38: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_132, getitem_210);  getitem_210 = None
        mul_130: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_131: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_130, arg83_1);  mul_130 = arg83_1 = None
        add_134: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_131, arg84_1);  mul_131 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_188: "f32[1576, 768]" = torch.ops.aten.view.default(add_134, [1576, 768]);  add_134 = None
        permute_115: "f32[768, 3072]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_75: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg86_1, view_188, permute_115);  arg86_1 = view_188 = permute_115 = None
        view_189: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_75, [8, 197, 3072]);  addmm_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_132: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_133: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_18: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_135: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_134: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_132, add_135);  mul_132 = add_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_190: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_134, [1576, 3072]);  mul_134 = None
        permute_116: "f32[3072, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_76: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg88_1, view_190, permute_116);  arg88_1 = view_190 = permute_116 = None
        view_191: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_76, [8, 197, 768]);  addmm_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_136: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_132, view_191);  add_132 = view_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_39 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_211: "f32[8, 197, 1]" = var_mean_39[0]
        getitem_212: "f32[8, 197, 1]" = var_mean_39[1];  var_mean_39 = None
        add_137: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_211, 1e-06);  getitem_211 = None
        rsqrt_39: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_39: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_136, getitem_212);  getitem_212 = None
        mul_135: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_136: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_135, arg89_1);  mul_135 = arg89_1 = None
        add_138: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_136, arg90_1);  mul_136 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_192: "f32[1576, 768]" = torch.ops.aten.view.default(add_138, [1576, 768]);  add_138 = None
        permute_117: "f32[768, 2304]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_77: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg92_1, view_192, permute_117);  arg92_1 = view_192 = permute_117 = None
        view_193: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_77, [8, 197, 2304]);  addmm_77 = None
        view_194: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_193, [8, 197, 3, 12, 64]);  view_193 = None
        permute_118: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_194, [2, 0, 3, 1, 4]);  view_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_19 = torch.ops.aten.unbind.int(permute_118);  permute_118 = None
        getitem_213: "f32[8, 12, 197, 64]" = unbind_19[0]
        getitem_214: "f32[8, 12, 197, 64]" = unbind_19[1]
        getitem_215: "f32[8, 12, 197, 64]" = unbind_19[2];  unbind_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_213, getitem_214, getitem_215, None, False);  getitem_213 = getitem_214 = getitem_215 = None
        getitem_216: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_119: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
        view_195: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_119, [8, 197, 768]);  permute_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_196: "f32[1576, 768]" = torch.ops.aten.view.default(view_195, [1576, 768]);  view_195 = None
        permute_120: "f32[768, 768]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_78: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg94_1, view_196, permute_120);  arg94_1 = view_196 = permute_120 = None
        view_197: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_78, [8, 197, 768]);  addmm_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_139: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_136, view_197);  add_136 = view_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_40 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
        getitem_220: "f32[8, 197, 1]" = var_mean_40[0]
        getitem_221: "f32[8, 197, 1]" = var_mean_40[1];  var_mean_40 = None
        add_140: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
        rsqrt_40: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_40: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_139, getitem_221);  getitem_221 = None
        mul_137: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_138: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_137, arg95_1);  mul_137 = arg95_1 = None
        add_141: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_138, arg96_1);  mul_138 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_198: "f32[1576, 768]" = torch.ops.aten.view.default(add_141, [1576, 768]);  add_141 = None
        permute_121: "f32[768, 3072]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_79: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg98_1, view_198, permute_121);  arg98_1 = view_198 = permute_121 = None
        view_199: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_79, [8, 197, 3072]);  addmm_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_139: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.5)
        mul_140: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476);  view_199 = None
        erf_19: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_142: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_141: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_139, add_142);  mul_139 = add_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_200: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_141, [1576, 3072]);  mul_141 = None
        permute_122: "f32[3072, 768]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_80: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg100_1, view_200, permute_122);  arg100_1 = view_200 = permute_122 = None
        view_201: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_80, [8, 197, 768]);  addmm_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_143: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_139, view_201);  add_139 = view_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_41 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_222: "f32[8, 197, 1]" = var_mean_41[0]
        getitem_223: "f32[8, 197, 1]" = var_mean_41[1];  var_mean_41 = None
        add_144: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_41: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_41: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_143, getitem_223);  getitem_223 = None
        mul_142: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_143: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_142, arg101_1);  mul_142 = arg101_1 = None
        add_145: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_143, arg102_1);  mul_143 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_202: "f32[1576, 768]" = torch.ops.aten.view.default(add_145, [1576, 768]);  add_145 = None
        permute_123: "f32[768, 2304]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_81: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg104_1, view_202, permute_123);  arg104_1 = view_202 = permute_123 = None
        view_203: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_81, [8, 197, 2304]);  addmm_81 = None
        view_204: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_203, [8, 197, 3, 12, 64]);  view_203 = None
        permute_124: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_204, [2, 0, 3, 1, 4]);  view_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_20 = torch.ops.aten.unbind.int(permute_124);  permute_124 = None
        getitem_224: "f32[8, 12, 197, 64]" = unbind_20[0]
        getitem_225: "f32[8, 12, 197, 64]" = unbind_20[1]
        getitem_226: "f32[8, 12, 197, 64]" = unbind_20[2];  unbind_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_224, getitem_225, getitem_226, None, False);  getitem_224 = getitem_225 = getitem_226 = None
        getitem_227: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_125: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_227, [0, 2, 1, 3]);  getitem_227 = None
        view_205: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_125, [8, 197, 768]);  permute_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_206: "f32[1576, 768]" = torch.ops.aten.view.default(view_205, [1576, 768]);  view_205 = None
        permute_126: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_82: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg106_1, view_206, permute_126);  arg106_1 = view_206 = permute_126 = None
        view_207: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_82, [8, 197, 768]);  addmm_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_146: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_143, view_207);  add_143 = view_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_42 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
        getitem_231: "f32[8, 197, 1]" = var_mean_42[0]
        getitem_232: "f32[8, 197, 1]" = var_mean_42[1];  var_mean_42 = None
        add_147: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_231, 1e-06);  getitem_231 = None
        rsqrt_42: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        sub_42: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_146, getitem_232);  getitem_232 = None
        mul_144: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_145: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_144, arg107_1);  mul_144 = arg107_1 = None
        add_148: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_145, arg108_1);  mul_145 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_208: "f32[1576, 768]" = torch.ops.aten.view.default(add_148, [1576, 768]);  add_148 = None
        permute_127: "f32[768, 3072]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_83: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg110_1, view_208, permute_127);  arg110_1 = view_208 = permute_127 = None
        view_209: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_83, [8, 197, 3072]);  addmm_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_146: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_209, 0.5)
        mul_147: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_209, 0.7071067811865476);  view_209 = None
        erf_20: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
        add_149: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_148: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_146, add_149);  mul_146 = add_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_210: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_148, [1576, 3072]);  mul_148 = None
        permute_128: "f32[3072, 768]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_84: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg112_1, view_210, permute_128);  arg112_1 = view_210 = permute_128 = None
        view_211: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_84, [8, 197, 768]);  addmm_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_150: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_146, view_211);  add_146 = view_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_43 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
        getitem_233: "f32[8, 197, 1]" = var_mean_43[0]
        getitem_234: "f32[8, 197, 1]" = var_mean_43[1];  var_mean_43 = None
        add_151: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_233, 1e-06);  getitem_233 = None
        rsqrt_43: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_43: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_150, getitem_234);  getitem_234 = None
        mul_149: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_150: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_149, arg113_1);  mul_149 = arg113_1 = None
        add_152: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_150, arg114_1);  mul_150 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_212: "f32[1576, 768]" = torch.ops.aten.view.default(add_152, [1576, 768]);  add_152 = None
        permute_129: "f32[768, 2304]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_85: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg116_1, view_212, permute_129);  arg116_1 = view_212 = permute_129 = None
        view_213: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_85, [8, 197, 2304]);  addmm_85 = None
        view_214: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_213, [8, 197, 3, 12, 64]);  view_213 = None
        permute_130: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_214, [2, 0, 3, 1, 4]);  view_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_21 = torch.ops.aten.unbind.int(permute_130);  permute_130 = None
        getitem_235: "f32[8, 12, 197, 64]" = unbind_21[0]
        getitem_236: "f32[8, 12, 197, 64]" = unbind_21[1]
        getitem_237: "f32[8, 12, 197, 64]" = unbind_21[2];  unbind_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_235, getitem_236, getitem_237, None, False);  getitem_235 = getitem_236 = getitem_237 = None
        getitem_238: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_131: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None
        view_215: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_131, [8, 197, 768]);  permute_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_216: "f32[1576, 768]" = torch.ops.aten.view.default(view_215, [1576, 768]);  view_215 = None
        permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_86: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg118_1, view_216, permute_132);  arg118_1 = view_216 = permute_132 = None
        view_217: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_86, [8, 197, 768]);  addmm_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_153: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_150, view_217);  add_150 = view_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_44 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_242: "f32[8, 197, 1]" = var_mean_44[0]
        getitem_243: "f32[8, 197, 1]" = var_mean_44[1];  var_mean_44 = None
        add_154: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_44: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_44: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_153, getitem_243);  getitem_243 = None
        mul_151: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_152: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_151, arg119_1);  mul_151 = arg119_1 = None
        add_155: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_152, arg120_1);  mul_152 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_218: "f32[1576, 768]" = torch.ops.aten.view.default(add_155, [1576, 768]);  add_155 = None
        permute_133: "f32[768, 3072]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_87: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg122_1, view_218, permute_133);  arg122_1 = view_218 = permute_133 = None
        view_219: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_87, [8, 197, 3072]);  addmm_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_153: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_154: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_21: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_156: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_155: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_153, add_156);  mul_153 = add_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_220: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_155, [1576, 3072]);  mul_155 = None
        permute_134: "f32[3072, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_88: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg124_1, view_220, permute_134);  arg124_1 = view_220 = permute_134 = None
        view_221: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_88, [8, 197, 768]);  addmm_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_157: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_153, view_221);  add_153 = view_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_45 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_244: "f32[8, 197, 1]" = var_mean_45[0]
        getitem_245: "f32[8, 197, 1]" = var_mean_45[1];  var_mean_45 = None
        add_158: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
        rsqrt_45: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_45: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_157, getitem_245);  getitem_245 = None
        mul_156: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_157: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_156, arg125_1);  mul_156 = arg125_1 = None
        add_159: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_157, arg126_1);  mul_157 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_222: "f32[1576, 768]" = torch.ops.aten.view.default(add_159, [1576, 768]);  add_159 = None
        permute_135: "f32[768, 2304]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_89: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg128_1, view_222, permute_135);  arg128_1 = view_222 = permute_135 = None
        view_223: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_89, [8, 197, 2304]);  addmm_89 = None
        view_224: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_223, [8, 197, 3, 12, 64]);  view_223 = None
        permute_136: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_224, [2, 0, 3, 1, 4]);  view_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_22 = torch.ops.aten.unbind.int(permute_136);  permute_136 = None
        getitem_246: "f32[8, 12, 197, 64]" = unbind_22[0]
        getitem_247: "f32[8, 12, 197, 64]" = unbind_22[1]
        getitem_248: "f32[8, 12, 197, 64]" = unbind_22[2];  unbind_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_246, getitem_247, getitem_248, None, False);  getitem_246 = getitem_247 = getitem_248 = None
        getitem_249: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_137: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_249, [0, 2, 1, 3]);  getitem_249 = None
        view_225: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_137, [8, 197, 768]);  permute_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_226: "f32[1576, 768]" = torch.ops.aten.view.default(view_225, [1576, 768]);  view_225 = None
        permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_90: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg130_1, view_226, permute_138);  arg130_1 = view_226 = permute_138 = None
        view_227: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_90, [8, 197, 768]);  addmm_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_160: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_157, view_227);  add_157 = view_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_46 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
        getitem_253: "f32[8, 197, 1]" = var_mean_46[0]
        getitem_254: "f32[8, 197, 1]" = var_mean_46[1];  var_mean_46 = None
        add_161: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_253, 1e-06);  getitem_253 = None
        rsqrt_46: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_46: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_160, getitem_254);  getitem_254 = None
        mul_158: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_159: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_158, arg131_1);  mul_158 = arg131_1 = None
        add_162: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_159, arg132_1);  mul_159 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_228: "f32[1576, 768]" = torch.ops.aten.view.default(add_162, [1576, 768]);  add_162 = None
        permute_139: "f32[768, 3072]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_91: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg134_1, view_228, permute_139);  arg134_1 = view_228 = permute_139 = None
        view_229: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_91, [8, 197, 3072]);  addmm_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_160: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_229, 0.5)
        mul_161: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476);  view_229 = None
        erf_22: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_163: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_162: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_160, add_163);  mul_160 = add_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_230: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_162, [1576, 3072]);  mul_162 = None
        permute_140: "f32[3072, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_92: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg136_1, view_230, permute_140);  arg136_1 = view_230 = permute_140 = None
        view_231: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_92, [8, 197, 768]);  addmm_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_164: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_160, view_231);  add_160 = view_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_47 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_255: "f32[8, 197, 1]" = var_mean_47[0]
        getitem_256: "f32[8, 197, 1]" = var_mean_47[1];  var_mean_47 = None
        add_165: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_255, 1e-06);  getitem_255 = None
        rsqrt_47: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_47: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_164, getitem_256);  getitem_256 = None
        mul_163: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_164: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_163, arg137_1);  mul_163 = arg137_1 = None
        add_166: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_164, arg138_1);  mul_164 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_232: "f32[1576, 768]" = torch.ops.aten.view.default(add_166, [1576, 768]);  add_166 = None
        permute_141: "f32[768, 2304]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_93: "f32[1576, 2304]" = torch.ops.aten.addmm.default(arg140_1, view_232, permute_141);  arg140_1 = view_232 = permute_141 = None
        view_233: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_93, [8, 197, 2304]);  addmm_93 = None
        view_234: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_233, [8, 197, 3, 12, 64]);  view_233 = None
        permute_142: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_234, [2, 0, 3, 1, 4]);  view_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_23 = torch.ops.aten.unbind.int(permute_142);  permute_142 = None
        getitem_257: "f32[8, 12, 197, 64]" = unbind_23[0]
        getitem_258: "f32[8, 12, 197, 64]" = unbind_23[1]
        getitem_259: "f32[8, 12, 197, 64]" = unbind_23[2];  unbind_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_257, getitem_258, getitem_259, None, False);  getitem_257 = getitem_258 = getitem_259 = None
        getitem_260: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_143: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        view_235: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_143, [8, 197, 768]);  permute_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_236: "f32[1576, 768]" = torch.ops.aten.view.default(view_235, [1576, 768]);  view_235 = None
        permute_144: "f32[768, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_94: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg142_1, view_236, permute_144);  arg142_1 = view_236 = permute_144 = None
        view_237: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_94, [8, 197, 768]);  addmm_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_167: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_164, view_237);  add_164 = view_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_48 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
        getitem_264: "f32[8, 197, 1]" = var_mean_48[0]
        getitem_265: "f32[8, 197, 1]" = var_mean_48[1];  var_mean_48 = None
        add_168: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-06);  getitem_264 = None
        rsqrt_48: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        sub_48: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_167, getitem_265);  getitem_265 = None
        mul_165: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_166: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_165, arg143_1);  mul_165 = arg143_1 = None
        add_169: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_166, arg144_1);  mul_166 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_238: "f32[1576, 768]" = torch.ops.aten.view.default(add_169, [1576, 768]);  add_169 = None
        permute_145: "f32[768, 3072]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_95: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg146_1, view_238, permute_145);  arg146_1 = view_238 = permute_145 = None
        view_239: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_95, [8, 197, 3072]);  addmm_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_167: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_168: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_23: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_170: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_169: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_167, add_170);  mul_167 = add_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_240: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_169, [1576, 3072]);  mul_169 = None
        permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_96: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg148_1, view_240, permute_146);  arg148_1 = view_240 = permute_146 = None
        view_241: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_96, [8, 197, 768]);  addmm_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_171: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_167, view_241);  add_167 = view_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:811 in forward_features, code: x = self.norm(x)
        var_mean_49 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
        getitem_266: "f32[8, 197, 1]" = var_mean_49[0]
        getitem_267: "f32[8, 197, 1]" = var_mean_49[1];  var_mean_49 = None
        add_172: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_49: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_49: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_171, getitem_267);  add_171 = getitem_267 = None
        mul_170: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_171: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_170, arg149_1);  mul_170 = arg149_1 = None
        add_173: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_171, arg150_1);  mul_171 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:399 in global_pool_nlc, code: x = x[:, 0]  # class token
        select_1: "f32[8, 768]" = torch.ops.aten.select.int(add_173, 1, 0);  add_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:825 in forward_head, code: x = self.head_drop(x)
        clone_75: "f32[8, 768]" = torch.ops.aten.clone.default(select_1);  select_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:826 in forward_head, code: return x if pre_logits else self.head(x)
        permute_147: "f32[768, 1000]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_97: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg152_1, clone_75, permute_147);  arg152_1 = clone_75 = permute_147 = None
        return (addmm_97,)
        