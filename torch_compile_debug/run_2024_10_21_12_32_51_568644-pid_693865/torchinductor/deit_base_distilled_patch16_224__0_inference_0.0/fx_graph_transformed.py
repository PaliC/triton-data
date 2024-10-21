class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[768, 3, 16, 16]", arg2_1: "f32[768]", arg3_1: "f32[1, 198, 768]", arg4_1: "f32[1, 1, 768]", arg5_1: "f32[1, 1, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[2304, 768]", arg9_1: "f32[2304]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[3072]", arg16_1: "f32[768, 3072]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[2304, 768]", arg21_1: "f32[2304]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768]", arg26_1: "f32[3072, 768]", arg27_1: "f32[3072]", arg28_1: "f32[768, 3072]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[2304, 768]", arg33_1: "f32[2304]", arg34_1: "f32[768, 768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[3072, 768]", arg39_1: "f32[3072]", arg40_1: "f32[768, 3072]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[2304, 768]", arg45_1: "f32[2304]", arg46_1: "f32[768, 768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[3072, 768]", arg51_1: "f32[3072]", arg52_1: "f32[768, 3072]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[2304, 768]", arg57_1: "f32[2304]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[3072]", arg64_1: "f32[768, 3072]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[2304, 768]", arg69_1: "f32[2304]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768]", arg74_1: "f32[3072, 768]", arg75_1: "f32[3072]", arg76_1: "f32[768, 3072]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[2304, 768]", arg81_1: "f32[2304]", arg82_1: "f32[768, 768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[3072, 768]", arg87_1: "f32[3072]", arg88_1: "f32[768, 3072]", arg89_1: "f32[768]", arg90_1: "f32[768]", arg91_1: "f32[768]", arg92_1: "f32[2304, 768]", arg93_1: "f32[2304]", arg94_1: "f32[768, 768]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[3072, 768]", arg99_1: "f32[3072]", arg100_1: "f32[768, 3072]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[2304, 768]", arg105_1: "f32[2304]", arg106_1: "f32[768, 768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[768]", arg110_1: "f32[3072, 768]", arg111_1: "f32[3072]", arg112_1: "f32[768, 3072]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[2304, 768]", arg117_1: "f32[2304]", arg118_1: "f32[768, 768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[3072, 768]", arg123_1: "f32[3072]", arg124_1: "f32[768, 3072]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[2304, 768]", arg129_1: "f32[2304]", arg130_1: "f32[768, 768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[768]", arg134_1: "f32[3072, 768]", arg135_1: "f32[3072]", arg136_1: "f32[768, 3072]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[2304, 768]", arg141_1: "f32[2304]", arg142_1: "f32[768, 768]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[3072, 768]", arg147_1: "f32[3072]", arg148_1: "f32[768, 3072]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[1000, 768]", arg153_1: "f32[1000]", arg154_1: "f32[1000, 768]", arg155_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:99 in _pos_embed, code: self.cls_token.expand(x.shape[0], -1, -1),
        expand_2: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg4_1, [8, -1, -1]);  arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:100 in _pos_embed, code: self.dist_token.expand(x.shape[0], -1, -1),
        expand_3: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg5_1, [8, -1, -1]);  arg5_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_1: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_121: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(convolution_1, [8, 768, 196]);  convolution_1 = None
        permute_75: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:98 in _pos_embed, code: x = torch.cat((
        cat_1: "f32[8, 198, 768]" = torch.ops.aten.cat.default([expand_2, expand_3, permute_75], 1);  expand_2 = expand_3 = permute_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:103 in _pos_embed, code: x = x + pos_embed
        add_88: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(cat_1, arg3_1);  cat_1 = arg3_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_25 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_134: "f32[8, 198, 1]" = var_mean_25[0]
        getitem_135: "f32[8, 198, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_25: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_135);  getitem_135 = None
        add_89: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
        rsqrt_25: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_86: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_87: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_86, arg6_1);  mul_86 = arg6_1 = None
        add_90: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_87, arg7_1);  mul_87 = arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_122: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_90, [1584, 768]);  add_90 = None
        permute_76: "f32[768, 2304]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_50: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg9_1, view_122, permute_76);  arg9_1 = view_122 = permute_76 = None
        view_123: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_50, [8, 198, 2304]);  addmm_50 = None
        view_124: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_123, [8, 198, 3, 12, 64]);  view_123 = None
        permute_77: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_124, [2, 0, 3, 1, 4]);  view_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_12 = torch.ops.aten.unbind.int(permute_77);  permute_77 = None
        getitem_136: "f32[8, 12, 198, 64]" = unbind_12[0]
        getitem_137: "f32[8, 12, 198, 64]" = unbind_12[1]
        getitem_138: "f32[8, 12, 198, 64]" = unbind_12[2];  unbind_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_136, getitem_137, getitem_138, None, False);  getitem_136 = getitem_137 = getitem_138 = None
        getitem_139: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_78: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
        view_125: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_78, [8, 198, 768]);  permute_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_126: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_125, [1584, 768]);  view_125 = None
        permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[1584, 768]" = torch.ops.aten.mm.default(view_126, permute_79);  view_126 = permute_79 = None
        add_tensor_37: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_37, arg11_1);  mm_default_37 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_127: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 198, 768]);  add_tensor_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_91: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_88, view_127);  add_88 = view_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_26 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_143: "f32[8, 198, 1]" = var_mean_26[0]
        getitem_144: "f32[8, 198, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_26: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_144);  getitem_144 = None
        add_92: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-06);  getitem_143 = None
        rsqrt_26: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_88: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        mul_89: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg12_1);  mul_88 = arg12_1 = None
        add_93: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_89, arg13_1);  mul_89 = arg13_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_128: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_93, [1584, 768]);  add_93 = None
        permute_80: "f32[768, 3072]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_128, permute_80);  view_128 = permute_80 = None
        add_tensor_36: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_36, arg15_1);  mm_default_36 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_129: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 198, 3072]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_90: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_91: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_12: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_94: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_92: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_90, add_94);  mul_90 = add_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_130: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_92, [1584, 3072]);  mul_92 = None
        permute_81: "f32[3072, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[1584, 768]" = torch.ops.aten.mm.default(view_130, permute_81);  view_130 = permute_81 = None
        add_tensor_35: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_35, arg17_1);  mm_default_35 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_131: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 198, 768]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_95: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_91, view_131);  add_91 = view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_27 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_145: "f32[8, 198, 1]" = var_mean_27[0]
        getitem_146: "f32[8, 198, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_27: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_146);  getitem_146 = None
        add_96: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-06);  getitem_145 = None
        rsqrt_27: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_93: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_94: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_93, arg18_1);  mul_93 = arg18_1 = None
        add_97: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_94, arg19_1);  mul_94 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_132: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_97, [1584, 768]);  add_97 = None
        permute_82: "f32[768, 2304]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_54: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg21_1, view_132, permute_82);  arg21_1 = view_132 = permute_82 = None
        view_133: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_54, [8, 198, 2304]);  addmm_54 = None
        view_134: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_133, [8, 198, 3, 12, 64]);  view_133 = None
        permute_83: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_134, [2, 0, 3, 1, 4]);  view_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_83);  permute_83 = None
        getitem_147: "f32[8, 12, 198, 64]" = unbind_13[0]
        getitem_148: "f32[8, 12, 198, 64]" = unbind_13[1]
        getitem_149: "f32[8, 12, 198, 64]" = unbind_13[2];  unbind_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_147, getitem_148, getitem_149, None, False);  getitem_147 = getitem_148 = getitem_149 = None
        getitem_150: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_84: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        view_135: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_84, [8, 198, 768]);  permute_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_136: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_135, [1584, 768]);  view_135 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[1584, 768]" = torch.ops.aten.mm.default(view_136, permute_85);  view_136 = permute_85 = None
        add_tensor_34: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_34, arg23_1);  mm_default_34 = arg23_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_137: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 198, 768]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_98: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_95, view_137);  add_95 = view_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_28 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_154: "f32[8, 198, 1]" = var_mean_28[0]
        getitem_155: "f32[8, 198, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_28: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_98, getitem_155);  getitem_155 = None
        add_99: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_28: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_95: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_96: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_95, arg24_1);  mul_95 = arg24_1 = None
        add_100: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_96, arg25_1);  mul_96 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_138: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_100, [1584, 768]);  add_100 = None
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_138, permute_86);  view_138 = permute_86 = None
        add_tensor_33: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_33, arg27_1);  mm_default_33 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_139: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 198, 3072]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_97: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
        mul_98: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_139, 0.7071067811865476);  view_139 = None
        erf_13: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_101: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_99: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_97, add_101);  mul_97 = add_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_140: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_99, [1584, 3072]);  mul_99 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[1584, 768]" = torch.ops.aten.mm.default(view_140, permute_87);  view_140 = permute_87 = None
        add_tensor_32: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_32, arg29_1);  mm_default_32 = arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_141: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 198, 768]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_102: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_98, view_141);  add_98 = view_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_29 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
        getitem_156: "f32[8, 198, 1]" = var_mean_29[0]
        getitem_157: "f32[8, 198, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_29: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_102, getitem_157);  getitem_157 = None
        add_103: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
        rsqrt_29: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_100: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_101: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_100, arg30_1);  mul_100 = arg30_1 = None
        add_104: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_101, arg31_1);  mul_101 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_142: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_104, [1584, 768]);  add_104 = None
        permute_88: "f32[768, 2304]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_58: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg33_1, view_142, permute_88);  arg33_1 = view_142 = permute_88 = None
        view_143: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_58, [8, 198, 2304]);  addmm_58 = None
        view_144: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_143, [8, 198, 3, 12, 64]);  view_143 = None
        permute_89: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_144, [2, 0, 3, 1, 4]);  view_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_89);  permute_89 = None
        getitem_158: "f32[8, 12, 198, 64]" = unbind_14[0]
        getitem_159: "f32[8, 12, 198, 64]" = unbind_14[1]
        getitem_160: "f32[8, 12, 198, 64]" = unbind_14[2];  unbind_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_158, getitem_159, getitem_160, None, False);  getitem_158 = getitem_159 = getitem_160 = None
        getitem_161: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_90: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None
        view_145: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_90, [8, 198, 768]);  permute_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_146: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_145, [1584, 768]);  view_145 = None
        permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[1584, 768]" = torch.ops.aten.mm.default(view_146, permute_91);  view_146 = permute_91 = None
        add_tensor_31: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_31, arg35_1);  mm_default_31 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_147: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 198, 768]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_105: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_102, view_147);  add_102 = view_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_30 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_165: "f32[8, 198, 1]" = var_mean_30[0]
        getitem_166: "f32[8, 198, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_30: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_105, getitem_166);  getitem_166 = None
        add_106: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_165, 1e-06);  getitem_165 = None
        rsqrt_30: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_102: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_103: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg36_1);  mul_102 = arg36_1 = None
        add_107: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_103, arg37_1);  mul_103 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_148: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_107, [1584, 768]);  add_107 = None
        permute_92: "f32[768, 3072]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_148, permute_92);  view_148 = permute_92 = None
        add_tensor_30: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_30, arg39_1);  mm_default_30 = arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_149: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 198, 3072]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_104: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
        mul_105: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
        erf_14: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_108: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_106: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_108);  mul_104 = add_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_150: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_106, [1584, 3072]);  mul_106 = None
        permute_93: "f32[3072, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[1584, 768]" = torch.ops.aten.mm.default(view_150, permute_93);  view_150 = permute_93 = None
        add_tensor_29: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_29, arg41_1);  mm_default_29 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_151: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 198, 768]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_109: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_105, view_151);  add_105 = view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_31 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_167: "f32[8, 198, 1]" = var_mean_31[0]
        getitem_168: "f32[8, 198, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_31: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_168);  getitem_168 = None
        add_110: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
        rsqrt_31: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_107: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_108: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_107, arg42_1);  mul_107 = arg42_1 = None
        add_111: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_108, arg43_1);  mul_108 = arg43_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_152: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_111, [1584, 768]);  add_111 = None
        permute_94: "f32[768, 2304]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_62: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg45_1, view_152, permute_94);  arg45_1 = view_152 = permute_94 = None
        view_153: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_62, [8, 198, 2304]);  addmm_62 = None
        view_154: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_153, [8, 198, 3, 12, 64]);  view_153 = None
        permute_95: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_154, [2, 0, 3, 1, 4]);  view_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
        getitem_169: "f32[8, 12, 198, 64]" = unbind_15[0]
        getitem_170: "f32[8, 12, 198, 64]" = unbind_15[1]
        getitem_171: "f32[8, 12, 198, 64]" = unbind_15[2];  unbind_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_169, getitem_170, getitem_171, None, False);  getitem_169 = getitem_170 = getitem_171 = None
        getitem_172: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_96: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_155: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_96, [8, 198, 768]);  permute_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_156: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_155, [1584, 768]);  view_155 = None
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[1584, 768]" = torch.ops.aten.mm.default(view_156, permute_97);  view_156 = permute_97 = None
        add_tensor_28: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_28, arg47_1);  mm_default_28 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_157: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 198, 768]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_112: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_109, view_157);  add_109 = view_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_32 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
        getitem_176: "f32[8, 198, 1]" = var_mean_32[0]
        getitem_177: "f32[8, 198, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_32: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_112, getitem_177);  getitem_177 = None
        add_113: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_32: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_109: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_110: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_109, arg48_1);  mul_109 = arg48_1 = None
        add_114: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_110, arg49_1);  mul_110 = arg49_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_158: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_114, [1584, 768]);  add_114 = None
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_158, permute_98);  view_158 = permute_98 = None
        add_tensor_27: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_27, arg51_1);  mm_default_27 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_159: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 198, 3072]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_111: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
        mul_112: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476);  view_159 = None
        erf_15: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_115: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_113: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_111, add_115);  mul_111 = add_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_160: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_113, [1584, 3072]);  mul_113 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[1584, 768]" = torch.ops.aten.mm.default(view_160, permute_99);  view_160 = permute_99 = None
        add_tensor_26: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_26, arg53_1);  mm_default_26 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_161: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 198, 768]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_116: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_112, view_161);  add_112 = view_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_33 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
        getitem_178: "f32[8, 198, 1]" = var_mean_33[0]
        getitem_179: "f32[8, 198, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_33: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_116, getitem_179);  getitem_179 = None
        add_117: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_33: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        mul_114: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_115: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_114, arg54_1);  mul_114 = arg54_1 = None
        add_118: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_115, arg55_1);  mul_115 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_162: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_118, [1584, 768]);  add_118 = None
        permute_100: "f32[768, 2304]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_66: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg57_1, view_162, permute_100);  arg57_1 = view_162 = permute_100 = None
        view_163: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_66, [8, 198, 2304]);  addmm_66 = None
        view_164: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_163, [8, 198, 3, 12, 64]);  view_163 = None
        permute_101: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_164, [2, 0, 3, 1, 4]);  view_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
        getitem_180: "f32[8, 12, 198, 64]" = unbind_16[0]
        getitem_181: "f32[8, 12, 198, 64]" = unbind_16[1]
        getitem_182: "f32[8, 12, 198, 64]" = unbind_16[2];  unbind_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_180, getitem_181, getitem_182, None, False);  getitem_180 = getitem_181 = getitem_182 = None
        getitem_183: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_102: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
        view_165: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_102, [8, 198, 768]);  permute_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_166: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_165, [1584, 768]);  view_165 = None
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[1584, 768]" = torch.ops.aten.mm.default(view_166, permute_103);  view_166 = permute_103 = None
        add_tensor_25: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_25, arg59_1);  mm_default_25 = arg59_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_167: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 198, 768]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_119: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_116, view_167);  add_116 = view_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_34 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_187: "f32[8, 198, 1]" = var_mean_34[0]
        getitem_188: "f32[8, 198, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_34: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_119, getitem_188);  getitem_188 = None
        add_120: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_187, 1e-06);  getitem_187 = None
        rsqrt_34: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_116: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_117: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_116, arg60_1);  mul_116 = arg60_1 = None
        add_121: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_117, arg61_1);  mul_117 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_168: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_121, [1584, 768]);  add_121 = None
        permute_104: "f32[768, 3072]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_168, permute_104);  view_168 = permute_104 = None
        add_tensor_24: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_24, arg63_1);  mm_default_24 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_169: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 198, 3072]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_118: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_169, 0.5)
        mul_119: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_169, 0.7071067811865476);  view_169 = None
        erf_16: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_122: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_120: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_118, add_122);  mul_118 = add_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_170: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_120, [1584, 3072]);  mul_120 = None
        permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[1584, 768]" = torch.ops.aten.mm.default(view_170, permute_105);  view_170 = permute_105 = None
        add_tensor_23: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_23, arg65_1);  mm_default_23 = arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_171: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 198, 768]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_123: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_119, view_171);  add_119 = view_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_35 = torch.ops.aten.var_mean.correction(add_123, [2], correction = 0, keepdim = True)
        getitem_189: "f32[8, 198, 1]" = var_mean_35[0]
        getitem_190: "f32[8, 198, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_35: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_123, getitem_190);  getitem_190 = None
        add_124: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_189, 1e-06);  getitem_189 = None
        rsqrt_35: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_121: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_122: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_121, arg66_1);  mul_121 = arg66_1 = None
        add_125: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_122, arg67_1);  mul_122 = arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_172: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_125, [1584, 768]);  add_125 = None
        permute_106: "f32[768, 2304]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_70: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg69_1, view_172, permute_106);  arg69_1 = view_172 = permute_106 = None
        view_173: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_70, [8, 198, 2304]);  addmm_70 = None
        view_174: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_173, [8, 198, 3, 12, 64]);  view_173 = None
        permute_107: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_174, [2, 0, 3, 1, 4]);  view_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
        getitem_191: "f32[8, 12, 198, 64]" = unbind_17[0]
        getitem_192: "f32[8, 12, 198, 64]" = unbind_17[1]
        getitem_193: "f32[8, 12, 198, 64]" = unbind_17[2];  unbind_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_191, getitem_192, getitem_193, None, False);  getitem_191 = getitem_192 = getitem_193 = None
        getitem_194: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_108: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
        view_175: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_108, [8, 198, 768]);  permute_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_176: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_175, [1584, 768]);  view_175 = None
        permute_109: "f32[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[1584, 768]" = torch.ops.aten.mm.default(view_176, permute_109);  view_176 = permute_109 = None
        add_tensor_22: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_22, arg71_1);  mm_default_22 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_177: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 198, 768]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_126: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_123, view_177);  add_123 = view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_36 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
        getitem_198: "f32[8, 198, 1]" = var_mean_36[0]
        getitem_199: "f32[8, 198, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_36: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_126, getitem_199);  getitem_199 = None
        add_127: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_36: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        mul_123: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_124: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_123, arg72_1);  mul_123 = arg72_1 = None
        add_128: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_124, arg73_1);  mul_124 = arg73_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_178: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_128, [1584, 768]);  add_128 = None
        permute_110: "f32[768, 3072]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_178, permute_110);  view_178 = permute_110 = None
        add_tensor_21: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_21, arg75_1);  mm_default_21 = arg75_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_179: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 198, 3072]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_125: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.5)
        mul_126: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.7071067811865476);  view_179 = None
        erf_17: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_129: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_127: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_125, add_129);  mul_125 = add_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_180: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_127, [1584, 3072]);  mul_127 = None
        permute_111: "f32[3072, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[1584, 768]" = torch.ops.aten.mm.default(view_180, permute_111);  view_180 = permute_111 = None
        add_tensor_20: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_20, arg77_1);  mm_default_20 = arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_181: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 198, 768]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_130: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_126, view_181);  add_126 = view_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_37 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
        getitem_200: "f32[8, 198, 1]" = var_mean_37[0]
        getitem_201: "f32[8, 198, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_37: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_130, getitem_201);  getitem_201 = None
        add_131: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_37: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_128: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_129: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_128, arg78_1);  mul_128 = arg78_1 = None
        add_132: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_129, arg79_1);  mul_129 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_182: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_132, [1584, 768]);  add_132 = None
        permute_112: "f32[768, 2304]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_74: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg81_1, view_182, permute_112);  arg81_1 = view_182 = permute_112 = None
        view_183: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_74, [8, 198, 2304]);  addmm_74 = None
        view_184: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_183, [8, 198, 3, 12, 64]);  view_183 = None
        permute_113: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_184, [2, 0, 3, 1, 4]);  view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_18 = torch.ops.aten.unbind.int(permute_113);  permute_113 = None
        getitem_202: "f32[8, 12, 198, 64]" = unbind_18[0]
        getitem_203: "f32[8, 12, 198, 64]" = unbind_18[1]
        getitem_204: "f32[8, 12, 198, 64]" = unbind_18[2];  unbind_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_202, getitem_203, getitem_204, None, False);  getitem_202 = getitem_203 = getitem_204 = None
        getitem_205: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_114: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
        view_185: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_114, [8, 198, 768]);  permute_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_186: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_185, [1584, 768]);  view_185 = None
        permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[1584, 768]" = torch.ops.aten.mm.default(view_186, permute_115);  view_186 = permute_115 = None
        add_tensor_19: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_19, arg83_1);  mm_default_19 = arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_187: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 198, 768]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_133: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_130, view_187);  add_130 = view_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_38 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_209: "f32[8, 198, 1]" = var_mean_38[0]
        getitem_210: "f32[8, 198, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_38: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_133, getitem_210);  getitem_210 = None
        add_134: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_209, 1e-06);  getitem_209 = None
        rsqrt_38: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_130: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_131: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_130, arg84_1);  mul_130 = arg84_1 = None
        add_135: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_131, arg85_1);  mul_131 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_188: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_135, [1584, 768]);  add_135 = None
        permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_188, permute_116);  view_188 = permute_116 = None
        add_tensor_18: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_18, arg87_1);  mm_default_18 = arg87_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_189: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 198, 3072]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_132: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_133: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_18: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_136: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_134: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_132, add_136);  mul_132 = add_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_190: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_134, [1584, 3072]);  mul_134 = None
        permute_117: "f32[3072, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[1584, 768]" = torch.ops.aten.mm.default(view_190, permute_117);  view_190 = permute_117 = None
        add_tensor_17: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_17, arg89_1);  mm_default_17 = arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_191: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 198, 768]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_137: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_133, view_191);  add_133 = view_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_39 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_211: "f32[8, 198, 1]" = var_mean_39[0]
        getitem_212: "f32[8, 198, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_39: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_137, getitem_212);  getitem_212 = None
        add_138: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_211, 1e-06);  getitem_211 = None
        rsqrt_39: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_135: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_136: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_135, arg90_1);  mul_135 = arg90_1 = None
        add_139: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_136, arg91_1);  mul_136 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_192: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_139, [1584, 768]);  add_139 = None
        permute_118: "f32[768, 2304]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_78: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg93_1, view_192, permute_118);  arg93_1 = view_192 = permute_118 = None
        view_193: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_78, [8, 198, 2304]);  addmm_78 = None
        view_194: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_193, [8, 198, 3, 12, 64]);  view_193 = None
        permute_119: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_194, [2, 0, 3, 1, 4]);  view_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_19 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
        getitem_213: "f32[8, 12, 198, 64]" = unbind_19[0]
        getitem_214: "f32[8, 12, 198, 64]" = unbind_19[1]
        getitem_215: "f32[8, 12, 198, 64]" = unbind_19[2];  unbind_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_213, getitem_214, getitem_215, None, False);  getitem_213 = getitem_214 = getitem_215 = None
        getitem_216: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_120: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
        view_195: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_120, [8, 198, 768]);  permute_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_196: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_195, [1584, 768]);  view_195 = None
        permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[1584, 768]" = torch.ops.aten.mm.default(view_196, permute_121);  view_196 = permute_121 = None
        add_tensor_16: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_16, arg95_1);  mm_default_16 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_197: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 198, 768]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_140: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_137, view_197);  add_137 = view_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_40 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_220: "f32[8, 198, 1]" = var_mean_40[0]
        getitem_221: "f32[8, 198, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_40: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_140, getitem_221);  getitem_221 = None
        add_141: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
        rsqrt_40: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_137: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_138: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_137, arg96_1);  mul_137 = arg96_1 = None
        add_142: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_138, arg97_1);  mul_138 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_198: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_142, [1584, 768]);  add_142 = None
        permute_122: "f32[768, 3072]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_198, permute_122);  view_198 = permute_122 = None
        add_tensor_15: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_15, arg99_1);  mm_default_15 = arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_199: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 198, 3072]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_139: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.5)
        mul_140: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476);  view_199 = None
        erf_19: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_143: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_141: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_139, add_143);  mul_139 = add_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_200: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_141, [1584, 3072]);  mul_141 = None
        permute_123: "f32[3072, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[1584, 768]" = torch.ops.aten.mm.default(view_200, permute_123);  view_200 = permute_123 = None
        add_tensor_14: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_14, arg101_1);  mm_default_14 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_201: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 198, 768]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_144: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_140, view_201);  add_140 = view_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_41 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
        getitem_222: "f32[8, 198, 1]" = var_mean_41[0]
        getitem_223: "f32[8, 198, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_41: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_144, getitem_223);  getitem_223 = None
        add_145: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_41: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        mul_142: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_143: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_142, arg102_1);  mul_142 = arg102_1 = None
        add_146: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_143, arg103_1);  mul_143 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_202: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_146, [1584, 768]);  add_146 = None
        permute_124: "f32[768, 2304]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_82: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg105_1, view_202, permute_124);  arg105_1 = view_202 = permute_124 = None
        view_203: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_82, [8, 198, 2304]);  addmm_82 = None
        view_204: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_203, [8, 198, 3, 12, 64]);  view_203 = None
        permute_125: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_204, [2, 0, 3, 1, 4]);  view_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_20 = torch.ops.aten.unbind.int(permute_125);  permute_125 = None
        getitem_224: "f32[8, 12, 198, 64]" = unbind_20[0]
        getitem_225: "f32[8, 12, 198, 64]" = unbind_20[1]
        getitem_226: "f32[8, 12, 198, 64]" = unbind_20[2];  unbind_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_224, getitem_225, getitem_226, None, False);  getitem_224 = getitem_225 = getitem_226 = None
        getitem_227: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_126: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_227, [0, 2, 1, 3]);  getitem_227 = None
        view_205: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_126, [8, 198, 768]);  permute_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_206: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_205, [1584, 768]);  view_205 = None
        permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[1584, 768]" = torch.ops.aten.mm.default(view_206, permute_127);  view_206 = permute_127 = None
        add_tensor_13: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_13, arg107_1);  mm_default_13 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_207: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 198, 768]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_147: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_144, view_207);  add_144 = view_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_42 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_231: "f32[8, 198, 1]" = var_mean_42[0]
        getitem_232: "f32[8, 198, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_42: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_147, getitem_232);  getitem_232 = None
        add_148: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_231, 1e-06);  getitem_231 = None
        rsqrt_42: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_144: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_145: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_144, arg108_1);  mul_144 = arg108_1 = None
        add_149: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_145, arg109_1);  mul_145 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_208: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_149, [1584, 768]);  add_149 = None
        permute_128: "f32[768, 3072]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_208, permute_128);  view_208 = permute_128 = None
        add_tensor_12: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_12, arg111_1);  mm_default_12 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_209: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 198, 3072]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_146: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_209, 0.5)
        mul_147: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_209, 0.7071067811865476);  view_209 = None
        erf_20: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
        add_150: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_148: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_146, add_150);  mul_146 = add_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_210: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_148, [1584, 3072]);  mul_148 = None
        permute_129: "f32[3072, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[1584, 768]" = torch.ops.aten.mm.default(view_210, permute_129);  view_210 = permute_129 = None
        add_tensor_11: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_11, arg113_1);  mm_default_11 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_211: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 198, 768]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_151: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_147, view_211);  add_147 = view_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_43 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
        getitem_233: "f32[8, 198, 1]" = var_mean_43[0]
        getitem_234: "f32[8, 198, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_43: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_151, getitem_234);  getitem_234 = None
        add_152: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_233, 1e-06);  getitem_233 = None
        rsqrt_43: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_149: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_150: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_149, arg114_1);  mul_149 = arg114_1 = None
        add_153: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_150, arg115_1);  mul_150 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_212: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_153, [1584, 768]);  add_153 = None
        permute_130: "f32[768, 2304]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_86: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg117_1, view_212, permute_130);  arg117_1 = view_212 = permute_130 = None
        view_213: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_86, [8, 198, 2304]);  addmm_86 = None
        view_214: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_213, [8, 198, 3, 12, 64]);  view_213 = None
        permute_131: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_214, [2, 0, 3, 1, 4]);  view_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_21 = torch.ops.aten.unbind.int(permute_131);  permute_131 = None
        getitem_235: "f32[8, 12, 198, 64]" = unbind_21[0]
        getitem_236: "f32[8, 12, 198, 64]" = unbind_21[1]
        getitem_237: "f32[8, 12, 198, 64]" = unbind_21[2];  unbind_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_235, getitem_236, getitem_237, None, False);  getitem_235 = getitem_236 = getitem_237 = None
        getitem_238: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_132: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None
        view_215: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_132, [8, 198, 768]);  permute_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_216: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_215, [1584, 768]);  view_215 = None
        permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[1584, 768]" = torch.ops.aten.mm.default(view_216, permute_133);  view_216 = permute_133 = None
        add_tensor_10: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_10, arg119_1);  mm_default_10 = arg119_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_217: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 198, 768]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_154: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_151, view_217);  add_151 = view_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_44 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
        getitem_242: "f32[8, 198, 1]" = var_mean_44[0]
        getitem_243: "f32[8, 198, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_44: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_154, getitem_243);  getitem_243 = None
        add_155: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_44: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_151: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_152: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_151, arg120_1);  mul_151 = arg120_1 = None
        add_156: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_152, arg121_1);  mul_152 = arg121_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_218: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_156, [1584, 768]);  add_156 = None
        permute_134: "f32[768, 3072]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_218, permute_134);  view_218 = permute_134 = None
        add_tensor_9: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_9, arg123_1);  mm_default_9 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_219: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 198, 3072]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_153: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_154: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_21: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_157: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_155: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_153, add_157);  mul_153 = add_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_220: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_155, [1584, 3072]);  mul_155 = None
        permute_135: "f32[3072, 768]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[1584, 768]" = torch.ops.aten.mm.default(view_220, permute_135);  view_220 = permute_135 = None
        add_tensor_8: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_8, arg125_1);  mm_default_8 = arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_221: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 198, 768]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_158: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_154, view_221);  add_154 = view_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_45 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_244: "f32[8, 198, 1]" = var_mean_45[0]
        getitem_245: "f32[8, 198, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_45: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_158, getitem_245);  getitem_245 = None
        add_159: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
        rsqrt_45: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_156: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_157: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_156, arg126_1);  mul_156 = arg126_1 = None
        add_160: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_157, arg127_1);  mul_157 = arg127_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_222: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_160, [1584, 768]);  add_160 = None
        permute_136: "f32[768, 2304]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_90: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg129_1, view_222, permute_136);  arg129_1 = view_222 = permute_136 = None
        view_223: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_90, [8, 198, 2304]);  addmm_90 = None
        view_224: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_223, [8, 198, 3, 12, 64]);  view_223 = None
        permute_137: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_224, [2, 0, 3, 1, 4]);  view_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_22 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
        getitem_246: "f32[8, 12, 198, 64]" = unbind_22[0]
        getitem_247: "f32[8, 12, 198, 64]" = unbind_22[1]
        getitem_248: "f32[8, 12, 198, 64]" = unbind_22[2];  unbind_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_246, getitem_247, getitem_248, None, False);  getitem_246 = getitem_247 = getitem_248 = None
        getitem_249: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_138: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_249, [0, 2, 1, 3]);  getitem_249 = None
        view_225: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_138, [8, 198, 768]);  permute_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_226: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_225, [1584, 768]);  view_225 = None
        permute_139: "f32[768, 768]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[1584, 768]" = torch.ops.aten.mm.default(view_226, permute_139);  view_226 = permute_139 = None
        add_tensor_7: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_7, arg131_1);  mm_default_7 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_227: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 198, 768]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_161: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_158, view_227);  add_158 = view_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_46 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_253: "f32[8, 198, 1]" = var_mean_46[0]
        getitem_254: "f32[8, 198, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_46: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_161, getitem_254);  getitem_254 = None
        add_162: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_253, 1e-06);  getitem_253 = None
        rsqrt_46: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_158: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_159: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_158, arg132_1);  mul_158 = arg132_1 = None
        add_163: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_159, arg133_1);  mul_159 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_228: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_163, [1584, 768]);  add_163 = None
        permute_140: "f32[768, 3072]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_228, permute_140);  view_228 = permute_140 = None
        add_tensor_6: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_6, arg135_1);  mm_default_6 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_229: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 198, 3072]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_160: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_229, 0.5)
        mul_161: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476);  view_229 = None
        erf_22: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_164: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_162: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_160, add_164);  mul_160 = add_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_230: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_162, [1584, 3072]);  mul_162 = None
        permute_141: "f32[3072, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[1584, 768]" = torch.ops.aten.mm.default(view_230, permute_141);  view_230 = permute_141 = None
        add_tensor_5: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_5, arg137_1);  mm_default_5 = arg137_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_231: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 198, 768]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_165: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_161, view_231);  add_161 = view_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_47 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_255: "f32[8, 198, 1]" = var_mean_47[0]
        getitem_256: "f32[8, 198, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_47: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_165, getitem_256);  getitem_256 = None
        add_166: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_255, 1e-06);  getitem_255 = None
        rsqrt_47: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_163: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_164: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_163, arg138_1);  mul_163 = arg138_1 = None
        add_167: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_164, arg139_1);  mul_164 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_232: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_167, [1584, 768]);  add_167 = None
        permute_142: "f32[768, 2304]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_94: "f32[1584, 2304]" = torch.ops.aten.addmm.default(arg141_1, view_232, permute_142);  arg141_1 = view_232 = permute_142 = None
        view_233: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_94, [8, 198, 2304]);  addmm_94 = None
        view_234: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(view_233, [8, 198, 3, 12, 64]);  view_233 = None
        permute_143: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_234, [2, 0, 3, 1, 4]);  view_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_23 = torch.ops.aten.unbind.int(permute_143);  permute_143 = None
        getitem_257: "f32[8, 12, 198, 64]" = unbind_23[0]
        getitem_258: "f32[8, 12, 198, 64]" = unbind_23[1]
        getitem_259: "f32[8, 12, 198, 64]" = unbind_23[2];  unbind_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_257, getitem_258, getitem_259, None, False);  getitem_257 = getitem_258 = getitem_259 = None
        getitem_260: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_144: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        view_235: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(permute_144, [8, 198, 768]);  permute_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_236: "f32[1584, 768]" = torch.ops.aten.reshape.default(view_235, [1584, 768]);  view_235 = None
        permute_145: "f32[768, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[1584, 768]" = torch.ops.aten.mm.default(view_236, permute_145);  view_236 = permute_145 = None
        add_tensor_4: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_4, arg143_1);  mm_default_4 = arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_237: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 198, 768]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_168: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_165, view_237);  add_165 = view_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_48 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
        getitem_264: "f32[8, 198, 1]" = var_mean_48[0]
        getitem_265: "f32[8, 198, 1]" = var_mean_48[1];  var_mean_48 = None
        sub_48: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_168, getitem_265);  getitem_265 = None
        add_169: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-06);  getitem_264 = None
        rsqrt_48: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        mul_165: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_166: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_165, arg144_1);  mul_165 = arg144_1 = None
        add_170: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_166, arg145_1);  mul_166 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_238: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_170, [1584, 768]);  add_170 = None
        permute_146: "f32[768, 3072]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_238, permute_146);  view_238 = permute_146 = None
        add_tensor_3: "f32[1584, 3072]" = torch.ops.aten.add.Tensor(mm_default_3, arg147_1);  mm_default_3 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_239: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 198, 3072]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_167: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_168: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_23: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_171: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_169: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_167, add_171);  mul_167 = add_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_240: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_169, [1584, 3072]);  mul_169 = None
        permute_147: "f32[3072, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[1584, 768]" = torch.ops.aten.mm.default(view_240, permute_147);  view_240 = permute_147 = None
        add_tensor_2: "f32[1584, 768]" = torch.ops.aten.add.Tensor(mm_default_2, arg149_1);  mm_default_2 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_241: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 198, 768]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_172: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_168, view_241);  add_168 = view_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:811 in forward_features, code: x = self.norm(x)
        var_mean_49 = torch.ops.aten.var_mean.correction(add_172, [2], correction = 0, keepdim = True)
        getitem_266: "f32[8, 198, 1]" = var_mean_49[0]
        getitem_267: "f32[8, 198, 1]" = var_mean_49[1];  var_mean_49 = None
        sub_49: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_172, getitem_267);  add_172 = getitem_267 = None
        add_173: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_49: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        mul_170: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_171: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_170, arg150_1);  mul_170 = arg150_1 = None
        add_174: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_171, arg151_1);  mul_171 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:107 in forward_head, code: x, x_dist = x[:, 0], x[:, 1]
        select_2: "f32[8, 768]" = torch.ops.aten.select.int(add_174, 1, 0)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:110 in forward_head, code: x = self.head(x)
        permute_148: "f32[768, 1000]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[8, 1000]" = torch.ops.aten.mm.default(select_2, permute_148);  select_2 = permute_148 = None
        add_tensor_1: "f32[8, 1000]" = torch.ops.aten.add.Tensor(mm_default_1, arg153_1);  mm_default_1 = arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:107 in forward_head, code: x, x_dist = x[:, 0], x[:, 1]
        select_3: "f32[8, 768]" = torch.ops.aten.select.int(add_174, 1, 1);  add_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:111 in forward_head, code: x_dist = self.head_dist(x_dist)
        permute_149: "f32[768, 1000]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[8, 1000]" = torch.ops.aten.mm.default(select_3, permute_149);  select_3 = permute_149 = None
        add_tensor: "f32[8, 1000]" = torch.ops.aten.add.Tensor(mm_default, arg155_1);  mm_default = arg155_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/deit.py:117 in forward_head, code: return (x + x_dist) / 2
        add_175: "f32[8, 1000]" = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor);  add_tensor_1 = add_tensor = None
        div_1: "f32[8, 1000]" = torch.ops.aten.div.Tensor(add_175, 2);  add_175 = None
        return (div_1,)
        