class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[256, 3, 14, 14]", arg1_1: "f32[256]", arg2_1: "f32[8, 3, 224, 224]", arg3_1: "f32[1, 256, 31, 31]", arg4_1: "f32[1, 1, 256]", arg5_1: "f32[256]", arg6_1: "f32[256]", arg7_1: "f32[768, 256]", arg8_1: "f32[768]", arg9_1: "f32[256, 256]", arg10_1: "f32[256]", arg11_1: "f32[256]", arg12_1: "f32[256]", arg13_1: "f32[1024, 256]", arg14_1: "f32[1024]", arg15_1: "f32[256, 1024]", arg16_1: "f32[256]", arg17_1: "f32[256]", arg18_1: "f32[256]", arg19_1: "f32[768, 256]", arg20_1: "f32[768]", arg21_1: "f32[256, 256]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[256]", arg25_1: "f32[1024, 256]", arg26_1: "f32[1024]", arg27_1: "f32[256, 1024]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[768, 256]", arg32_1: "f32[768]", arg33_1: "f32[256, 256]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[1024, 256]", arg38_1: "f32[1024]", arg39_1: "f32[256, 1024]", arg40_1: "f32[256]", arg41_1: "f32[512, 1, 3, 3]", arg42_1: "f32[512]", arg43_1: "f32[512, 256]", arg44_1: "f32[512]", arg45_1: "f32[512]", arg46_1: "f32[512]", arg47_1: "f32[1536, 512]", arg48_1: "f32[1536]", arg49_1: "f32[512, 512]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[512]", arg53_1: "f32[2048, 512]", arg54_1: "f32[2048]", arg55_1: "f32[512, 2048]", arg56_1: "f32[512]", arg57_1: "f32[512]", arg58_1: "f32[512]", arg59_1: "f32[1536, 512]", arg60_1: "f32[1536]", arg61_1: "f32[512, 512]", arg62_1: "f32[512]", arg63_1: "f32[512]", arg64_1: "f32[512]", arg65_1: "f32[2048, 512]", arg66_1: "f32[2048]", arg67_1: "f32[512, 2048]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[512]", arg71_1: "f32[1536, 512]", arg72_1: "f32[1536]", arg73_1: "f32[512, 512]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[2048, 512]", arg78_1: "f32[2048]", arg79_1: "f32[512, 2048]", arg80_1: "f32[512]", arg81_1: "f32[512]", arg82_1: "f32[512]", arg83_1: "f32[1536, 512]", arg84_1: "f32[1536]", arg85_1: "f32[512, 512]", arg86_1: "f32[512]", arg87_1: "f32[512]", arg88_1: "f32[512]", arg89_1: "f32[2048, 512]", arg90_1: "f32[2048]", arg91_1: "f32[512, 2048]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[512]", arg95_1: "f32[1536, 512]", arg96_1: "f32[1536]", arg97_1: "f32[512, 512]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[512]", arg101_1: "f32[2048, 512]", arg102_1: "f32[2048]", arg103_1: "f32[512, 2048]", arg104_1: "f32[512]", arg105_1: "f32[512]", arg106_1: "f32[512]", arg107_1: "f32[1536, 512]", arg108_1: "f32[1536]", arg109_1: "f32[512, 512]", arg110_1: "f32[512]", arg111_1: "f32[512]", arg112_1: "f32[512]", arg113_1: "f32[2048, 512]", arg114_1: "f32[2048]", arg115_1: "f32[512, 2048]", arg116_1: "f32[512]", arg117_1: "f32[1024, 1, 3, 3]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 512]", arg120_1: "f32[1024]", arg121_1: "f32[1024]", arg122_1: "f32[1024]", arg123_1: "f32[3072, 1024]", arg124_1: "f32[3072]", arg125_1: "f32[1024, 1024]", arg126_1: "f32[1024]", arg127_1: "f32[1024]", arg128_1: "f32[1024]", arg129_1: "f32[4096, 1024]", arg130_1: "f32[4096]", arg131_1: "f32[1024, 4096]", arg132_1: "f32[1024]", arg133_1: "f32[1024]", arg134_1: "f32[1024]", arg135_1: "f32[3072, 1024]", arg136_1: "f32[3072]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024]", arg140_1: "f32[1024]", arg141_1: "f32[4096, 1024]", arg142_1: "f32[4096]", arg143_1: "f32[1024, 4096]", arg144_1: "f32[1024]", arg145_1: "f32[1024]", arg146_1: "f32[1024]", arg147_1: "f32[3072, 1024]", arg148_1: "f32[3072]", arg149_1: "f32[1024, 1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024]", arg152_1: "f32[1024]", arg153_1: "f32[4096, 1024]", arg154_1: "f32[4096]", arg155_1: "f32[1024, 4096]", arg156_1: "f32[1024]", arg157_1: "f32[1024]", arg158_1: "f32[1024]", arg159_1: "f32[3072, 1024]", arg160_1: "f32[3072]", arg161_1: "f32[1024, 1024]", arg162_1: "f32[1024]", arg163_1: "f32[1024]", arg164_1: "f32[1024]", arg165_1: "f32[4096, 1024]", arg166_1: "f32[4096]", arg167_1: "f32[1024, 4096]", arg168_1: "f32[1024]", arg169_1: "f32[1024]", arg170_1: "f32[1024]", arg171_1: "f32[1000, 1024]", arg172_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:138 in forward, code: x = self.conv(x)
        convolution_3: "f32[8, 256, 31, 31]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [7, 7], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:259 in forward_features, code: x = self.pos_drop(x + self.pos_embed)
        add_96: "f32[8, 256, 31, 31]" = torch.ops.aten.add.Tensor(convolution_3, arg3_1);  convolution_3 = arg3_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:260 in forward_features, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        expand_1: "f32[8, 1, 256]" = torch.ops.aten.expand.default(arg4_1, [8, -1, -1]);  arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:81 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_140: "f32[8, 256, 961]" = torch.ops.aten.view.default(add_96, [8, 256, 961]);  add_96 = None
        permute_87: "f32[8, 961, 256]" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:82 in forward, code: x = torch.cat((cls_tokens, x), dim=1)
        cat_3: "f32[8, 962, 256]" = torch.ops.aten.cat.default([expand_1, permute_87], 1);  expand_1 = permute_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_27 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
        getitem_145: "f32[8, 962, 1]" = var_mean_27[0]
        getitem_146: "f32[8, 962, 1]" = var_mean_27[1];  var_mean_27 = None
        add_97: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-06);  getitem_145 = None
        rsqrt_27: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_27: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(cat_3, getitem_146);  getitem_146 = None
        mul_93: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_94: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_93, arg5_1);  mul_93 = arg5_1 = None
        add_98: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_94, arg6_1);  mul_94 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_141: "f32[7696, 256]" = torch.ops.aten.view.default(add_98, [7696, 256]);  add_98 = None
        permute_88: "f32[256, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_53: "f32[7696, 768]" = torch.ops.aten.addmm.default(arg8_1, view_141, permute_88);  arg8_1 = view_141 = permute_88 = None
        view_142: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm_53, [8, 962, 768]);  addmm_53 = None
        view_143: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_142, [8, 962, 3, 4, 64]);  view_142 = None
        permute_89: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_143, [2, 0, 3, 1, 4]);  view_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_89);  permute_89 = None
        getitem_147: "f32[8, 4, 962, 64]" = unbind_13[0]
        getitem_148: "f32[8, 4, 962, 64]" = unbind_13[1]
        getitem_149: "f32[8, 4, 962, 64]" = unbind_13[2];  unbind_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_147, getitem_148, getitem_149, None, False);  getitem_147 = getitem_148 = getitem_149 = None
        getitem_150: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_90: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        view_144: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_90, [8, 962, 256]);  permute_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_145: "f32[7696, 256]" = torch.ops.aten.view.default(view_144, [7696, 256]);  view_144 = None
        permute_91: "f32[256, 256]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_54: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg10_1, view_145, permute_91);  arg10_1 = view_145 = permute_91 = None
        view_146: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_54, [8, 962, 256]);  addmm_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_99: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(cat_3, view_146);  cat_3 = view_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_28 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_154: "f32[8, 962, 1]" = var_mean_28[0]
        getitem_155: "f32[8, 962, 1]" = var_mean_28[1];  var_mean_28 = None
        add_100: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_28: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_28: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_99, getitem_155);  getitem_155 = None
        mul_95: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_96: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_95, arg11_1);  mul_95 = arg11_1 = None
        add_101: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_96, arg12_1);  mul_96 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_147: "f32[7696, 256]" = torch.ops.aten.view.default(add_101, [7696, 256]);  add_101 = None
        permute_92: "f32[256, 1024]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_55: "f32[7696, 1024]" = torch.ops.aten.addmm.default(arg14_1, view_147, permute_92);  arg14_1 = view_147 = permute_92 = None
        view_148: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_55, [8, 962, 1024]);  addmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_97: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_148, 0.5)
        mul_98: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_148, 0.7071067811865476);  view_148 = None
        erf_13: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_102: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_99: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_97, add_102);  mul_97 = add_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_149: "f32[7696, 1024]" = torch.ops.aten.view.default(mul_99, [7696, 1024]);  mul_99 = None
        permute_93: "f32[1024, 256]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_56: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg16_1, view_149, permute_93);  arg16_1 = view_149 = permute_93 = None
        view_150: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_56, [8, 962, 256]);  addmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_103: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_99, view_150);  add_99 = view_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_29 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_156: "f32[8, 962, 1]" = var_mean_29[0]
        getitem_157: "f32[8, 962, 1]" = var_mean_29[1];  var_mean_29 = None
        add_104: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
        rsqrt_29: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_29: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_103, getitem_157);  getitem_157 = None
        mul_100: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_101: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_100, arg17_1);  mul_100 = arg17_1 = None
        add_105: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_101, arg18_1);  mul_101 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_151: "f32[7696, 256]" = torch.ops.aten.view.default(add_105, [7696, 256]);  add_105 = None
        permute_94: "f32[256, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_57: "f32[7696, 768]" = torch.ops.aten.addmm.default(arg20_1, view_151, permute_94);  arg20_1 = view_151 = permute_94 = None
        view_152: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm_57, [8, 962, 768]);  addmm_57 = None
        view_153: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_152, [8, 962, 3, 4, 64]);  view_152 = None
        permute_95: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_153, [2, 0, 3, 1, 4]);  view_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
        getitem_158: "f32[8, 4, 962, 64]" = unbind_14[0]
        getitem_159: "f32[8, 4, 962, 64]" = unbind_14[1]
        getitem_160: "f32[8, 4, 962, 64]" = unbind_14[2];  unbind_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_158, getitem_159, getitem_160, None, False);  getitem_158 = getitem_159 = getitem_160 = None
        getitem_161: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_96: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None
        view_154: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_96, [8, 962, 256]);  permute_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_155: "f32[7696, 256]" = torch.ops.aten.view.default(view_154, [7696, 256]);  view_154 = None
        permute_97: "f32[256, 256]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_58: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg22_1, view_155, permute_97);  arg22_1 = view_155 = permute_97 = None
        view_156: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_58, [8, 962, 256]);  addmm_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_106: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_103, view_156);  add_103 = view_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_30 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
        getitem_165: "f32[8, 962, 1]" = var_mean_30[0]
        getitem_166: "f32[8, 962, 1]" = var_mean_30[1];  var_mean_30 = None
        add_107: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_165, 1e-06);  getitem_165 = None
        rsqrt_30: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_30: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_106, getitem_166);  getitem_166 = None
        mul_102: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_103: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_102, arg23_1);  mul_102 = arg23_1 = None
        add_108: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_103, arg24_1);  mul_103 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_157: "f32[7696, 256]" = torch.ops.aten.view.default(add_108, [7696, 256]);  add_108 = None
        permute_98: "f32[256, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_59: "f32[7696, 1024]" = torch.ops.aten.addmm.default(arg26_1, view_157, permute_98);  arg26_1 = view_157 = permute_98 = None
        view_158: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_59, [8, 962, 1024]);  addmm_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_104: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
        mul_105: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
        erf_14: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_109: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_106: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_104, add_109);  mul_104 = add_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_159: "f32[7696, 1024]" = torch.ops.aten.view.default(mul_106, [7696, 1024]);  mul_106 = None
        permute_99: "f32[1024, 256]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_60: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg28_1, view_159, permute_99);  arg28_1 = view_159 = permute_99 = None
        view_160: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_60, [8, 962, 256]);  addmm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_110: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_106, view_160);  add_106 = view_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_31 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_167: "f32[8, 962, 1]" = var_mean_31[0]
        getitem_168: "f32[8, 962, 1]" = var_mean_31[1];  var_mean_31 = None
        add_111: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-06);  getitem_167 = None
        rsqrt_31: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_31: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_110, getitem_168);  getitem_168 = None
        mul_107: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_108: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_107, arg29_1);  mul_107 = arg29_1 = None
        add_112: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_108, arg30_1);  mul_108 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_161: "f32[7696, 256]" = torch.ops.aten.view.default(add_112, [7696, 256]);  add_112 = None
        permute_100: "f32[256, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_61: "f32[7696, 768]" = torch.ops.aten.addmm.default(arg32_1, view_161, permute_100);  arg32_1 = view_161 = permute_100 = None
        view_162: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm_61, [8, 962, 768]);  addmm_61 = None
        view_163: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_162, [8, 962, 3, 4, 64]);  view_162 = None
        permute_101: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
        getitem_169: "f32[8, 4, 962, 64]" = unbind_15[0]
        getitem_170: "f32[8, 4, 962, 64]" = unbind_15[1]
        getitem_171: "f32[8, 4, 962, 64]" = unbind_15[2];  unbind_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_169, getitem_170, getitem_171, None, False);  getitem_169 = getitem_170 = getitem_171 = None
        getitem_172: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_102: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_164: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_102, [8, 962, 256]);  permute_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_165: "f32[7696, 256]" = torch.ops.aten.view.default(view_164, [7696, 256]);  view_164 = None
        permute_103: "f32[256, 256]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_62: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg34_1, view_165, permute_103);  arg34_1 = view_165 = permute_103 = None
        view_166: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_62, [8, 962, 256]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_113: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_110, view_166);  add_110 = view_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_32 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_176: "f32[8, 962, 1]" = var_mean_32[0]
        getitem_177: "f32[8, 962, 1]" = var_mean_32[1];  var_mean_32 = None
        add_114: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_32: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_32: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_113, getitem_177);  getitem_177 = None
        mul_109: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_110: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_109, arg35_1);  mul_109 = arg35_1 = None
        add_115: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_110, arg36_1);  mul_110 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_167: "f32[7696, 256]" = torch.ops.aten.view.default(add_115, [7696, 256]);  add_115 = None
        permute_104: "f32[256, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_63: "f32[7696, 1024]" = torch.ops.aten.addmm.default(arg38_1, view_167, permute_104);  arg38_1 = view_167 = permute_104 = None
        view_168: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_63, [8, 962, 1024]);  addmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_111: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_168, 0.5)
        mul_112: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_168, 0.7071067811865476);  view_168 = None
        erf_15: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_116: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_113: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_111, add_116);  mul_111 = add_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_169: "f32[7696, 1024]" = torch.ops.aten.view.default(mul_113, [7696, 1024]);  mul_113 = None
        permute_105: "f32[1024, 256]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_64: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg40_1, view_169, permute_105);  arg40_1 = view_169 = permute_105 = None
        view_170: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_64, [8, 962, 256]);  addmm_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_117: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_113, view_170);  add_113 = view_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:87 in forward, code: cls_tokens = x[:, :token_length]
        slice_15: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_117, 1, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:88 in forward, code: x = x[:, token_length:]
        slice_17: "f32[8, 961, 256]" = torch.ops.aten.slice.Tensor(add_117, 1, 1, 9223372036854775807);  add_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:89 in forward, code: x = x.transpose(1, 2).reshape(B, C, H, W)
        permute_106: "f32[8, 256, 961]" = torch.ops.aten.permute.default(slice_17, [0, 2, 1]);  slice_17 = None
        view_171: "f32[8, 256, 31, 31]" = torch.ops.aten.view.default(permute_106, [8, 256, 31, 31]);  permute_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:110 in forward, code: x = self.conv(x)
        convolution_4: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(view_171, arg41_1, arg42_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 256);  view_171 = arg41_1 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:111 in forward, code: cls_token = self.fc(cls_token)
        permute_107: "f32[256, 512]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        view_172: "f32[8, 256]" = torch.ops.aten.view.default(slice_15, [8, 256]);  slice_15 = None
        mm_2: "f32[8, 512]" = torch.ops.aten.mm.default(view_172, permute_107);  view_172 = permute_107 = None
        view_173: "f32[8, 1, 512]" = torch.ops.aten.view.default(mm_2, [8, 1, 512]);  mm_2 = None
        add_118: "f32[8, 1, 512]" = torch.ops.aten.add.Tensor(view_173, arg44_1);  view_173 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:81 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_174: "f32[8, 512, 256]" = torch.ops.aten.view.default(convolution_4, [8, 512, 256]);  convolution_4 = None
        permute_108: "f32[8, 256, 512]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:82 in forward, code: x = torch.cat((cls_tokens, x), dim=1)
        cat_4: "f32[8, 257, 512]" = torch.ops.aten.cat.default([add_118, permute_108], 1);  add_118 = permute_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_33 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
        getitem_178: "f32[8, 257, 1]" = var_mean_33[0]
        getitem_179: "f32[8, 257, 1]" = var_mean_33[1];  var_mean_33 = None
        add_119: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_33: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_33: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(cat_4, getitem_179);  getitem_179 = None
        mul_114: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_115: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_114, arg45_1);  mul_114 = arg45_1 = None
        add_120: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_115, arg46_1);  mul_115 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_175: "f32[2056, 512]" = torch.ops.aten.view.default(add_120, [2056, 512]);  add_120 = None
        permute_109: "f32[512, 1536]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_65: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg48_1, view_175, permute_109);  arg48_1 = view_175 = permute_109 = None
        view_176: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_65, [8, 257, 1536]);  addmm_65 = None
        view_177: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_176, [8, 257, 3, 8, 64]);  view_176 = None
        permute_110: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_177, [2, 0, 3, 1, 4]);  view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_110);  permute_110 = None
        getitem_180: "f32[8, 8, 257, 64]" = unbind_16[0]
        getitem_181: "f32[8, 8, 257, 64]" = unbind_16[1]
        getitem_182: "f32[8, 8, 257, 64]" = unbind_16[2];  unbind_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_180, getitem_181, getitem_182, None, False);  getitem_180 = getitem_181 = getitem_182 = None
        getitem_183: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_111: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
        view_178: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_111, [8, 257, 512]);  permute_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_179: "f32[2056, 512]" = torch.ops.aten.view.default(view_178, [2056, 512]);  view_178 = None
        permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_66: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg50_1, view_179, permute_112);  arg50_1 = view_179 = permute_112 = None
        view_180: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_66, [8, 257, 512]);  addmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_121: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(cat_4, view_180);  cat_4 = view_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_34 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_187: "f32[8, 257, 1]" = var_mean_34[0]
        getitem_188: "f32[8, 257, 1]" = var_mean_34[1];  var_mean_34 = None
        add_122: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_187, 1e-06);  getitem_187 = None
        rsqrt_34: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_34: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_121, getitem_188);  getitem_188 = None
        mul_116: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_117: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_116, arg51_1);  mul_116 = arg51_1 = None
        add_123: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_117, arg52_1);  mul_117 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_181: "f32[2056, 512]" = torch.ops.aten.view.default(add_123, [2056, 512]);  add_123 = None
        permute_113: "f32[512, 2048]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_67: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg54_1, view_181, permute_113);  arg54_1 = view_181 = permute_113 = None
        view_182: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_67, [8, 257, 2048]);  addmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_118: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_182, 0.5)
        mul_119: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_182, 0.7071067811865476);  view_182 = None
        erf_16: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_124: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_120: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_118, add_124);  mul_118 = add_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_183: "f32[2056, 2048]" = torch.ops.aten.view.default(mul_120, [2056, 2048]);  mul_120 = None
        permute_114: "f32[2048, 512]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_68: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg56_1, view_183, permute_114);  arg56_1 = view_183 = permute_114 = None
        view_184: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_68, [8, 257, 512]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_125: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_121, view_184);  add_121 = view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_35 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_189: "f32[8, 257, 1]" = var_mean_35[0]
        getitem_190: "f32[8, 257, 1]" = var_mean_35[1];  var_mean_35 = None
        add_126: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_189, 1e-06);  getitem_189 = None
        rsqrt_35: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_35: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_125, getitem_190);  getitem_190 = None
        mul_121: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_122: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_121, arg57_1);  mul_121 = arg57_1 = None
        add_127: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_122, arg58_1);  mul_122 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_185: "f32[2056, 512]" = torch.ops.aten.view.default(add_127, [2056, 512]);  add_127 = None
        permute_115: "f32[512, 1536]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_69: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg60_1, view_185, permute_115);  arg60_1 = view_185 = permute_115 = None
        view_186: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_69, [8, 257, 1536]);  addmm_69 = None
        view_187: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_186, [8, 257, 3, 8, 64]);  view_186 = None
        permute_116: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_187, [2, 0, 3, 1, 4]);  view_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_116);  permute_116 = None
        getitem_191: "f32[8, 8, 257, 64]" = unbind_17[0]
        getitem_192: "f32[8, 8, 257, 64]" = unbind_17[1]
        getitem_193: "f32[8, 8, 257, 64]" = unbind_17[2];  unbind_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_191, getitem_192, getitem_193, None, False);  getitem_191 = getitem_192 = getitem_193 = None
        getitem_194: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_117: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
        view_188: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_117, [8, 257, 512]);  permute_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_189: "f32[2056, 512]" = torch.ops.aten.view.default(view_188, [2056, 512]);  view_188 = None
        permute_118: "f32[512, 512]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_70: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg62_1, view_189, permute_118);  arg62_1 = view_189 = permute_118 = None
        view_190: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_70, [8, 257, 512]);  addmm_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_128: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_125, view_190);  add_125 = view_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_36 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_198: "f32[8, 257, 1]" = var_mean_36[0]
        getitem_199: "f32[8, 257, 1]" = var_mean_36[1];  var_mean_36 = None
        add_129: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_36: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_36: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_128, getitem_199);  getitem_199 = None
        mul_123: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_124: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_123, arg63_1);  mul_123 = arg63_1 = None
        add_130: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_124, arg64_1);  mul_124 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_191: "f32[2056, 512]" = torch.ops.aten.view.default(add_130, [2056, 512]);  add_130 = None
        permute_119: "f32[512, 2048]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_71: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg66_1, view_191, permute_119);  arg66_1 = view_191 = permute_119 = None
        view_192: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_71, [8, 257, 2048]);  addmm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_125: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_192, 0.5)
        mul_126: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476);  view_192 = None
        erf_17: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_131: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_127: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_125, add_131);  mul_125 = add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_193: "f32[2056, 2048]" = torch.ops.aten.view.default(mul_127, [2056, 2048]);  mul_127 = None
        permute_120: "f32[2048, 512]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_72: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg68_1, view_193, permute_120);  arg68_1 = view_193 = permute_120 = None
        view_194: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_72, [8, 257, 512]);  addmm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_132: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_128, view_194);  add_128 = view_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_37 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
        getitem_200: "f32[8, 257, 1]" = var_mean_37[0]
        getitem_201: "f32[8, 257, 1]" = var_mean_37[1];  var_mean_37 = None
        add_133: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_37: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_37: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_132, getitem_201);  getitem_201 = None
        mul_128: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_129: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_128, arg69_1);  mul_128 = arg69_1 = None
        add_134: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_129, arg70_1);  mul_129 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_195: "f32[2056, 512]" = torch.ops.aten.view.default(add_134, [2056, 512]);  add_134 = None
        permute_121: "f32[512, 1536]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_73: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg72_1, view_195, permute_121);  arg72_1 = view_195 = permute_121 = None
        view_196: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_73, [8, 257, 1536]);  addmm_73 = None
        view_197: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_196, [8, 257, 3, 8, 64]);  view_196 = None
        permute_122: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_197, [2, 0, 3, 1, 4]);  view_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_18 = torch.ops.aten.unbind.int(permute_122);  permute_122 = None
        getitem_202: "f32[8, 8, 257, 64]" = unbind_18[0]
        getitem_203: "f32[8, 8, 257, 64]" = unbind_18[1]
        getitem_204: "f32[8, 8, 257, 64]" = unbind_18[2];  unbind_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_202, getitem_203, getitem_204, None, False);  getitem_202 = getitem_203 = getitem_204 = None
        getitem_205: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_123: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
        view_198: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_123, [8, 257, 512]);  permute_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_199: "f32[2056, 512]" = torch.ops.aten.view.default(view_198, [2056, 512]);  view_198 = None
        permute_124: "f32[512, 512]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_74: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg74_1, view_199, permute_124);  arg74_1 = view_199 = permute_124 = None
        view_200: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_74, [8, 257, 512]);  addmm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_135: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_132, view_200);  add_132 = view_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_38 = torch.ops.aten.var_mean.correction(add_135, [2], correction = 0, keepdim = True)
        getitem_209: "f32[8, 257, 1]" = var_mean_38[0]
        getitem_210: "f32[8, 257, 1]" = var_mean_38[1];  var_mean_38 = None
        add_136: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_209, 1e-06);  getitem_209 = None
        rsqrt_38: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_38: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_135, getitem_210);  getitem_210 = None
        mul_130: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_131: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_130, arg75_1);  mul_130 = arg75_1 = None
        add_137: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_131, arg76_1);  mul_131 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_201: "f32[2056, 512]" = torch.ops.aten.view.default(add_137, [2056, 512]);  add_137 = None
        permute_125: "f32[512, 2048]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_75: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg78_1, view_201, permute_125);  arg78_1 = view_201 = permute_125 = None
        view_202: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_75, [8, 257, 2048]);  addmm_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_132: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_202, 0.5)
        mul_133: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_202, 0.7071067811865476);  view_202 = None
        erf_18: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_138: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_134: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_132, add_138);  mul_132 = add_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_203: "f32[2056, 2048]" = torch.ops.aten.view.default(mul_134, [2056, 2048]);  mul_134 = None
        permute_126: "f32[2048, 512]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_76: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg80_1, view_203, permute_126);  arg80_1 = view_203 = permute_126 = None
        view_204: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_76, [8, 257, 512]);  addmm_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_139: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_135, view_204);  add_135 = view_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_39 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
        getitem_211: "f32[8, 257, 1]" = var_mean_39[0]
        getitem_212: "f32[8, 257, 1]" = var_mean_39[1];  var_mean_39 = None
        add_140: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_211, 1e-06);  getitem_211 = None
        rsqrt_39: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_39: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_139, getitem_212);  getitem_212 = None
        mul_135: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_136: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_135, arg81_1);  mul_135 = arg81_1 = None
        add_141: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_136, arg82_1);  mul_136 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_205: "f32[2056, 512]" = torch.ops.aten.view.default(add_141, [2056, 512]);  add_141 = None
        permute_127: "f32[512, 1536]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_77: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg84_1, view_205, permute_127);  arg84_1 = view_205 = permute_127 = None
        view_206: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_77, [8, 257, 1536]);  addmm_77 = None
        view_207: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_206, [8, 257, 3, 8, 64]);  view_206 = None
        permute_128: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_207, [2, 0, 3, 1, 4]);  view_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_19 = torch.ops.aten.unbind.int(permute_128);  permute_128 = None
        getitem_213: "f32[8, 8, 257, 64]" = unbind_19[0]
        getitem_214: "f32[8, 8, 257, 64]" = unbind_19[1]
        getitem_215: "f32[8, 8, 257, 64]" = unbind_19[2];  unbind_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_213, getitem_214, getitem_215, None, False);  getitem_213 = getitem_214 = getitem_215 = None
        getitem_216: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_129: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
        view_208: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_129, [8, 257, 512]);  permute_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_209: "f32[2056, 512]" = torch.ops.aten.view.default(view_208, [2056, 512]);  view_208 = None
        permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_78: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg86_1, view_209, permute_130);  arg86_1 = view_209 = permute_130 = None
        view_210: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_78, [8, 257, 512]);  addmm_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_142: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_139, view_210);  add_139 = view_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_40 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
        getitem_220: "f32[8, 257, 1]" = var_mean_40[0]
        getitem_221: "f32[8, 257, 1]" = var_mean_40[1];  var_mean_40 = None
        add_143: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
        rsqrt_40: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_40: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_142, getitem_221);  getitem_221 = None
        mul_137: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_138: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_137, arg87_1);  mul_137 = arg87_1 = None
        add_144: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_138, arg88_1);  mul_138 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_211: "f32[2056, 512]" = torch.ops.aten.view.default(add_144, [2056, 512]);  add_144 = None
        permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_79: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg90_1, view_211, permute_131);  arg90_1 = view_211 = permute_131 = None
        view_212: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_79, [8, 257, 2048]);  addmm_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_139: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_212, 0.5)
        mul_140: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_212, 0.7071067811865476);  view_212 = None
        erf_19: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_145: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_141: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_139, add_145);  mul_139 = add_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_213: "f32[2056, 2048]" = torch.ops.aten.view.default(mul_141, [2056, 2048]);  mul_141 = None
        permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_80: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg92_1, view_213, permute_132);  arg92_1 = view_213 = permute_132 = None
        view_214: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_80, [8, 257, 512]);  addmm_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_146: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_142, view_214);  add_142 = view_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_41 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
        getitem_222: "f32[8, 257, 1]" = var_mean_41[0]
        getitem_223: "f32[8, 257, 1]" = var_mean_41[1];  var_mean_41 = None
        add_147: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_41: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        sub_41: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_146, getitem_223);  getitem_223 = None
        mul_142: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_143: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_142, arg93_1);  mul_142 = arg93_1 = None
        add_148: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_143, arg94_1);  mul_143 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_215: "f32[2056, 512]" = torch.ops.aten.view.default(add_148, [2056, 512]);  add_148 = None
        permute_133: "f32[512, 1536]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_81: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg96_1, view_215, permute_133);  arg96_1 = view_215 = permute_133 = None
        view_216: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_81, [8, 257, 1536]);  addmm_81 = None
        view_217: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_216, [8, 257, 3, 8, 64]);  view_216 = None
        permute_134: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_217, [2, 0, 3, 1, 4]);  view_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_20 = torch.ops.aten.unbind.int(permute_134);  permute_134 = None
        getitem_224: "f32[8, 8, 257, 64]" = unbind_20[0]
        getitem_225: "f32[8, 8, 257, 64]" = unbind_20[1]
        getitem_226: "f32[8, 8, 257, 64]" = unbind_20[2];  unbind_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_224, getitem_225, getitem_226, None, False);  getitem_224 = getitem_225 = getitem_226 = None
        getitem_227: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_135: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_227, [0, 2, 1, 3]);  getitem_227 = None
        view_218: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_135, [8, 257, 512]);  permute_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_219: "f32[2056, 512]" = torch.ops.aten.view.default(view_218, [2056, 512]);  view_218 = None
        permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_82: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg98_1, view_219, permute_136);  arg98_1 = view_219 = permute_136 = None
        view_220: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_82, [8, 257, 512]);  addmm_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_149: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_146, view_220);  add_146 = view_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_42 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
        getitem_231: "f32[8, 257, 1]" = var_mean_42[0]
        getitem_232: "f32[8, 257, 1]" = var_mean_42[1];  var_mean_42 = None
        add_150: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_231, 1e-06);  getitem_231 = None
        rsqrt_42: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_42: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_149, getitem_232);  getitem_232 = None
        mul_144: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_145: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_144, arg99_1);  mul_144 = arg99_1 = None
        add_151: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_145, arg100_1);  mul_145 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_221: "f32[2056, 512]" = torch.ops.aten.view.default(add_151, [2056, 512]);  add_151 = None
        permute_137: "f32[512, 2048]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_83: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg102_1, view_221, permute_137);  arg102_1 = view_221 = permute_137 = None
        view_222: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_83, [8, 257, 2048]);  addmm_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_146: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_222, 0.5)
        mul_147: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_222, 0.7071067811865476);  view_222 = None
        erf_20: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
        add_152: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_148: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_146, add_152);  mul_146 = add_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_223: "f32[2056, 2048]" = torch.ops.aten.view.default(mul_148, [2056, 2048]);  mul_148 = None
        permute_138: "f32[2048, 512]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_84: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg104_1, view_223, permute_138);  arg104_1 = view_223 = permute_138 = None
        view_224: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_84, [8, 257, 512]);  addmm_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_153: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_149, view_224);  add_149 = view_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_43 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_233: "f32[8, 257, 1]" = var_mean_43[0]
        getitem_234: "f32[8, 257, 1]" = var_mean_43[1];  var_mean_43 = None
        add_154: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_233, 1e-06);  getitem_233 = None
        rsqrt_43: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_43: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_153, getitem_234);  getitem_234 = None
        mul_149: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_150: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_149, arg105_1);  mul_149 = arg105_1 = None
        add_155: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_150, arg106_1);  mul_150 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_225: "f32[2056, 512]" = torch.ops.aten.view.default(add_155, [2056, 512]);  add_155 = None
        permute_139: "f32[512, 1536]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_85: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg108_1, view_225, permute_139);  arg108_1 = view_225 = permute_139 = None
        view_226: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_85, [8, 257, 1536]);  addmm_85 = None
        view_227: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_226, [8, 257, 3, 8, 64]);  view_226 = None
        permute_140: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_227, [2, 0, 3, 1, 4]);  view_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_21 = torch.ops.aten.unbind.int(permute_140);  permute_140 = None
        getitem_235: "f32[8, 8, 257, 64]" = unbind_21[0]
        getitem_236: "f32[8, 8, 257, 64]" = unbind_21[1]
        getitem_237: "f32[8, 8, 257, 64]" = unbind_21[2];  unbind_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_235, getitem_236, getitem_237, None, False);  getitem_235 = getitem_236 = getitem_237 = None
        getitem_238: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_141: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None
        view_228: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_141, [8, 257, 512]);  permute_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_229: "f32[2056, 512]" = torch.ops.aten.view.default(view_228, [2056, 512]);  view_228 = None
        permute_142: "f32[512, 512]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_86: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg110_1, view_229, permute_142);  arg110_1 = view_229 = permute_142 = None
        view_230: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_86, [8, 257, 512]);  addmm_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_156: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_153, view_230);  add_153 = view_230 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_44 = torch.ops.aten.var_mean.correction(add_156, [2], correction = 0, keepdim = True)
        getitem_242: "f32[8, 257, 1]" = var_mean_44[0]
        getitem_243: "f32[8, 257, 1]" = var_mean_44[1];  var_mean_44 = None
        add_157: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_44: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        sub_44: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_156, getitem_243);  getitem_243 = None
        mul_151: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_152: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_151, arg111_1);  mul_151 = arg111_1 = None
        add_158: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_152, arg112_1);  mul_152 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_231: "f32[2056, 512]" = torch.ops.aten.view.default(add_158, [2056, 512]);  add_158 = None
        permute_143: "f32[512, 2048]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_87: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg114_1, view_231, permute_143);  arg114_1 = view_231 = permute_143 = None
        view_232: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_87, [8, 257, 2048]);  addmm_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_153: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_232, 0.5)
        mul_154: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476);  view_232 = None
        erf_21: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_159: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_155: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_153, add_159);  mul_153 = add_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_233: "f32[2056, 2048]" = torch.ops.aten.view.default(mul_155, [2056, 2048]);  mul_155 = None
        permute_144: "f32[2048, 512]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_88: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg116_1, view_233, permute_144);  arg116_1 = view_233 = permute_144 = None
        view_234: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_88, [8, 257, 512]);  addmm_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_160: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_156, view_234);  add_156 = view_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:87 in forward, code: cls_tokens = x[:, :token_length]
        slice_19: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_160, 1, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:88 in forward, code: x = x[:, token_length:]
        slice_21: "f32[8, 256, 512]" = torch.ops.aten.slice.Tensor(add_160, 1, 1, 9223372036854775807);  add_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:89 in forward, code: x = x.transpose(1, 2).reshape(B, C, H, W)
        permute_145: "f32[8, 512, 256]" = torch.ops.aten.permute.default(slice_21, [0, 2, 1]);  slice_21 = None
        view_235: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(permute_145, [8, 512, 16, 16]);  permute_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:110 in forward, code: x = self.conv(x)
        convolution_5: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(view_235, arg117_1, arg118_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 512);  view_235 = arg117_1 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:111 in forward, code: cls_token = self.fc(cls_token)
        permute_146: "f32[512, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        view_236: "f32[8, 512]" = torch.ops.aten.view.default(slice_19, [8, 512]);  slice_19 = None
        mm_3: "f32[8, 1024]" = torch.ops.aten.mm.default(view_236, permute_146);  view_236 = permute_146 = None
        view_237: "f32[8, 1, 1024]" = torch.ops.aten.view.default(mm_3, [8, 1, 1024]);  mm_3 = None
        add_161: "f32[8, 1, 1024]" = torch.ops.aten.add.Tensor(view_237, arg120_1);  view_237 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:81 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_238: "f32[8, 1024, 64]" = torch.ops.aten.view.default(convolution_5, [8, 1024, 64]);  convolution_5 = None
        permute_147: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:82 in forward, code: x = torch.cat((cls_tokens, x), dim=1)
        cat_5: "f32[8, 65, 1024]" = torch.ops.aten.cat.default([add_161, permute_147], 1);  add_161 = permute_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_45 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
        getitem_244: "f32[8, 65, 1]" = var_mean_45[0]
        getitem_245: "f32[8, 65, 1]" = var_mean_45[1];  var_mean_45 = None
        add_162: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-06);  getitem_244 = None
        rsqrt_45: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_45: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(cat_5, getitem_245);  getitem_245 = None
        mul_156: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_157: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_156, arg121_1);  mul_156 = arg121_1 = None
        add_163: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_157, arg122_1);  mul_157 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_239: "f32[520, 1024]" = torch.ops.aten.view.default(add_163, [520, 1024]);  add_163 = None
        permute_148: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_89: "f32[520, 3072]" = torch.ops.aten.addmm.default(arg124_1, view_239, permute_148);  arg124_1 = view_239 = permute_148 = None
        view_240: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_89, [8, 65, 3072]);  addmm_89 = None
        view_241: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_240, [8, 65, 3, 16, 64]);  view_240 = None
        permute_149: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_241, [2, 0, 3, 1, 4]);  view_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_22 = torch.ops.aten.unbind.int(permute_149);  permute_149 = None
        getitem_246: "f32[8, 16, 65, 64]" = unbind_22[0]
        getitem_247: "f32[8, 16, 65, 64]" = unbind_22[1]
        getitem_248: "f32[8, 16, 65, 64]" = unbind_22[2];  unbind_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_246, getitem_247, getitem_248, None, False);  getitem_246 = getitem_247 = getitem_248 = None
        getitem_249: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_150: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_249, [0, 2, 1, 3]);  getitem_249 = None
        view_242: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_150, [8, 65, 1024]);  permute_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_243: "f32[520, 1024]" = torch.ops.aten.view.default(view_242, [520, 1024]);  view_242 = None
        permute_151: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_90: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg126_1, view_243, permute_151);  arg126_1 = view_243 = permute_151 = None
        view_244: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_90, [8, 65, 1024]);  addmm_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_164: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(cat_5, view_244);  cat_5 = view_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_46 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_253: "f32[8, 65, 1]" = var_mean_46[0]
        getitem_254: "f32[8, 65, 1]" = var_mean_46[1];  var_mean_46 = None
        add_165: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_253, 1e-06);  getitem_253 = None
        rsqrt_46: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_46: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_164, getitem_254);  getitem_254 = None
        mul_158: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_159: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_158, arg127_1);  mul_158 = arg127_1 = None
        add_166: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_159, arg128_1);  mul_159 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_245: "f32[520, 1024]" = torch.ops.aten.view.default(add_166, [520, 1024]);  add_166 = None
        permute_152: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_91: "f32[520, 4096]" = torch.ops.aten.addmm.default(arg130_1, view_245, permute_152);  arg130_1 = view_245 = permute_152 = None
        view_246: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_91, [8, 65, 4096]);  addmm_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_160: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_246, 0.5)
        mul_161: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_246, 0.7071067811865476);  view_246 = None
        erf_22: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_167: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_162: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_160, add_167);  mul_160 = add_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_247: "f32[520, 4096]" = torch.ops.aten.view.default(mul_162, [520, 4096]);  mul_162 = None
        permute_153: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_92: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg132_1, view_247, permute_153);  arg132_1 = view_247 = permute_153 = None
        view_248: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_92, [8, 65, 1024]);  addmm_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_168: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_164, view_248);  add_164 = view_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_47 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
        getitem_255: "f32[8, 65, 1]" = var_mean_47[0]
        getitem_256: "f32[8, 65, 1]" = var_mean_47[1];  var_mean_47 = None
        add_169: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_255, 1e-06);  getitem_255 = None
        rsqrt_47: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_47: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_168, getitem_256);  getitem_256 = None
        mul_163: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_164: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_163, arg133_1);  mul_163 = arg133_1 = None
        add_170: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_164, arg134_1);  mul_164 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_249: "f32[520, 1024]" = torch.ops.aten.view.default(add_170, [520, 1024]);  add_170 = None
        permute_154: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_93: "f32[520, 3072]" = torch.ops.aten.addmm.default(arg136_1, view_249, permute_154);  arg136_1 = view_249 = permute_154 = None
        view_250: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_93, [8, 65, 3072]);  addmm_93 = None
        view_251: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_250, [8, 65, 3, 16, 64]);  view_250 = None
        permute_155: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_251, [2, 0, 3, 1, 4]);  view_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_23 = torch.ops.aten.unbind.int(permute_155);  permute_155 = None
        getitem_257: "f32[8, 16, 65, 64]" = unbind_23[0]
        getitem_258: "f32[8, 16, 65, 64]" = unbind_23[1]
        getitem_259: "f32[8, 16, 65, 64]" = unbind_23[2];  unbind_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_257, getitem_258, getitem_259, None, False);  getitem_257 = getitem_258 = getitem_259 = None
        getitem_260: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_156: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        view_252: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_156, [8, 65, 1024]);  permute_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_253: "f32[520, 1024]" = torch.ops.aten.view.default(view_252, [520, 1024]);  view_252 = None
        permute_157: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_94: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_253, permute_157);  arg138_1 = view_253 = permute_157 = None
        view_254: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_94, [8, 65, 1024]);  addmm_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_171: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_168, view_254);  add_168 = view_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_48 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
        getitem_264: "f32[8, 65, 1]" = var_mean_48[0]
        getitem_265: "f32[8, 65, 1]" = var_mean_48[1];  var_mean_48 = None
        add_172: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-06);  getitem_264 = None
        rsqrt_48: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_48: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_171, getitem_265);  getitem_265 = None
        mul_165: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_166: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_165, arg139_1);  mul_165 = arg139_1 = None
        add_173: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_166, arg140_1);  mul_166 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_255: "f32[520, 1024]" = torch.ops.aten.view.default(add_173, [520, 1024]);  add_173 = None
        permute_158: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_95: "f32[520, 4096]" = torch.ops.aten.addmm.default(arg142_1, view_255, permute_158);  arg142_1 = view_255 = permute_158 = None
        view_256: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_95, [8, 65, 4096]);  addmm_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_167: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_256, 0.5)
        mul_168: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_256, 0.7071067811865476);  view_256 = None
        erf_23: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_174: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_169: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_167, add_174);  mul_167 = add_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_257: "f32[520, 4096]" = torch.ops.aten.view.default(mul_169, [520, 4096]);  mul_169 = None
        permute_159: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_96: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg144_1, view_257, permute_159);  arg144_1 = view_257 = permute_159 = None
        view_258: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_96, [8, 65, 1024]);  addmm_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_175: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_171, view_258);  add_171 = view_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_49 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
        getitem_266: "f32[8, 65, 1]" = var_mean_49[0]
        getitem_267: "f32[8, 65, 1]" = var_mean_49[1];  var_mean_49 = None
        add_176: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_49: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_49: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_175, getitem_267);  getitem_267 = None
        mul_170: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_171: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_170, arg145_1);  mul_170 = arg145_1 = None
        add_177: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_171, arg146_1);  mul_171 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_259: "f32[520, 1024]" = torch.ops.aten.view.default(add_177, [520, 1024]);  add_177 = None
        permute_160: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_97: "f32[520, 3072]" = torch.ops.aten.addmm.default(arg148_1, view_259, permute_160);  arg148_1 = view_259 = permute_160 = None
        view_260: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_97, [8, 65, 3072]);  addmm_97 = None
        view_261: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_260, [8, 65, 3, 16, 64]);  view_260 = None
        permute_161: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_261, [2, 0, 3, 1, 4]);  view_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_24 = torch.ops.aten.unbind.int(permute_161);  permute_161 = None
        getitem_268: "f32[8, 16, 65, 64]" = unbind_24[0]
        getitem_269: "f32[8, 16, 65, 64]" = unbind_24[1]
        getitem_270: "f32[8, 16, 65, 64]" = unbind_24[2];  unbind_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_268, getitem_269, getitem_270, None, False);  getitem_268 = getitem_269 = getitem_270 = None
        getitem_271: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_24[0];  _scaled_dot_product_efficient_attention_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_162: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_271, [0, 2, 1, 3]);  getitem_271 = None
        view_262: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_162, [8, 65, 1024]);  permute_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_263: "f32[520, 1024]" = torch.ops.aten.view.default(view_262, [520, 1024]);  view_262 = None
        permute_163: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_98: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg150_1, view_263, permute_163);  arg150_1 = view_263 = permute_163 = None
        view_264: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_98, [8, 65, 1024]);  addmm_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_178: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_175, view_264);  add_175 = view_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_50 = torch.ops.aten.var_mean.correction(add_178, [2], correction = 0, keepdim = True)
        getitem_275: "f32[8, 65, 1]" = var_mean_50[0]
        getitem_276: "f32[8, 65, 1]" = var_mean_50[1];  var_mean_50 = None
        add_179: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_275, 1e-06);  getitem_275 = None
        rsqrt_50: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        sub_50: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_178, getitem_276);  getitem_276 = None
        mul_172: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        mul_173: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_172, arg151_1);  mul_172 = arg151_1 = None
        add_180: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_173, arg152_1);  mul_173 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_265: "f32[520, 1024]" = torch.ops.aten.view.default(add_180, [520, 1024]);  add_180 = None
        permute_164: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_99: "f32[520, 4096]" = torch.ops.aten.addmm.default(arg154_1, view_265, permute_164);  arg154_1 = view_265 = permute_164 = None
        view_266: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_99, [8, 65, 4096]);  addmm_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_174: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_266, 0.5)
        mul_175: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_266, 0.7071067811865476);  view_266 = None
        erf_24: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
        add_181: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_176: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_174, add_181);  mul_174 = add_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_267: "f32[520, 4096]" = torch.ops.aten.view.default(mul_176, [520, 4096]);  mul_176 = None
        permute_165: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_100: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg156_1, view_267, permute_165);  arg156_1 = view_267 = permute_165 = None
        view_268: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_100, [8, 65, 1024]);  addmm_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_182: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_178, view_268);  add_178 = view_268 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_51 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
        getitem_277: "f32[8, 65, 1]" = var_mean_51[0]
        getitem_278: "f32[8, 65, 1]" = var_mean_51[1];  var_mean_51 = None
        add_183: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_277, 1e-06);  getitem_277 = None
        rsqrt_51: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        sub_51: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_182, getitem_278);  getitem_278 = None
        mul_177: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        mul_178: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_177, arg157_1);  mul_177 = arg157_1 = None
        add_184: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_178, arg158_1);  mul_178 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_269: "f32[520, 1024]" = torch.ops.aten.view.default(add_184, [520, 1024]);  add_184 = None
        permute_166: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_101: "f32[520, 3072]" = torch.ops.aten.addmm.default(arg160_1, view_269, permute_166);  arg160_1 = view_269 = permute_166 = None
        view_270: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_101, [8, 65, 3072]);  addmm_101 = None
        view_271: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_270, [8, 65, 3, 16, 64]);  view_270 = None
        permute_167: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_271, [2, 0, 3, 1, 4]);  view_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_25 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
        getitem_279: "f32[8, 16, 65, 64]" = unbind_25[0]
        getitem_280: "f32[8, 16, 65, 64]" = unbind_25[1]
        getitem_281: "f32[8, 16, 65, 64]" = unbind_25[2];  unbind_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_279, getitem_280, getitem_281, None, False);  getitem_279 = getitem_280 = getitem_281 = None
        getitem_282: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_25[0];  _scaled_dot_product_efficient_attention_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_168: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_282, [0, 2, 1, 3]);  getitem_282 = None
        view_272: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_168, [8, 65, 1024]);  permute_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_273: "f32[520, 1024]" = torch.ops.aten.view.default(view_272, [520, 1024]);  view_272 = None
        permute_169: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_102: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg162_1, view_273, permute_169);  arg162_1 = view_273 = permute_169 = None
        view_274: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_102, [8, 65, 1024]);  addmm_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_185: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_182, view_274);  add_182 = view_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_52 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_286: "f32[8, 65, 1]" = var_mean_52[0]
        getitem_287: "f32[8, 65, 1]" = var_mean_52[1];  var_mean_52 = None
        add_186: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_52: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_52: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_185, getitem_287);  getitem_287 = None
        mul_179: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        mul_180: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_179, arg163_1);  mul_179 = arg163_1 = None
        add_187: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_180, arg164_1);  mul_180 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_275: "f32[520, 1024]" = torch.ops.aten.view.default(add_187, [520, 1024]);  add_187 = None
        permute_170: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_103: "f32[520, 4096]" = torch.ops.aten.addmm.default(arg166_1, view_275, permute_170);  arg166_1 = view_275 = permute_170 = None
        view_276: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_103, [8, 65, 4096]);  addmm_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_181: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_276, 0.5)
        mul_182: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_276, 0.7071067811865476);  view_276 = None
        erf_25: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_188: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_183: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_181, add_188);  mul_181 = add_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_277: "f32[520, 4096]" = torch.ops.aten.view.default(mul_183, [520, 4096]);  mul_183 = None
        permute_171: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_104: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg168_1, view_277, permute_171);  arg168_1 = view_277 = permute_171 = None
        view_278: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_104, [8, 65, 1024]);  addmm_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_189: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_185, view_278);  add_185 = view_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:87 in forward, code: cls_tokens = x[:, :token_length]
        slice_23: "f32[8, 1, 1024]" = torch.ops.aten.slice.Tensor(add_189, 1, 0, 1);  add_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:262 in forward_features, code: cls_tokens = self.norm(cls_tokens)
        clone_82: "f32[8, 1, 1024]" = torch.ops.aten.clone.default(slice_23, memory_format = torch.contiguous_format);  slice_23 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
        getitem_288: "f32[8, 1, 1]" = var_mean_53[0]
        getitem_289: "f32[8, 1, 1]" = var_mean_53[1];  var_mean_53 = None
        add_190: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-06);  getitem_288 = None
        rsqrt_53: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_53: "f32[8, 1, 1024]" = torch.ops.aten.sub.Tensor(clone_82, getitem_289);  clone_82 = getitem_289 = None
        mul_184: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
        mul_185: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_184, arg169_1);  mul_184 = arg169_1 = None
        add_191: "f32[8, 1, 1024]" = torch.ops.aten.add.Tensor(mul_185, arg170_1);  mul_185 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:282 in forward_head, code: x = x[:, 0]
        select_1: "f32[8, 1024]" = torch.ops.aten.select.int(add_191, 1, 0);  add_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/pit.py:285 in forward_head, code: x = self.head(x)
        permute_173: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_105: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg172_1, select_1, permute_173);  arg172_1 = select_1 = permute_173 = None
        return (addmm_105,)
        