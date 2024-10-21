class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[64, 3, 4, 4]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[1, 1, 64]", arg6_1: "f32[64, 1, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[192, 64]", arg11_1: "f32[192]", arg12_1: "f32[16, 1, 3, 3]", arg13_1: "f32[16]", arg14_1: "f32[24, 1, 5, 5]", arg15_1: "f32[24]", arg16_1: "f32[24, 1, 7, 7]", arg17_1: "f32[24]", arg18_1: "f32[64, 64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[64]", arg22_1: "f32[512, 64]", arg23_1: "f32[512]", arg24_1: "f32[64, 512]", arg25_1: "f32[64]", arg26_1: "f32[64]", arg27_1: "f32[64]", arg28_1: "f32[192, 64]", arg29_1: "f32[192]", arg30_1: "f32[64, 64]", arg31_1: "f32[64]", arg32_1: "f32[64]", arg33_1: "f32[64]", arg34_1: "f32[512, 64]", arg35_1: "f32[512]", arg36_1: "f32[64, 512]", arg37_1: "f32[64]", arg38_1: "f32[128, 64, 2, 2]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[128]", arg42_1: "f32[1, 1, 128]", arg43_1: "f32[128, 1, 3, 3]", arg44_1: "f32[128]", arg45_1: "f32[128]", arg46_1: "f32[128]", arg47_1: "f32[384, 128]", arg48_1: "f32[384]", arg49_1: "f32[32, 1, 3, 3]", arg50_1: "f32[32]", arg51_1: "f32[48, 1, 5, 5]", arg52_1: "f32[48]", arg53_1: "f32[48, 1, 7, 7]", arg54_1: "f32[48]", arg55_1: "f32[128, 128]", arg56_1: "f32[128]", arg57_1: "f32[128]", arg58_1: "f32[128]", arg59_1: "f32[1024, 128]", arg60_1: "f32[1024]", arg61_1: "f32[128, 1024]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[384, 128]", arg66_1: "f32[384]", arg67_1: "f32[128, 128]", arg68_1: "f32[128]", arg69_1: "f32[128]", arg70_1: "f32[128]", arg71_1: "f32[1024, 128]", arg72_1: "f32[1024]", arg73_1: "f32[128, 1024]", arg74_1: "f32[128]", arg75_1: "f32[320, 128, 2, 2]", arg76_1: "f32[320]", arg77_1: "f32[320]", arg78_1: "f32[320]", arg79_1: "f32[1, 1, 320]", arg80_1: "f32[320, 1, 3, 3]", arg81_1: "f32[320]", arg82_1: "f32[320]", arg83_1: "f32[320]", arg84_1: "f32[960, 320]", arg85_1: "f32[960]", arg86_1: "f32[80, 1, 3, 3]", arg87_1: "f32[80]", arg88_1: "f32[120, 1, 5, 5]", arg89_1: "f32[120]", arg90_1: "f32[120, 1, 7, 7]", arg91_1: "f32[120]", arg92_1: "f32[320, 320]", arg93_1: "f32[320]", arg94_1: "f32[320]", arg95_1: "f32[320]", arg96_1: "f32[1280, 320]", arg97_1: "f32[1280]", arg98_1: "f32[320, 1280]", arg99_1: "f32[320]", arg100_1: "f32[320]", arg101_1: "f32[320]", arg102_1: "f32[960, 320]", arg103_1: "f32[960]", arg104_1: "f32[320, 320]", arg105_1: "f32[320]", arg106_1: "f32[320]", arg107_1: "f32[320]", arg108_1: "f32[1280, 320]", arg109_1: "f32[1280]", arg110_1: "f32[320, 1280]", arg111_1: "f32[320]", arg112_1: "f32[512, 320, 2, 2]", arg113_1: "f32[512]", arg114_1: "f32[512]", arg115_1: "f32[512]", arg116_1: "f32[1, 1, 512]", arg117_1: "f32[512, 1, 3, 3]", arg118_1: "f32[512]", arg119_1: "f32[512]", arg120_1: "f32[512]", arg121_1: "f32[1536, 512]", arg122_1: "f32[1536]", arg123_1: "f32[128, 1, 3, 3]", arg124_1: "f32[128]", arg125_1: "f32[192, 1, 5, 5]", arg126_1: "f32[192]", arg127_1: "f32[192, 1, 7, 7]", arg128_1: "f32[192]", arg129_1: "f32[512, 512]", arg130_1: "f32[512]", arg131_1: "f32[512]", arg132_1: "f32[512]", arg133_1: "f32[2048, 512]", arg134_1: "f32[2048]", arg135_1: "f32[512, 2048]", arg136_1: "f32[512]", arg137_1: "f32[512]", arg138_1: "f32[512]", arg139_1: "f32[1536, 512]", arg140_1: "f32[1536]", arg141_1: "f32[512, 512]", arg142_1: "f32[512]", arg143_1: "f32[512]", arg144_1: "f32[512]", arg145_1: "f32[2048, 512]", arg146_1: "f32[2048]", arg147_1: "f32[512, 2048]", arg148_1: "f32[512]", arg149_1: "f32[512]", arg150_1: "f32[512]", arg151_1: "f32[1000, 512]", arg152_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_36: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_168: "f32[8, 64, 3136]" = torch.ops.aten.reshape.default(convolution_36, [8, 64, 3136]);  convolution_36 = None
        permute_97: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        clone_65: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(clone_65, [2], correction = 0, keepdim = True)
        getitem_90: "f32[8, 3136, 1]" = var_mean_21[0]
        getitem_91: "f32[8, 3136, 1]" = var_mean_21[1];  var_mean_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:676 in insert_cls, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        expand_36: "f32[8, 1, 64]" = torch.ops.aten.expand.default(arg5_1, [8, -1, -1]);  arg5_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        sub_29: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(clone_65, getitem_91);  clone_65 = getitem_91 = None
        add_82: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_21: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_82: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_21);  sub_29 = rsqrt_21 = None
        mul_83: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_82, arg3_1);  mul_82 = arg3_1 = None
        add_83: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(mul_83, arg4_1);  mul_83 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:677 in insert_cls, code: x = torch.cat((cls_tokens, x), dim=1)
        cat_20: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([expand_36, add_83], 1);  expand_36 = add_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:154 in forward, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        slice_111: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(cat_20, 1, 0, 1)
        slice_113: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(cat_20, 1, 1, 9223372036854775807);  cat_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:157 in forward, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        permute_98: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_113, [0, 2, 1]);  slice_113 = None
        view_169: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_98, [8, 64, 56, 56]);  permute_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:158 in forward, code: x = self.proj(feat) + feat
        convolution_37: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_169, arg6_1, arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)
        add_84: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_37, view_169);  convolution_37 = view_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:159 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_170: "f32[8, 64, 3136]" = torch.ops.aten.reshape.default(add_84, [8, 64, 3136]);  add_84 = None
        permute_99: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:162 in forward, code: x = torch.cat((cls_token, x), dim=1)
        cat_21: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([slice_111, permute_99], 1);  slice_111 = permute_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_22 = torch.ops.aten.var_mean.correction(cat_21, [2], correction = 0, keepdim = True)
        getitem_92: "f32[8, 3137, 1]" = var_mean_22[0]
        getitem_93: "f32[8, 3137, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_30: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_21, getitem_93);  getitem_93 = None
        add_85: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
        rsqrt_22: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_84: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_22);  sub_30 = rsqrt_22 = None
        mul_85: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_84, arg8_1);  mul_84 = arg8_1 = None
        add_86: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_85, arg9_1);  mul_85 = arg9_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:118 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_171: "f32[25096, 64]" = torch.ops.aten.reshape.default(add_86, [25096, 64]);  add_86 = None
        permute_100: "f32[64, 192]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_33: "f32[25096, 192]" = torch.ops.aten.addmm.default(arg11_1, view_171, permute_100);  arg11_1 = view_171 = permute_100 = None
        view_172: "f32[8, 3137, 192]" = torch.ops.aten.reshape.default(addmm_33, [8, 3137, 192]);  addmm_33 = None
        view_173: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.reshape.default(view_172, [8, 3137, 3, 8, 8]);  view_172 = None
        permute_101: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_173, [2, 0, 3, 1, 4]);  view_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:119 in forward, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
        unbind_8 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
        getitem_94: "f32[8, 8, 3137, 8]" = unbind_8[0]
        getitem_95: "f32[8, 8, 3137, 8]" = unbind_8[1]
        getitem_96: "f32[8, 8, 3137, 8]" = unbind_8[2];  unbind_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:75 in forward, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_120: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(getitem_96, 2, 1, 9223372036854775807)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:77 in forward, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        permute_103: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(slice_120, [0, 1, 3, 2]);  slice_120 = None
        view_180: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_103, [8, 64, 56, 56]);  permute_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:78 in forward, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_180, [16, 24, 24], 1);  view_180 = None
        getitem_97: "f32[8, 16, 56, 56]" = split_with_sizes_8[0]
        getitem_98: "f32[8, 24, 56, 56]" = split_with_sizes_8[1]
        getitem_99: "f32[8, 24, 56, 56]" = split_with_sizes_8[2];  split_with_sizes_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_39: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_94, [8, 8, 3137, 8])
        clone_68: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_177: "f32[64, 3137, 8]" = torch.ops.aten.reshape.default(clone_68, [64, 3137, 8]);  clone_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:122 in forward, code: k_softmax = k.softmax(dim=2)
        clone_66: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(getitem_95, memory_format = torch.contiguous_format);  getitem_95 = None
        amax_8: "f32[8, 8, 1, 8]" = torch.ops.aten.amax.default(clone_66, [2], True)
        sub_31: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(clone_66, amax_8);  clone_66 = amax_8 = None
        exp_8: "f32[8, 8, 3137, 8]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_9: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(exp_8, [2], True)
        div_8: "f32[8, 8, 3137, 8]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:123 in forward, code: factor_att = k_softmax.transpose(-1, -2) @ v
        permute_102: "f32[8, 8, 8, 3137]" = torch.ops.aten.permute.default(div_8, [0, 1, 3, 2]);  div_8 = None
        expand_37: "f32[8, 8, 8, 3137]" = torch.ops.aten.expand.default(permute_102, [8, 8, 8, 3137]);  permute_102 = None
        view_174: "f32[64, 8, 3137]" = torch.ops.aten.reshape.default(expand_37, [64, 8, 3137]);  expand_37 = None
        expand_38: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_96, [8, 8, 3137, 8]);  getitem_96 = None
        clone_67: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_175: "f32[64, 3137, 8]" = torch.ops.aten.reshape.default(clone_67, [64, 3137, 8]);  clone_67 = None
        bmm_16: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(view_174, view_175);  view_174 = view_175 = None
        view_176: "f32[8, 8, 8, 8]" = torch.ops.aten.reshape.default(bmm_16, [8, 8, 8, 8]);  bmm_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_40: "f32[8, 8, 8, 8]" = torch.ops.aten.expand.default(view_176, [8, 8, 8, 8]);  view_176 = None
        view_178: "f32[64, 8, 8]" = torch.ops.aten.reshape.default(expand_40, [64, 8, 8]);  expand_40 = None
        bmm_17: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_177, view_178);  view_177 = view_178 = None
        view_179: "f32[8, 8, 3137, 8]" = torch.ops.aten.reshape.default(bmm_17, [8, 8, 3137, 8]);  bmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        mul_87: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(view_179, 0.3535533905932738);  view_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:74 in forward, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_116: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(getitem_94, 2, 1, 9223372036854775807);  getitem_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:81 in forward, code: conv_v_img_list.append(conv(v_img_list[i]))
        convolution_38: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(getitem_97, arg12_1, arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  getitem_97 = None
        convolution_39: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_98, arg14_1, arg15_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 24);  getitem_98 = None
        convolution_40: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_99, arg16_1, arg17_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 24);  getitem_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:82 in forward, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
        cat_22: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([convolution_38, convolution_39, convolution_40], 1);  convolution_38 = convolution_39 = convolution_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:83 in forward, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
        view_181: "f32[8, 8, 8, 3136]" = torch.ops.aten.reshape.default(cat_22, [8, 8, 8, 3136]);  cat_22 = None
        permute_104: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_181, [0, 1, 3, 2]);  view_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:85 in forward, code: EV_hat = q_img * conv_v_img
        mul_86: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(slice_116, permute_104);  slice_116 = permute_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "f32[8, 8, 3137, 8]" = torch.ops.aten.constant_pad_nd.default(mul_86, [0, 0, 1, 0, 0, 0], 0.0);  mul_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        add_87: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(mul_87, constant_pad_nd_8);  mul_87 = constant_pad_nd_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:131 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
        permute_105: "f32[8, 3137, 8, 8]" = torch.ops.aten.permute.default(add_87, [0, 2, 1, 3]);  add_87 = None
        clone_69: "f32[8, 3137, 8, 8]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_182: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(clone_69, [8, 3137, 64]);  clone_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_183: "f32[25096, 64]" = torch.ops.aten.reshape.default(view_182, [25096, 64]);  view_182 = None
        permute_106: "f32[64, 64]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[25096, 64]" = torch.ops.aten.mm.default(view_183, permute_106);  view_183 = permute_106 = None
        add_tensor_23: "f32[25096, 64]" = torch.ops.aten.add.Tensor(mm_default_23, arg19_1);  mm_default_23 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_184: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 3137, 64]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:215 in forward, code: x = x + self.drop_path(cur)
        add_88: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(cat_21, view_184);  cat_21 = view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_100: "f32[8, 3137, 1]" = var_mean_23[0]
        getitem_101: "f32[8, 3137, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_32: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_88, getitem_101);  getitem_101 = None
        add_89: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
        rsqrt_23: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_88: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_23);  sub_32 = rsqrt_23 = None
        mul_89: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_88, arg20_1);  mul_88 = arg20_1 = None
        add_90: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_89, arg21_1);  mul_89 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_185: "f32[25096, 64]" = torch.ops.aten.reshape.default(add_90, [25096, 64]);  add_90 = None
        permute_107: "f32[64, 512]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[25096, 512]" = torch.ops.aten.mm.default(view_185, permute_107);  view_185 = permute_107 = None
        add_tensor_22: "f32[25096, 512]" = torch.ops.aten.add.Tensor(mm_default_22, arg23_1);  mm_default_22 = arg23_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_186: "f32[8, 3137, 512]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 3137, 512]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_90: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_186, 0.5)
        mul_91: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_186, 0.7071067811865476);  view_186 = None
        erf_8: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_91: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_92: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_90, add_91);  mul_90 = add_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_187: "f32[25096, 512]" = torch.ops.aten.reshape.default(mul_92, [25096, 512]);  mul_92 = None
        permute_108: "f32[512, 64]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[25096, 64]" = torch.ops.aten.mm.default(view_187, permute_108);  view_187 = permute_108 = None
        add_tensor_21: "f32[25096, 64]" = torch.ops.aten.add.Tensor(mm_default_21, arg25_1);  mm_default_21 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_188: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 3137, 64]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:220 in forward, code: x = x + self.drop_path(cur)
        add_92: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_88, view_188);  add_88 = view_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:154 in forward, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        slice_123: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(add_92, 1, 0, 1)
        slice_125: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_92, 1, 1, 9223372036854775807);  add_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:157 in forward, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        permute_109: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_125, [0, 2, 1]);  slice_125 = None
        view_189: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_109, [8, 64, 56, 56]);  permute_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:158 in forward, code: x = self.proj(feat) + feat
        convolution_41: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(view_189, arg6_1, arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  arg6_1 = arg7_1 = None
        add_93: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(convolution_41, view_189);  convolution_41 = view_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:159 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_190: "f32[8, 64, 3136]" = torch.ops.aten.reshape.default(add_93, [8, 64, 3136]);  add_93 = None
        permute_110: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:162 in forward, code: x = torch.cat((cls_token, x), dim=1)
        cat_23: "f32[8, 3137, 64]" = torch.ops.aten.cat.default([slice_123, permute_110], 1);  slice_123 = permute_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_24 = torch.ops.aten.var_mean.correction(cat_23, [2], correction = 0, keepdim = True)
        getitem_102: "f32[8, 3137, 1]" = var_mean_24[0]
        getitem_103: "f32[8, 3137, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_33: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_23, getitem_103);  getitem_103 = None
        add_94: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
        rsqrt_24: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_93: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_24);  sub_33 = rsqrt_24 = None
        mul_94: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_93, arg26_1);  mul_93 = arg26_1 = None
        add_95: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_94, arg27_1);  mul_94 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:118 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_191: "f32[25096, 64]" = torch.ops.aten.reshape.default(add_95, [25096, 64]);  add_95 = None
        permute_111: "f32[64, 192]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_37: "f32[25096, 192]" = torch.ops.aten.addmm.default(arg29_1, view_191, permute_111);  arg29_1 = view_191 = permute_111 = None
        view_192: "f32[8, 3137, 192]" = torch.ops.aten.reshape.default(addmm_37, [8, 3137, 192]);  addmm_37 = None
        view_193: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.reshape.default(view_192, [8, 3137, 3, 8, 8]);  view_192 = None
        permute_112: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_193, [2, 0, 3, 1, 4]);  view_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:119 in forward, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
        unbind_9 = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
        getitem_104: "f32[8, 8, 3137, 8]" = unbind_9[0]
        getitem_105: "f32[8, 8, 3137, 8]" = unbind_9[1]
        getitem_106: "f32[8, 8, 3137, 8]" = unbind_9[2];  unbind_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:75 in forward, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_132: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(getitem_106, 2, 1, 9223372036854775807)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:77 in forward, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        permute_114: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(slice_132, [0, 1, 3, 2]);  slice_132 = None
        view_200: "f32[8, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_114, [8, 64, 56, 56]);  permute_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:78 in forward, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_200, [16, 24, 24], 1);  view_200 = None
        getitem_107: "f32[8, 16, 56, 56]" = split_with_sizes_9[0]
        getitem_108: "f32[8, 24, 56, 56]" = split_with_sizes_9[1]
        getitem_109: "f32[8, 24, 56, 56]" = split_with_sizes_9[2];  split_with_sizes_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_43: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_104, [8, 8, 3137, 8])
        clone_75: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_197: "f32[64, 3137, 8]" = torch.ops.aten.reshape.default(clone_75, [64, 3137, 8]);  clone_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:122 in forward, code: k_softmax = k.softmax(dim=2)
        clone_73: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(getitem_105, memory_format = torch.contiguous_format);  getitem_105 = None
        amax_9: "f32[8, 8, 1, 8]" = torch.ops.aten.amax.default(clone_73, [2], True)
        sub_34: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(clone_73, amax_9);  clone_73 = amax_9 = None
        exp_9: "f32[8, 8, 3137, 8]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_10: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(exp_9, [2], True)
        div_9: "f32[8, 8, 3137, 8]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:123 in forward, code: factor_att = k_softmax.transpose(-1, -2) @ v
        permute_113: "f32[8, 8, 8, 3137]" = torch.ops.aten.permute.default(div_9, [0, 1, 3, 2]);  div_9 = None
        expand_41: "f32[8, 8, 8, 3137]" = torch.ops.aten.expand.default(permute_113, [8, 8, 8, 3137]);  permute_113 = None
        view_194: "f32[64, 8, 3137]" = torch.ops.aten.reshape.default(expand_41, [64, 8, 3137]);  expand_41 = None
        expand_42: "f32[8, 8, 3137, 8]" = torch.ops.aten.expand.default(getitem_106, [8, 8, 3137, 8]);  getitem_106 = None
        clone_74: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_195: "f32[64, 3137, 8]" = torch.ops.aten.reshape.default(clone_74, [64, 3137, 8]);  clone_74 = None
        bmm_18: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(view_194, view_195);  view_194 = view_195 = None
        view_196: "f32[8, 8, 8, 8]" = torch.ops.aten.reshape.default(bmm_18, [8, 8, 8, 8]);  bmm_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_44: "f32[8, 8, 8, 8]" = torch.ops.aten.expand.default(view_196, [8, 8, 8, 8]);  view_196 = None
        view_198: "f32[64, 8, 8]" = torch.ops.aten.reshape.default(expand_44, [64, 8, 8]);  expand_44 = None
        bmm_19: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_197, view_198);  view_197 = view_198 = None
        view_199: "f32[8, 8, 3137, 8]" = torch.ops.aten.reshape.default(bmm_19, [8, 8, 3137, 8]);  bmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        mul_96: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(view_199, 0.3535533905932738);  view_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:74 in forward, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_128: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice.Tensor(getitem_104, 2, 1, 9223372036854775807);  getitem_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:81 in forward, code: conv_v_img_list.append(conv(v_img_list[i]))
        convolution_42: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(getitem_107, arg12_1, arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 16);  getitem_107 = arg12_1 = arg13_1 = None
        convolution_43: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_108, arg14_1, arg15_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 24);  getitem_108 = arg14_1 = arg15_1 = None
        convolution_44: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(getitem_109, arg16_1, arg17_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 24);  getitem_109 = arg16_1 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:82 in forward, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
        cat_24: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([convolution_42, convolution_43, convolution_44], 1);  convolution_42 = convolution_43 = convolution_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:83 in forward, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
        view_201: "f32[8, 8, 8, 3136]" = torch.ops.aten.reshape.default(cat_24, [8, 8, 8, 3136]);  cat_24 = None
        permute_115: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_201, [0, 1, 3, 2]);  view_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:85 in forward, code: EV_hat = q_img * conv_v_img
        mul_95: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(slice_128, permute_115);  slice_128 = permute_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_9: "f32[8, 8, 3137, 8]" = torch.ops.aten.constant_pad_nd.default(mul_95, [0, 0, 1, 0, 0, 0], 0.0);  mul_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        add_96: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(mul_96, constant_pad_nd_9);  mul_96 = constant_pad_nd_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:131 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
        permute_116: "f32[8, 3137, 8, 8]" = torch.ops.aten.permute.default(add_96, [0, 2, 1, 3]);  add_96 = None
        clone_76: "f32[8, 3137, 8, 8]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_202: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(clone_76, [8, 3137, 64]);  clone_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_203: "f32[25096, 64]" = torch.ops.aten.reshape.default(view_202, [25096, 64]);  view_202 = None
        permute_117: "f32[64, 64]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[25096, 64]" = torch.ops.aten.mm.default(view_203, permute_117);  view_203 = permute_117 = None
        add_tensor_20: "f32[25096, 64]" = torch.ops.aten.add.Tensor(mm_default_20, arg31_1);  mm_default_20 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_204: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 3137, 64]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:215 in forward, code: x = x + self.drop_path(cur)
        add_97: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(cat_23, view_204);  cat_23 = view_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_110: "f32[8, 3137, 1]" = var_mean_25[0]
        getitem_111: "f32[8, 3137, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_35: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(add_97, getitem_111);  getitem_111 = None
        add_98: "f32[8, 3137, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_25: "f32[8, 3137, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_97: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_25);  sub_35 = rsqrt_25 = None
        mul_98: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_97, arg32_1);  mul_97 = arg32_1 = None
        add_99: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(mul_98, arg33_1);  mul_98 = arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_205: "f32[25096, 64]" = torch.ops.aten.reshape.default(add_99, [25096, 64]);  add_99 = None
        permute_118: "f32[64, 512]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[25096, 512]" = torch.ops.aten.mm.default(view_205, permute_118);  view_205 = permute_118 = None
        add_tensor_19: "f32[25096, 512]" = torch.ops.aten.add.Tensor(mm_default_19, arg35_1);  mm_default_19 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_206: "f32[8, 3137, 512]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 3137, 512]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_99: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
        mul_100: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476);  view_206 = None
        erf_9: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_100: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_101: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_99, add_100);  mul_99 = add_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_207: "f32[25096, 512]" = torch.ops.aten.reshape.default(mul_101, [25096, 512]);  mul_101 = None
        permute_119: "f32[512, 64]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[25096, 64]" = torch.ops.aten.mm.default(view_207, permute_119);  view_207 = permute_119 = None
        add_tensor_18: "f32[25096, 64]" = torch.ops.aten.add.Tensor(mm_default_18, arg37_1);  mm_default_18 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_208: "f32[8, 3137, 64]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 3137, 64]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:220 in forward, code: x = x + self.drop_path(cur)
        add_101: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_97, view_208);  add_97 = view_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:683 in remove_cls, code: return x[:, 1:, :]
        slice_135: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_101, 1, 1, 9223372036854775807);  add_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:578 in forward_features, code: x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        view_209: "f32[8, 56, 56, 64]" = torch.ops.aten.reshape.default(slice_135, [8, 56, 56, 64]);  slice_135 = None
        permute_120: "f32[8, 64, 56, 56]" = torch.ops.aten.permute.default(view_209, [0, 3, 1, 2]);  view_209 = None
        clone_80: "f32[8, 64, 56, 56]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_45: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(clone_80, arg38_1, arg39_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_80 = arg38_1 = arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_210: "f32[8, 128, 784]" = torch.ops.aten.reshape.default(convolution_45, [8, 128, 784]);  convolution_45 = None
        permute_121: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        clone_81: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(clone_81, [2], correction = 0, keepdim = True)
        getitem_112: "f32[8, 784, 1]" = var_mean_26[0]
        getitem_113: "f32[8, 784, 1]" = var_mean_26[1];  var_mean_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:676 in insert_cls, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        expand_45: "f32[8, 1, 128]" = torch.ops.aten.expand.default(arg42_1, [8, -1, -1]);  arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        sub_36: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(clone_81, getitem_113);  clone_81 = getitem_113 = None
        add_102: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_26: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_102: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_26);  sub_36 = rsqrt_26 = None
        mul_103: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_102, arg40_1);  mul_102 = arg40_1 = None
        add_103: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(mul_103, arg41_1);  mul_103 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:677 in insert_cls, code: x = torch.cat((cls_tokens, x), dim=1)
        cat_25: "f32[8, 785, 128]" = torch.ops.aten.cat.default([expand_45, add_103], 1);  expand_45 = add_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:154 in forward, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        slice_138: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(cat_25, 1, 0, 1)
        slice_140: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(cat_25, 1, 1, 9223372036854775807);  cat_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:157 in forward, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        permute_122: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_140, [0, 2, 1]);  slice_140 = None
        view_211: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_122, [8, 128, 28, 28]);  permute_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:158 in forward, code: x = self.proj(feat) + feat
        convolution_46: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_211, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128)
        add_104: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_46, view_211);  convolution_46 = view_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:159 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_212: "f32[8, 128, 784]" = torch.ops.aten.reshape.default(add_104, [8, 128, 784]);  add_104 = None
        permute_123: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:162 in forward, code: x = torch.cat((cls_token, x), dim=1)
        cat_26: "f32[8, 785, 128]" = torch.ops.aten.cat.default([slice_138, permute_123], 1);  slice_138 = permute_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_27 = torch.ops.aten.var_mean.correction(cat_26, [2], correction = 0, keepdim = True)
        getitem_114: "f32[8, 785, 1]" = var_mean_27[0]
        getitem_115: "f32[8, 785, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_37: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_26, getitem_115);  getitem_115 = None
        add_105: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
        rsqrt_27: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_104: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_27);  sub_37 = rsqrt_27 = None
        mul_105: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_104, arg45_1);  mul_104 = arg45_1 = None
        add_106: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_105, arg46_1);  mul_105 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:118 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_213: "f32[6280, 128]" = torch.ops.aten.reshape.default(add_106, [6280, 128]);  add_106 = None
        permute_124: "f32[128, 384]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_41: "f32[6280, 384]" = torch.ops.aten.addmm.default(arg48_1, view_213, permute_124);  arg48_1 = view_213 = permute_124 = None
        view_214: "f32[8, 785, 384]" = torch.ops.aten.reshape.default(addmm_41, [8, 785, 384]);  addmm_41 = None
        view_215: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.reshape.default(view_214, [8, 785, 3, 8, 16]);  view_214 = None
        permute_125: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.permute.default(view_215, [2, 0, 3, 1, 4]);  view_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:119 in forward, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
        unbind_10 = torch.ops.aten.unbind.int(permute_125);  permute_125 = None
        getitem_116: "f32[8, 8, 785, 16]" = unbind_10[0]
        getitem_117: "f32[8, 8, 785, 16]" = unbind_10[1]
        getitem_118: "f32[8, 8, 785, 16]" = unbind_10[2];  unbind_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:75 in forward, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_147: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(getitem_118, 2, 1, 9223372036854775807)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:77 in forward, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        permute_127: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(slice_147, [0, 1, 3, 2]);  slice_147 = None
        view_222: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_127, [8, 128, 28, 28]);  permute_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:78 in forward, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_222, [32, 48, 48], 1);  view_222 = None
        getitem_119: "f32[8, 32, 28, 28]" = split_with_sizes_10[0]
        getitem_120: "f32[8, 48, 28, 28]" = split_with_sizes_10[1]
        getitem_121: "f32[8, 48, 28, 28]" = split_with_sizes_10[2];  split_with_sizes_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_48: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_116, [8, 8, 785, 16])
        clone_84: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_219: "f32[64, 785, 16]" = torch.ops.aten.reshape.default(clone_84, [64, 785, 16]);  clone_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:122 in forward, code: k_softmax = k.softmax(dim=2)
        clone_82: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(getitem_117, memory_format = torch.contiguous_format);  getitem_117 = None
        amax_10: "f32[8, 8, 1, 16]" = torch.ops.aten.amax.default(clone_82, [2], True)
        sub_38: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(clone_82, amax_10);  clone_82 = amax_10 = None
        exp_10: "f32[8, 8, 785, 16]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_11: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(exp_10, [2], True)
        div_10: "f32[8, 8, 785, 16]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:123 in forward, code: factor_att = k_softmax.transpose(-1, -2) @ v
        permute_126: "f32[8, 8, 16, 785]" = torch.ops.aten.permute.default(div_10, [0, 1, 3, 2]);  div_10 = None
        expand_46: "f32[8, 8, 16, 785]" = torch.ops.aten.expand.default(permute_126, [8, 8, 16, 785]);  permute_126 = None
        view_216: "f32[64, 16, 785]" = torch.ops.aten.reshape.default(expand_46, [64, 16, 785]);  expand_46 = None
        expand_47: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_118, [8, 8, 785, 16]);  getitem_118 = None
        clone_83: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_217: "f32[64, 785, 16]" = torch.ops.aten.reshape.default(clone_83, [64, 785, 16]);  clone_83 = None
        bmm_20: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(view_216, view_217);  view_216 = view_217 = None
        view_218: "f32[8, 8, 16, 16]" = torch.ops.aten.reshape.default(bmm_20, [8, 8, 16, 16]);  bmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_49: "f32[8, 8, 16, 16]" = torch.ops.aten.expand.default(view_218, [8, 8, 16, 16]);  view_218 = None
        view_220: "f32[64, 16, 16]" = torch.ops.aten.reshape.default(expand_49, [64, 16, 16]);  expand_49 = None
        bmm_21: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
        view_221: "f32[8, 8, 785, 16]" = torch.ops.aten.reshape.default(bmm_21, [8, 8, 785, 16]);  bmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        mul_107: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(view_221, 0.25);  view_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:74 in forward, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_143: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(getitem_116, 2, 1, 9223372036854775807);  getitem_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:81 in forward, code: conv_v_img_list.append(conv(v_img_list[i]))
        convolution_47: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(getitem_119, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  getitem_119 = None
        convolution_48: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_120, arg51_1, arg52_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  getitem_120 = None
        convolution_49: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_121, arg53_1, arg54_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 48);  getitem_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:82 in forward, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
        cat_27: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([convolution_47, convolution_48, convolution_49], 1);  convolution_47 = convolution_48 = convolution_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:83 in forward, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
        view_223: "f32[8, 8, 16, 784]" = torch.ops.aten.reshape.default(cat_27, [8, 8, 16, 784]);  cat_27 = None
        permute_128: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_223, [0, 1, 3, 2]);  view_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:85 in forward, code: EV_hat = q_img * conv_v_img
        mul_106: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(slice_143, permute_128);  slice_143 = permute_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_10: "f32[8, 8, 785, 16]" = torch.ops.aten.constant_pad_nd.default(mul_106, [0, 0, 1, 0, 0, 0], 0.0);  mul_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        add_107: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(mul_107, constant_pad_nd_10);  mul_107 = constant_pad_nd_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:131 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
        permute_129: "f32[8, 785, 8, 16]" = torch.ops.aten.permute.default(add_107, [0, 2, 1, 3]);  add_107 = None
        clone_85: "f32[8, 785, 8, 16]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_224: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(clone_85, [8, 785, 128]);  clone_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_225: "f32[6280, 128]" = torch.ops.aten.reshape.default(view_224, [6280, 128]);  view_224 = None
        permute_130: "f32[128, 128]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[6280, 128]" = torch.ops.aten.mm.default(view_225, permute_130);  view_225 = permute_130 = None
        add_tensor_17: "f32[6280, 128]" = torch.ops.aten.add.Tensor(mm_default_17, arg56_1);  mm_default_17 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_226: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 785, 128]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:215 in forward, code: x = x + self.drop_path(cur)
        add_108: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(cat_26, view_226);  cat_26 = view_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_122: "f32[8, 785, 1]" = var_mean_28[0]
        getitem_123: "f32[8, 785, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_39: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_108, getitem_123);  getitem_123 = None
        add_109: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
        rsqrt_28: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_108: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_28);  sub_39 = rsqrt_28 = None
        mul_109: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_108, arg57_1);  mul_108 = arg57_1 = None
        add_110: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_109, arg58_1);  mul_109 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_227: "f32[6280, 128]" = torch.ops.aten.reshape.default(add_110, [6280, 128]);  add_110 = None
        permute_131: "f32[128, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[6280, 1024]" = torch.ops.aten.mm.default(view_227, permute_131);  view_227 = permute_131 = None
        add_tensor_16: "f32[6280, 1024]" = torch.ops.aten.add.Tensor(mm_default_16, arg60_1);  mm_default_16 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_228: "f32[8, 785, 1024]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 785, 1024]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_110: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_228, 0.5)
        mul_111: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_228, 0.7071067811865476);  view_228 = None
        erf_10: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_111: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_112: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_110, add_111);  mul_110 = add_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_229: "f32[6280, 1024]" = torch.ops.aten.reshape.default(mul_112, [6280, 1024]);  mul_112 = None
        permute_132: "f32[1024, 128]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[6280, 128]" = torch.ops.aten.mm.default(view_229, permute_132);  view_229 = permute_132 = None
        add_tensor_15: "f32[6280, 128]" = torch.ops.aten.add.Tensor(mm_default_15, arg62_1);  mm_default_15 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_230: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 785, 128]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:220 in forward, code: x = x + self.drop_path(cur)
        add_112: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_108, view_230);  add_108 = view_230 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:154 in forward, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        slice_150: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_112, 1, 0, 1)
        slice_152: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_112, 1, 1, 9223372036854775807);  add_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:157 in forward, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        permute_133: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_152, [0, 2, 1]);  slice_152 = None
        view_231: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_133, [8, 128, 28, 28]);  permute_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:158 in forward, code: x = self.proj(feat) + feat
        convolution_50: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(view_231, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  arg43_1 = arg44_1 = None
        add_113: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(convolution_50, view_231);  convolution_50 = view_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:159 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_232: "f32[8, 128, 784]" = torch.ops.aten.reshape.default(add_113, [8, 128, 784]);  add_113 = None
        permute_134: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:162 in forward, code: x = torch.cat((cls_token, x), dim=1)
        cat_28: "f32[8, 785, 128]" = torch.ops.aten.cat.default([slice_150, permute_134], 1);  slice_150 = permute_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_29 = torch.ops.aten.var_mean.correction(cat_28, [2], correction = 0, keepdim = True)
        getitem_124: "f32[8, 785, 1]" = var_mean_29[0]
        getitem_125: "f32[8, 785, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_40: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_28, getitem_125);  getitem_125 = None
        add_114: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
        rsqrt_29: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_113: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_29);  sub_40 = rsqrt_29 = None
        mul_114: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_113, arg63_1);  mul_113 = arg63_1 = None
        add_115: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_114, arg64_1);  mul_114 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:118 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_233: "f32[6280, 128]" = torch.ops.aten.reshape.default(add_115, [6280, 128]);  add_115 = None
        permute_135: "f32[128, 384]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_45: "f32[6280, 384]" = torch.ops.aten.addmm.default(arg66_1, view_233, permute_135);  arg66_1 = view_233 = permute_135 = None
        view_234: "f32[8, 785, 384]" = torch.ops.aten.reshape.default(addmm_45, [8, 785, 384]);  addmm_45 = None
        view_235: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.reshape.default(view_234, [8, 785, 3, 8, 16]);  view_234 = None
        permute_136: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.permute.default(view_235, [2, 0, 3, 1, 4]);  view_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:119 in forward, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
        unbind_11 = torch.ops.aten.unbind.int(permute_136);  permute_136 = None
        getitem_126: "f32[8, 8, 785, 16]" = unbind_11[0]
        getitem_127: "f32[8, 8, 785, 16]" = unbind_11[1]
        getitem_128: "f32[8, 8, 785, 16]" = unbind_11[2];  unbind_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:75 in forward, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_159: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(getitem_128, 2, 1, 9223372036854775807)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:77 in forward, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        permute_138: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(slice_159, [0, 1, 3, 2]);  slice_159 = None
        view_242: "f32[8, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_138, [8, 128, 28, 28]);  permute_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:78 in forward, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_242, [32, 48, 48], 1);  view_242 = None
        getitem_129: "f32[8, 32, 28, 28]" = split_with_sizes_11[0]
        getitem_130: "f32[8, 48, 28, 28]" = split_with_sizes_11[1]
        getitem_131: "f32[8, 48, 28, 28]" = split_with_sizes_11[2];  split_with_sizes_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_52: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_126, [8, 8, 785, 16])
        clone_91: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_239: "f32[64, 785, 16]" = torch.ops.aten.reshape.default(clone_91, [64, 785, 16]);  clone_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:122 in forward, code: k_softmax = k.softmax(dim=2)
        clone_89: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(getitem_127, memory_format = torch.contiguous_format);  getitem_127 = None
        amax_11: "f32[8, 8, 1, 16]" = torch.ops.aten.amax.default(clone_89, [2], True)
        sub_41: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(clone_89, amax_11);  clone_89 = amax_11 = None
        exp_11: "f32[8, 8, 785, 16]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_12: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(exp_11, [2], True)
        div_11: "f32[8, 8, 785, 16]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:123 in forward, code: factor_att = k_softmax.transpose(-1, -2) @ v
        permute_137: "f32[8, 8, 16, 785]" = torch.ops.aten.permute.default(div_11, [0, 1, 3, 2]);  div_11 = None
        expand_50: "f32[8, 8, 16, 785]" = torch.ops.aten.expand.default(permute_137, [8, 8, 16, 785]);  permute_137 = None
        view_236: "f32[64, 16, 785]" = torch.ops.aten.reshape.default(expand_50, [64, 16, 785]);  expand_50 = None
        expand_51: "f32[8, 8, 785, 16]" = torch.ops.aten.expand.default(getitem_128, [8, 8, 785, 16]);  getitem_128 = None
        clone_90: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_237: "f32[64, 785, 16]" = torch.ops.aten.reshape.default(clone_90, [64, 785, 16]);  clone_90 = None
        bmm_22: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(view_236, view_237);  view_236 = view_237 = None
        view_238: "f32[8, 8, 16, 16]" = torch.ops.aten.reshape.default(bmm_22, [8, 8, 16, 16]);  bmm_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_53: "f32[8, 8, 16, 16]" = torch.ops.aten.expand.default(view_238, [8, 8, 16, 16]);  view_238 = None
        view_240: "f32[64, 16, 16]" = torch.ops.aten.reshape.default(expand_53, [64, 16, 16]);  expand_53 = None
        bmm_23: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_239, view_240);  view_239 = view_240 = None
        view_241: "f32[8, 8, 785, 16]" = torch.ops.aten.reshape.default(bmm_23, [8, 8, 785, 16]);  bmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        mul_116: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(view_241, 0.25);  view_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:74 in forward, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_155: "f32[8, 8, 784, 16]" = torch.ops.aten.slice.Tensor(getitem_126, 2, 1, 9223372036854775807);  getitem_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:81 in forward, code: conv_v_img_list.append(conv(v_img_list[i]))
        convolution_51: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(getitem_129, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  getitem_129 = arg49_1 = arg50_1 = None
        convolution_52: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_130, arg51_1, arg52_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 48);  getitem_130 = arg51_1 = arg52_1 = None
        convolution_53: "f32[8, 48, 28, 28]" = torch.ops.aten.convolution.default(getitem_131, arg53_1, arg54_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 48);  getitem_131 = arg53_1 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:82 in forward, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
        cat_29: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([convolution_51, convolution_52, convolution_53], 1);  convolution_51 = convolution_52 = convolution_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:83 in forward, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
        view_243: "f32[8, 8, 16, 784]" = torch.ops.aten.reshape.default(cat_29, [8, 8, 16, 784]);  cat_29 = None
        permute_139: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_243, [0, 1, 3, 2]);  view_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:85 in forward, code: EV_hat = q_img * conv_v_img
        mul_115: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(slice_155, permute_139);  slice_155 = permute_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_11: "f32[8, 8, 785, 16]" = torch.ops.aten.constant_pad_nd.default(mul_115, [0, 0, 1, 0, 0, 0], 0.0);  mul_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        add_116: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(mul_116, constant_pad_nd_11);  mul_116 = constant_pad_nd_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:131 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
        permute_140: "f32[8, 785, 8, 16]" = torch.ops.aten.permute.default(add_116, [0, 2, 1, 3]);  add_116 = None
        clone_92: "f32[8, 785, 8, 16]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_244: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(clone_92, [8, 785, 128]);  clone_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_245: "f32[6280, 128]" = torch.ops.aten.reshape.default(view_244, [6280, 128]);  view_244 = None
        permute_141: "f32[128, 128]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[6280, 128]" = torch.ops.aten.mm.default(view_245, permute_141);  view_245 = permute_141 = None
        add_tensor_14: "f32[6280, 128]" = torch.ops.aten.add.Tensor(mm_default_14, arg68_1);  mm_default_14 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_246: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 785, 128]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:215 in forward, code: x = x + self.drop_path(cur)
        add_117: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(cat_28, view_246);  cat_28 = view_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_30 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_132: "f32[8, 785, 1]" = var_mean_30[0]
        getitem_133: "f32[8, 785, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_42: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(add_117, getitem_133);  getitem_133 = None
        add_118: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_30: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_117: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_30);  sub_42 = rsqrt_30 = None
        mul_118: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_117, arg69_1);  mul_117 = arg69_1 = None
        add_119: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(mul_118, arg70_1);  mul_118 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_247: "f32[6280, 128]" = torch.ops.aten.reshape.default(add_119, [6280, 128]);  add_119 = None
        permute_142: "f32[128, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[6280, 1024]" = torch.ops.aten.mm.default(view_247, permute_142);  view_247 = permute_142 = None
        add_tensor_13: "f32[6280, 1024]" = torch.ops.aten.add.Tensor(mm_default_13, arg72_1);  mm_default_13 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_248: "f32[8, 785, 1024]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 785, 1024]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_119: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_248, 0.5)
        mul_120: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_248, 0.7071067811865476);  view_248 = None
        erf_11: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
        add_120: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_121: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_119, add_120);  mul_119 = add_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_249: "f32[6280, 1024]" = torch.ops.aten.reshape.default(mul_121, [6280, 1024]);  mul_121 = None
        permute_143: "f32[1024, 128]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[6280, 128]" = torch.ops.aten.mm.default(view_249, permute_143);  view_249 = permute_143 = None
        add_tensor_12: "f32[6280, 128]" = torch.ops.aten.add.Tensor(mm_default_12, arg74_1);  mm_default_12 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_250: "f32[8, 785, 128]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 785, 128]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:220 in forward, code: x = x + self.drop_path(cur)
        add_121: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_117, view_250);  add_117 = view_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:683 in remove_cls, code: return x[:, 1:, :]
        slice_162: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_121, 1, 1, 9223372036854775807);  add_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:586 in forward_features, code: x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        view_251: "f32[8, 28, 28, 128]" = torch.ops.aten.reshape.default(slice_162, [8, 28, 28, 128]);  slice_162 = None
        permute_144: "f32[8, 128, 28, 28]" = torch.ops.aten.permute.default(view_251, [0, 3, 1, 2]);  view_251 = None
        clone_96: "f32[8, 128, 28, 28]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_54: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(clone_96, arg75_1, arg76_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_96 = arg75_1 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_252: "f32[8, 320, 196]" = torch.ops.aten.reshape.default(convolution_54, [8, 320, 196]);  convolution_54 = None
        permute_145: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        clone_97: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
        getitem_134: "f32[8, 196, 1]" = var_mean_31[0]
        getitem_135: "f32[8, 196, 1]" = var_mean_31[1];  var_mean_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:676 in insert_cls, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        expand_54: "f32[8, 1, 320]" = torch.ops.aten.expand.default(arg79_1, [8, -1, -1]);  arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        sub_43: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(clone_97, getitem_135);  clone_97 = getitem_135 = None
        add_122: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
        rsqrt_31: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_122: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_31);  sub_43 = rsqrt_31 = None
        mul_123: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_122, arg77_1);  mul_122 = arg77_1 = None
        add_123: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(mul_123, arg78_1);  mul_123 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:677 in insert_cls, code: x = torch.cat((cls_tokens, x), dim=1)
        cat_30: "f32[8, 197, 320]" = torch.ops.aten.cat.default([expand_54, add_123], 1);  expand_54 = add_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:154 in forward, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        slice_165: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(cat_30, 1, 0, 1)
        slice_167: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(cat_30, 1, 1, 9223372036854775807);  cat_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:157 in forward, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        permute_146: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_167, [0, 2, 1]);  slice_167 = None
        view_253: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_146, [8, 320, 14, 14]);  permute_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:158 in forward, code: x = self.proj(feat) + feat
        convolution_55: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_253, arg80_1, arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320)
        add_124: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_55, view_253);  convolution_55 = view_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:159 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_254: "f32[8, 320, 196]" = torch.ops.aten.reshape.default(add_124, [8, 320, 196]);  add_124 = None
        permute_147: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:162 in forward, code: x = torch.cat((cls_token, x), dim=1)
        cat_31: "f32[8, 197, 320]" = torch.ops.aten.cat.default([slice_165, permute_147], 1);  slice_165 = permute_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_32 = torch.ops.aten.var_mean.correction(cat_31, [2], correction = 0, keepdim = True)
        getitem_136: "f32[8, 197, 1]" = var_mean_32[0]
        getitem_137: "f32[8, 197, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_44: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_31, getitem_137);  getitem_137 = None
        add_125: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
        rsqrt_32: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_124: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_32);  sub_44 = rsqrt_32 = None
        mul_125: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_124, arg82_1);  mul_124 = arg82_1 = None
        add_126: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_125, arg83_1);  mul_125 = arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:118 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_255: "f32[1576, 320]" = torch.ops.aten.reshape.default(add_126, [1576, 320]);  add_126 = None
        permute_148: "f32[320, 960]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_49: "f32[1576, 960]" = torch.ops.aten.addmm.default(arg85_1, view_255, permute_148);  arg85_1 = view_255 = permute_148 = None
        view_256: "f32[8, 197, 960]" = torch.ops.aten.reshape.default(addmm_49, [8, 197, 960]);  addmm_49 = None
        view_257: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.reshape.default(view_256, [8, 197, 3, 8, 40]);  view_256 = None
        permute_149: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.permute.default(view_257, [2, 0, 3, 1, 4]);  view_257 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:119 in forward, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
        unbind_12 = torch.ops.aten.unbind.int(permute_149);  permute_149 = None
        getitem_138: "f32[8, 8, 197, 40]" = unbind_12[0]
        getitem_139: "f32[8, 8, 197, 40]" = unbind_12[1]
        getitem_140: "f32[8, 8, 197, 40]" = unbind_12[2];  unbind_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:75 in forward, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_174: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(getitem_140, 2, 1, 9223372036854775807)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:77 in forward, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        permute_151: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(slice_174, [0, 1, 3, 2]);  slice_174 = None
        view_264: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_151, [8, 320, 14, 14]);  permute_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:78 in forward, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(view_264, [80, 120, 120], 1);  view_264 = None
        getitem_141: "f32[8, 80, 14, 14]" = split_with_sizes_12[0]
        getitem_142: "f32[8, 120, 14, 14]" = split_with_sizes_12[1]
        getitem_143: "f32[8, 120, 14, 14]" = split_with_sizes_12[2];  split_with_sizes_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_57: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_138, [8, 8, 197, 40])
        clone_100: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_261: "f32[64, 197, 40]" = torch.ops.aten.reshape.default(clone_100, [64, 197, 40]);  clone_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:122 in forward, code: k_softmax = k.softmax(dim=2)
        clone_98: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(getitem_139, memory_format = torch.contiguous_format);  getitem_139 = None
        amax_12: "f32[8, 8, 1, 40]" = torch.ops.aten.amax.default(clone_98, [2], True)
        sub_45: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(clone_98, amax_12);  clone_98 = amax_12 = None
        exp_12: "f32[8, 8, 197, 40]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_13: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(exp_12, [2], True)
        div_12: "f32[8, 8, 197, 40]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:123 in forward, code: factor_att = k_softmax.transpose(-1, -2) @ v
        permute_150: "f32[8, 8, 40, 197]" = torch.ops.aten.permute.default(div_12, [0, 1, 3, 2]);  div_12 = None
        expand_55: "f32[8, 8, 40, 197]" = torch.ops.aten.expand.default(permute_150, [8, 8, 40, 197]);  permute_150 = None
        view_258: "f32[64, 40, 197]" = torch.ops.aten.reshape.default(expand_55, [64, 40, 197]);  expand_55 = None
        expand_56: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_140, [8, 8, 197, 40]);  getitem_140 = None
        clone_99: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_259: "f32[64, 197, 40]" = torch.ops.aten.reshape.default(clone_99, [64, 197, 40]);  clone_99 = None
        bmm_24: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(view_258, view_259);  view_258 = view_259 = None
        view_260: "f32[8, 8, 40, 40]" = torch.ops.aten.reshape.default(bmm_24, [8, 8, 40, 40]);  bmm_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_58: "f32[8, 8, 40, 40]" = torch.ops.aten.expand.default(view_260, [8, 8, 40, 40]);  view_260 = None
        view_262: "f32[64, 40, 40]" = torch.ops.aten.reshape.default(expand_58, [64, 40, 40]);  expand_58 = None
        bmm_25: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_261, view_262);  view_261 = view_262 = None
        view_263: "f32[8, 8, 197, 40]" = torch.ops.aten.reshape.default(bmm_25, [8, 8, 197, 40]);  bmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        mul_127: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(view_263, 0.15811388300841897);  view_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:74 in forward, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_170: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(getitem_138, 2, 1, 9223372036854775807);  getitem_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:81 in forward, code: conv_v_img_list.append(conv(v_img_list[i]))
        convolution_56: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_141, arg86_1, arg87_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  getitem_141 = None
        convolution_57: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_142, arg88_1, arg89_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_142 = None
        convolution_58: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_143, arg90_1, arg91_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:82 in forward, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
        cat_32: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([convolution_56, convolution_57, convolution_58], 1);  convolution_56 = convolution_57 = convolution_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:83 in forward, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
        view_265: "f32[8, 8, 40, 196]" = torch.ops.aten.reshape.default(cat_32, [8, 8, 40, 196]);  cat_32 = None
        permute_152: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_265, [0, 1, 3, 2]);  view_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:85 in forward, code: EV_hat = q_img * conv_v_img
        mul_126: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(slice_170, permute_152);  slice_170 = permute_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_12: "f32[8, 8, 197, 40]" = torch.ops.aten.constant_pad_nd.default(mul_126, [0, 0, 1, 0, 0, 0], 0.0);  mul_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        add_127: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(mul_127, constant_pad_nd_12);  mul_127 = constant_pad_nd_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:131 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
        permute_153: "f32[8, 197, 8, 40]" = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None
        clone_101: "f32[8, 197, 8, 40]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_266: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(clone_101, [8, 197, 320]);  clone_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_267: "f32[1576, 320]" = torch.ops.aten.reshape.default(view_266, [1576, 320]);  view_266 = None
        permute_154: "f32[320, 320]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[1576, 320]" = torch.ops.aten.mm.default(view_267, permute_154);  view_267 = permute_154 = None
        add_tensor_11: "f32[1576, 320]" = torch.ops.aten.add.Tensor(mm_default_11, arg93_1);  mm_default_11 = arg93_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_268: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 197, 320]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:215 in forward, code: x = x + self.drop_path(cur)
        add_128: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(cat_31, view_268);  cat_31 = view_268 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_33 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_144: "f32[8, 197, 1]" = var_mean_33[0]
        getitem_145: "f32[8, 197, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_46: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_128, getitem_145);  getitem_145 = None
        add_129: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_33: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_128: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_33);  sub_46 = rsqrt_33 = None
        mul_129: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_128, arg94_1);  mul_128 = arg94_1 = None
        add_130: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_129, arg95_1);  mul_129 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_269: "f32[1576, 320]" = torch.ops.aten.reshape.default(add_130, [1576, 320]);  add_130 = None
        permute_155: "f32[320, 1280]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[1576, 1280]" = torch.ops.aten.mm.default(view_269, permute_155);  view_269 = permute_155 = None
        add_tensor_10: "f32[1576, 1280]" = torch.ops.aten.add.Tensor(mm_default_10, arg97_1);  mm_default_10 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_270: "f32[8, 197, 1280]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 197, 1280]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_130: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_270, 0.5)
        mul_131: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_270, 0.7071067811865476);  view_270 = None
        erf_12: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_131: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_132: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_130, add_131);  mul_130 = add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_271: "f32[1576, 1280]" = torch.ops.aten.reshape.default(mul_132, [1576, 1280]);  mul_132 = None
        permute_156: "f32[1280, 320]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[1576, 320]" = torch.ops.aten.mm.default(view_271, permute_156);  view_271 = permute_156 = None
        add_tensor_9: "f32[1576, 320]" = torch.ops.aten.add.Tensor(mm_default_9, arg99_1);  mm_default_9 = arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_272: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 197, 320]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:220 in forward, code: x = x + self.drop_path(cur)
        add_132: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_128, view_272);  add_128 = view_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:154 in forward, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        slice_177: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(add_132, 1, 0, 1)
        slice_179: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_132, 1, 1, 9223372036854775807);  add_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:157 in forward, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        permute_157: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_179, [0, 2, 1]);  slice_179 = None
        view_273: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_157, [8, 320, 14, 14]);  permute_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:158 in forward, code: x = self.proj(feat) + feat
        convolution_59: "f32[8, 320, 14, 14]" = torch.ops.aten.convolution.default(view_273, arg80_1, arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 320);  arg80_1 = arg81_1 = None
        add_133: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_59, view_273);  convolution_59 = view_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:159 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_274: "f32[8, 320, 196]" = torch.ops.aten.reshape.default(add_133, [8, 320, 196]);  add_133 = None
        permute_158: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:162 in forward, code: x = torch.cat((cls_token, x), dim=1)
        cat_33: "f32[8, 197, 320]" = torch.ops.aten.cat.default([slice_177, permute_158], 1);  slice_177 = permute_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_34 = torch.ops.aten.var_mean.correction(cat_33, [2], correction = 0, keepdim = True)
        getitem_146: "f32[8, 197, 1]" = var_mean_34[0]
        getitem_147: "f32[8, 197, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_47: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_33, getitem_147);  getitem_147 = None
        add_134: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
        rsqrt_34: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_133: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_34);  sub_47 = rsqrt_34 = None
        mul_134: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_133, arg100_1);  mul_133 = arg100_1 = None
        add_135: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_134, arg101_1);  mul_134 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:118 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_275: "f32[1576, 320]" = torch.ops.aten.reshape.default(add_135, [1576, 320]);  add_135 = None
        permute_159: "f32[320, 960]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_53: "f32[1576, 960]" = torch.ops.aten.addmm.default(arg103_1, view_275, permute_159);  arg103_1 = view_275 = permute_159 = None
        view_276: "f32[8, 197, 960]" = torch.ops.aten.reshape.default(addmm_53, [8, 197, 960]);  addmm_53 = None
        view_277: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.reshape.default(view_276, [8, 197, 3, 8, 40]);  view_276 = None
        permute_160: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.permute.default(view_277, [2, 0, 3, 1, 4]);  view_277 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:119 in forward, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
        unbind_13 = torch.ops.aten.unbind.int(permute_160);  permute_160 = None
        getitem_148: "f32[8, 8, 197, 40]" = unbind_13[0]
        getitem_149: "f32[8, 8, 197, 40]" = unbind_13[1]
        getitem_150: "f32[8, 8, 197, 40]" = unbind_13[2];  unbind_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:75 in forward, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_186: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(getitem_150, 2, 1, 9223372036854775807)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:77 in forward, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        permute_162: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(slice_186, [0, 1, 3, 2]);  slice_186 = None
        view_284: "f32[8, 320, 14, 14]" = torch.ops.aten.reshape.default(permute_162, [8, 320, 14, 14]);  permute_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:78 in forward, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(view_284, [80, 120, 120], 1);  view_284 = None
        getitem_151: "f32[8, 80, 14, 14]" = split_with_sizes_13[0]
        getitem_152: "f32[8, 120, 14, 14]" = split_with_sizes_13[1]
        getitem_153: "f32[8, 120, 14, 14]" = split_with_sizes_13[2];  split_with_sizes_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_61: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_148, [8, 8, 197, 40])
        clone_107: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_281: "f32[64, 197, 40]" = torch.ops.aten.reshape.default(clone_107, [64, 197, 40]);  clone_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:122 in forward, code: k_softmax = k.softmax(dim=2)
        clone_105: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(getitem_149, memory_format = torch.contiguous_format);  getitem_149 = None
        amax_13: "f32[8, 8, 1, 40]" = torch.ops.aten.amax.default(clone_105, [2], True)
        sub_48: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(clone_105, amax_13);  clone_105 = amax_13 = None
        exp_13: "f32[8, 8, 197, 40]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
        sum_14: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(exp_13, [2], True)
        div_13: "f32[8, 8, 197, 40]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:123 in forward, code: factor_att = k_softmax.transpose(-1, -2) @ v
        permute_161: "f32[8, 8, 40, 197]" = torch.ops.aten.permute.default(div_13, [0, 1, 3, 2]);  div_13 = None
        expand_59: "f32[8, 8, 40, 197]" = torch.ops.aten.expand.default(permute_161, [8, 8, 40, 197]);  permute_161 = None
        view_278: "f32[64, 40, 197]" = torch.ops.aten.reshape.default(expand_59, [64, 40, 197]);  expand_59 = None
        expand_60: "f32[8, 8, 197, 40]" = torch.ops.aten.expand.default(getitem_150, [8, 8, 197, 40]);  getitem_150 = None
        clone_106: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_279: "f32[64, 197, 40]" = torch.ops.aten.reshape.default(clone_106, [64, 197, 40]);  clone_106 = None
        bmm_26: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(view_278, view_279);  view_278 = view_279 = None
        view_280: "f32[8, 8, 40, 40]" = torch.ops.aten.reshape.default(bmm_26, [8, 8, 40, 40]);  bmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_62: "f32[8, 8, 40, 40]" = torch.ops.aten.expand.default(view_280, [8, 8, 40, 40]);  view_280 = None
        view_282: "f32[64, 40, 40]" = torch.ops.aten.reshape.default(expand_62, [64, 40, 40]);  expand_62 = None
        bmm_27: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_281, view_282);  view_281 = view_282 = None
        view_283: "f32[8, 8, 197, 40]" = torch.ops.aten.reshape.default(bmm_27, [8, 8, 197, 40]);  bmm_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        mul_136: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(view_283, 0.15811388300841897);  view_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:74 in forward, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_182: "f32[8, 8, 196, 40]" = torch.ops.aten.slice.Tensor(getitem_148, 2, 1, 9223372036854775807);  getitem_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:81 in forward, code: conv_v_img_list.append(conv(v_img_list[i]))
        convolution_60: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_151, arg86_1, arg87_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 80);  getitem_151 = arg86_1 = arg87_1 = None
        convolution_61: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_152, arg88_1, arg89_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 120);  getitem_152 = arg88_1 = arg89_1 = None
        convolution_62: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_153, arg90_1, arg91_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 120);  getitem_153 = arg90_1 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:82 in forward, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
        cat_34: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([convolution_60, convolution_61, convolution_62], 1);  convolution_60 = convolution_61 = convolution_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:83 in forward, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
        view_285: "f32[8, 8, 40, 196]" = torch.ops.aten.reshape.default(cat_34, [8, 8, 40, 196]);  cat_34 = None
        permute_163: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_285, [0, 1, 3, 2]);  view_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:85 in forward, code: EV_hat = q_img * conv_v_img
        mul_135: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(slice_182, permute_163);  slice_182 = permute_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_13: "f32[8, 8, 197, 40]" = torch.ops.aten.constant_pad_nd.default(mul_135, [0, 0, 1, 0, 0, 0], 0.0);  mul_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        add_136: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(mul_136, constant_pad_nd_13);  mul_136 = constant_pad_nd_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:131 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
        permute_164: "f32[8, 197, 8, 40]" = torch.ops.aten.permute.default(add_136, [0, 2, 1, 3]);  add_136 = None
        clone_108: "f32[8, 197, 8, 40]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_286: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(clone_108, [8, 197, 320]);  clone_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_287: "f32[1576, 320]" = torch.ops.aten.reshape.default(view_286, [1576, 320]);  view_286 = None
        permute_165: "f32[320, 320]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[1576, 320]" = torch.ops.aten.mm.default(view_287, permute_165);  view_287 = permute_165 = None
        add_tensor_8: "f32[1576, 320]" = torch.ops.aten.add.Tensor(mm_default_8, arg105_1);  mm_default_8 = arg105_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_288: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 197, 320]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:215 in forward, code: x = x + self.drop_path(cur)
        add_137: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(cat_33, view_288);  cat_33 = view_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_35 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_154: "f32[8, 197, 1]" = var_mean_35[0]
        getitem_155: "f32[8, 197, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_49: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(add_137, getitem_155);  getitem_155 = None
        add_138: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_35: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_137: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_35);  sub_49 = rsqrt_35 = None
        mul_138: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_137, arg106_1);  mul_137 = arg106_1 = None
        add_139: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(mul_138, arg107_1);  mul_138 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_289: "f32[1576, 320]" = torch.ops.aten.reshape.default(add_139, [1576, 320]);  add_139 = None
        permute_166: "f32[320, 1280]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[1576, 1280]" = torch.ops.aten.mm.default(view_289, permute_166);  view_289 = permute_166 = None
        add_tensor_7: "f32[1576, 1280]" = torch.ops.aten.add.Tensor(mm_default_7, arg109_1);  mm_default_7 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_290: "f32[8, 197, 1280]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 197, 1280]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_139: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_290, 0.5)
        mul_140: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476);  view_290 = None
        erf_13: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_140: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_141: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_139, add_140);  mul_139 = add_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_291: "f32[1576, 1280]" = torch.ops.aten.reshape.default(mul_141, [1576, 1280]);  mul_141 = None
        permute_167: "f32[1280, 320]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[1576, 320]" = torch.ops.aten.mm.default(view_291, permute_167);  view_291 = permute_167 = None
        add_tensor_6: "f32[1576, 320]" = torch.ops.aten.add.Tensor(mm_default_6, arg111_1);  mm_default_6 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_292: "f32[8, 197, 320]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 197, 320]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:220 in forward, code: x = x + self.drop_path(cur)
        add_141: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_137, view_292);  add_137 = view_292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:683 in remove_cls, code: return x[:, 1:, :]
        slice_189: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_141, 1, 1, 9223372036854775807);  add_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:594 in forward_features, code: x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        view_293: "f32[8, 14, 14, 320]" = torch.ops.aten.reshape.default(slice_189, [8, 14, 14, 320]);  slice_189 = None
        permute_168: "f32[8, 320, 14, 14]" = torch.ops.aten.permute.default(view_293, [0, 3, 1, 2]);  view_293 = None
        clone_112: "f32[8, 320, 14, 14]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_63: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(clone_112, arg112_1, arg113_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  clone_112 = arg112_1 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_294: "f32[8, 512, 49]" = torch.ops.aten.reshape.default(convolution_63, [8, 512, 49]);  convolution_63 = None
        permute_169: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_294, [0, 2, 1]);  view_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        clone_113: "f32[8, 49, 512]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(clone_113, [2], correction = 0, keepdim = True)
        getitem_156: "f32[8, 49, 1]" = var_mean_36[0]
        getitem_157: "f32[8, 49, 1]" = var_mean_36[1];  var_mean_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:676 in insert_cls, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        expand_63: "f32[8, 1, 512]" = torch.ops.aten.expand.default(arg116_1, [8, -1, -1]);  arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        sub_50: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(clone_113, getitem_157);  clone_113 = getitem_157 = None
        add_142: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_36: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_142: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_36);  sub_50 = rsqrt_36 = None
        mul_143: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_142, arg114_1);  mul_142 = arg114_1 = None
        add_143: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_143, arg115_1);  mul_143 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:677 in insert_cls, code: x = torch.cat((cls_tokens, x), dim=1)
        cat_35: "f32[8, 50, 512]" = torch.ops.aten.cat.default([expand_63, add_143], 1);  expand_63 = add_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:154 in forward, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        slice_192: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(cat_35, 1, 0, 1)
        slice_194: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(cat_35, 1, 1, 9223372036854775807);  cat_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:157 in forward, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        permute_170: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_194, [0, 2, 1]);  slice_194 = None
        view_295: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_170, [8, 512, 7, 7]);  permute_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:158 in forward, code: x = self.proj(feat) + feat
        convolution_64: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_295, arg117_1, arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512)
        add_144: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_64, view_295);  convolution_64 = view_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:159 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_296: "f32[8, 512, 49]" = torch.ops.aten.reshape.default(add_144, [8, 512, 49]);  add_144 = None
        permute_171: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:162 in forward, code: x = torch.cat((cls_token, x), dim=1)
        cat_36: "f32[8, 50, 512]" = torch.ops.aten.cat.default([slice_192, permute_171], 1);  slice_192 = permute_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_37 = torch.ops.aten.var_mean.correction(cat_36, [2], correction = 0, keepdim = True)
        getitem_158: "f32[8, 50, 1]" = var_mean_37[0]
        getitem_159: "f32[8, 50, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_51: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_36, getitem_159);  getitem_159 = None
        add_145: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_37: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        mul_144: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_37);  sub_51 = rsqrt_37 = None
        mul_145: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_144, arg119_1);  mul_144 = arg119_1 = None
        add_146: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_145, arg120_1);  mul_145 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:118 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_297: "f32[400, 512]" = torch.ops.aten.reshape.default(add_146, [400, 512]);  add_146 = None
        permute_172: "f32[512, 1536]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_57: "f32[400, 1536]" = torch.ops.aten.addmm.default(arg122_1, view_297, permute_172);  arg122_1 = view_297 = permute_172 = None
        view_298: "f32[8, 50, 1536]" = torch.ops.aten.reshape.default(addmm_57, [8, 50, 1536]);  addmm_57 = None
        view_299: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.reshape.default(view_298, [8, 50, 3, 8, 64]);  view_298 = None
        permute_173: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.permute.default(view_299, [2, 0, 3, 1, 4]);  view_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:119 in forward, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
        unbind_14 = torch.ops.aten.unbind.int(permute_173);  permute_173 = None
        getitem_160: "f32[8, 8, 50, 64]" = unbind_14[0]
        getitem_161: "f32[8, 8, 50, 64]" = unbind_14[1]
        getitem_162: "f32[8, 8, 50, 64]" = unbind_14[2];  unbind_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:75 in forward, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_201: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(getitem_162, 2, 1, 9223372036854775807)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:77 in forward, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        permute_175: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(slice_201, [0, 1, 3, 2]);  slice_201 = None
        view_306: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_175, [8, 512, 7, 7]);  permute_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:78 in forward, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(view_306, [128, 192, 192], 1);  view_306 = None
        getitem_163: "f32[8, 128, 7, 7]" = split_with_sizes_14[0]
        getitem_164: "f32[8, 192, 7, 7]" = split_with_sizes_14[1]
        getitem_165: "f32[8, 192, 7, 7]" = split_with_sizes_14[2];  split_with_sizes_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_66: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_160, [8, 8, 50, 64])
        clone_116: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_303: "f32[64, 50, 64]" = torch.ops.aten.reshape.default(clone_116, [64, 50, 64]);  clone_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:122 in forward, code: k_softmax = k.softmax(dim=2)
        clone_114: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(getitem_161, memory_format = torch.contiguous_format);  getitem_161 = None
        amax_14: "f32[8, 8, 1, 64]" = torch.ops.aten.amax.default(clone_114, [2], True)
        sub_52: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(clone_114, amax_14);  clone_114 = amax_14 = None
        exp_14: "f32[8, 8, 50, 64]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_15: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_14, [2], True)
        div_14: "f32[8, 8, 50, 64]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:123 in forward, code: factor_att = k_softmax.transpose(-1, -2) @ v
        permute_174: "f32[8, 8, 64, 50]" = torch.ops.aten.permute.default(div_14, [0, 1, 3, 2]);  div_14 = None
        expand_64: "f32[8, 8, 64, 50]" = torch.ops.aten.expand.default(permute_174, [8, 8, 64, 50]);  permute_174 = None
        view_300: "f32[64, 64, 50]" = torch.ops.aten.reshape.default(expand_64, [64, 64, 50]);  expand_64 = None
        expand_65: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_162, [8, 8, 50, 64]);  getitem_162 = None
        clone_115: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_301: "f32[64, 50, 64]" = torch.ops.aten.reshape.default(clone_115, [64, 50, 64]);  clone_115 = None
        bmm_28: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(view_300, view_301);  view_300 = view_301 = None
        view_302: "f32[8, 8, 64, 64]" = torch.ops.aten.reshape.default(bmm_28, [8, 8, 64, 64]);  bmm_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_67: "f32[8, 8, 64, 64]" = torch.ops.aten.expand.default(view_302, [8, 8, 64, 64]);  view_302 = None
        view_304: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(expand_67, [64, 64, 64]);  expand_67 = None
        bmm_29: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_303, view_304);  view_303 = view_304 = None
        view_305: "f32[8, 8, 50, 64]" = torch.ops.aten.reshape.default(bmm_29, [8, 8, 50, 64]);  bmm_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        mul_147: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(view_305, 0.125);  view_305 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:74 in forward, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_197: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(getitem_160, 2, 1, 9223372036854775807);  getitem_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:81 in forward, code: conv_v_img_list.append(conv(v_img_list[i]))
        convolution_65: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(getitem_163, arg123_1, arg124_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  getitem_163 = None
        convolution_66: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_164, arg125_1, arg126_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 192);  getitem_164 = None
        convolution_67: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_165, arg127_1, arg128_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  getitem_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:82 in forward, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
        cat_37: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([convolution_65, convolution_66, convolution_67], 1);  convolution_65 = convolution_66 = convolution_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:83 in forward, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
        view_307: "f32[8, 8, 64, 49]" = torch.ops.aten.reshape.default(cat_37, [8, 8, 64, 49]);  cat_37 = None
        permute_176: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_307, [0, 1, 3, 2]);  view_307 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:85 in forward, code: EV_hat = q_img * conv_v_img
        mul_146: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(slice_197, permute_176);  slice_197 = permute_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_14: "f32[8, 8, 50, 64]" = torch.ops.aten.constant_pad_nd.default(mul_146, [0, 0, 1, 0, 0, 0], 0.0);  mul_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        add_147: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(mul_147, constant_pad_nd_14);  mul_147 = constant_pad_nd_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:131 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
        permute_177: "f32[8, 50, 8, 64]" = torch.ops.aten.permute.default(add_147, [0, 2, 1, 3]);  add_147 = None
        clone_117: "f32[8, 50, 8, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_308: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(clone_117, [8, 50, 512]);  clone_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_309: "f32[400, 512]" = torch.ops.aten.reshape.default(view_308, [400, 512]);  view_308 = None
        permute_178: "f32[512, 512]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[400, 512]" = torch.ops.aten.mm.default(view_309, permute_178);  view_309 = permute_178 = None
        add_tensor_5: "f32[400, 512]" = torch.ops.aten.add.Tensor(mm_default_5, arg130_1);  mm_default_5 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_310: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 50, 512]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:215 in forward, code: x = x + self.drop_path(cur)
        add_148: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(cat_36, view_310);  cat_36 = view_310 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
        getitem_166: "f32[8, 50, 1]" = var_mean_38[0]
        getitem_167: "f32[8, 50, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_53: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_148, getitem_167);  getitem_167 = None
        add_149: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-06);  getitem_166 = None
        rsqrt_38: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_148: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_38);  sub_53 = rsqrt_38 = None
        mul_149: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_148, arg131_1);  mul_148 = arg131_1 = None
        add_150: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_149, arg132_1);  mul_149 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_311: "f32[400, 512]" = torch.ops.aten.reshape.default(add_150, [400, 512]);  add_150 = None
        permute_179: "f32[512, 2048]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[400, 2048]" = torch.ops.aten.mm.default(view_311, permute_179);  view_311 = permute_179 = None
        add_tensor_4: "f32[400, 2048]" = torch.ops.aten.add.Tensor(mm_default_4, arg134_1);  mm_default_4 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_312: "f32[8, 50, 2048]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 50, 2048]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_150: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_312, 0.5)
        mul_151: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_312, 0.7071067811865476);  view_312 = None
        erf_14: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_151: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_152: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_150, add_151);  mul_150 = add_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_313: "f32[400, 2048]" = torch.ops.aten.reshape.default(mul_152, [400, 2048]);  mul_152 = None
        permute_180: "f32[2048, 512]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[400, 512]" = torch.ops.aten.mm.default(view_313, permute_180);  view_313 = permute_180 = None
        add_tensor_3: "f32[400, 512]" = torch.ops.aten.add.Tensor(mm_default_3, arg136_1);  mm_default_3 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_314: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 50, 512]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:220 in forward, code: x = x + self.drop_path(cur)
        add_152: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_148, view_314);  add_148 = view_314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:154 in forward, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
        slice_204: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_152, 1, 0, 1)
        slice_206: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(add_152, 1, 1, 9223372036854775807);  add_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:157 in forward, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        permute_181: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_206, [0, 2, 1]);  slice_206 = None
        view_315: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_181, [8, 512, 7, 7]);  permute_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:158 in forward, code: x = self.proj(feat) + feat
        convolution_68: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(view_315, arg117_1, arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 512);  arg117_1 = arg118_1 = None
        add_153: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(convolution_68, view_315);  convolution_68 = view_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:159 in forward, code: x = x.flatten(2).transpose(1, 2)
        view_316: "f32[8, 512, 49]" = torch.ops.aten.reshape.default(add_153, [8, 512, 49]);  add_153 = None
        permute_182: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_316, [0, 2, 1]);  view_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:162 in forward, code: x = torch.cat((cls_token, x), dim=1)
        cat_38: "f32[8, 50, 512]" = torch.ops.aten.cat.default([slice_204, permute_182], 1);  slice_204 = permute_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_39 = torch.ops.aten.var_mean.correction(cat_38, [2], correction = 0, keepdim = True)
        getitem_168: "f32[8, 50, 1]" = var_mean_39[0]
        getitem_169: "f32[8, 50, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_54: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_38, getitem_169);  getitem_169 = None
        add_154: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
        rsqrt_39: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        mul_153: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_39);  sub_54 = rsqrt_39 = None
        mul_154: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_153, arg137_1);  mul_153 = arg137_1 = None
        add_155: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_154, arg138_1);  mul_154 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:118 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        view_317: "f32[400, 512]" = torch.ops.aten.reshape.default(add_155, [400, 512]);  add_155 = None
        permute_183: "f32[512, 1536]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_61: "f32[400, 1536]" = torch.ops.aten.addmm.default(arg140_1, view_317, permute_183);  arg140_1 = view_317 = permute_183 = None
        view_318: "f32[8, 50, 1536]" = torch.ops.aten.reshape.default(addmm_61, [8, 50, 1536]);  addmm_61 = None
        view_319: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.reshape.default(view_318, [8, 50, 3, 8, 64]);  view_318 = None
        permute_184: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.permute.default(view_319, [2, 0, 3, 1, 4]);  view_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:119 in forward, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
        unbind_15 = torch.ops.aten.unbind.int(permute_184);  permute_184 = None
        getitem_170: "f32[8, 8, 50, 64]" = unbind_15[0]
        getitem_171: "f32[8, 8, 50, 64]" = unbind_15[1]
        getitem_172: "f32[8, 8, 50, 64]" = unbind_15[2];  unbind_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:75 in forward, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_213: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(getitem_172, 2, 1, 9223372036854775807)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:77 in forward, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        permute_186: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(slice_213, [0, 1, 3, 2]);  slice_213 = None
        view_326: "f32[8, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_186, [8, 512, 7, 7]);  permute_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:78 in forward, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(view_326, [128, 192, 192], 1);  view_326 = None
        getitem_173: "f32[8, 128, 7, 7]" = split_with_sizes_15[0]
        getitem_174: "f32[8, 192, 7, 7]" = split_with_sizes_15[1]
        getitem_175: "f32[8, 192, 7, 7]" = split_with_sizes_15[2];  split_with_sizes_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_70: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_170, [8, 8, 50, 64])
        clone_123: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_323: "f32[64, 50, 64]" = torch.ops.aten.reshape.default(clone_123, [64, 50, 64]);  clone_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:122 in forward, code: k_softmax = k.softmax(dim=2)
        clone_121: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(getitem_171, memory_format = torch.contiguous_format);  getitem_171 = None
        amax_15: "f32[8, 8, 1, 64]" = torch.ops.aten.amax.default(clone_121, [2], True)
        sub_55: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(clone_121, amax_15);  clone_121 = amax_15 = None
        exp_15: "f32[8, 8, 50, 64]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_16: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_15, [2], True)
        div_15: "f32[8, 8, 50, 64]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:123 in forward, code: factor_att = k_softmax.transpose(-1, -2) @ v
        permute_185: "f32[8, 8, 64, 50]" = torch.ops.aten.permute.default(div_15, [0, 1, 3, 2]);  div_15 = None
        expand_68: "f32[8, 8, 64, 50]" = torch.ops.aten.expand.default(permute_185, [8, 8, 64, 50]);  permute_185 = None
        view_320: "f32[64, 64, 50]" = torch.ops.aten.reshape.default(expand_68, [64, 64, 50]);  expand_68 = None
        expand_69: "f32[8, 8, 50, 64]" = torch.ops.aten.expand.default(getitem_172, [8, 8, 50, 64]);  getitem_172 = None
        clone_122: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_321: "f32[64, 50, 64]" = torch.ops.aten.reshape.default(clone_122, [64, 50, 64]);  clone_122 = None
        bmm_30: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = view_321 = None
        view_322: "f32[8, 8, 64, 64]" = torch.ops.aten.reshape.default(bmm_30, [8, 8, 64, 64]);  bmm_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:124 in forward, code: factor_att = q @ factor_att
        expand_71: "f32[8, 8, 64, 64]" = torch.ops.aten.expand.default(view_322, [8, 8, 64, 64]);  view_322 = None
        view_324: "f32[64, 64, 64]" = torch.ops.aten.reshape.default(expand_71, [64, 64, 64]);  expand_71 = None
        bmm_31: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_323, view_324);  view_323 = view_324 = None
        view_325: "f32[8, 8, 50, 64]" = torch.ops.aten.reshape.default(bmm_31, [8, 8, 50, 64]);  bmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        mul_156: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(view_325, 0.125);  view_325 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:74 in forward, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        slice_209: "f32[8, 8, 49, 64]" = torch.ops.aten.slice.Tensor(getitem_170, 2, 1, 9223372036854775807);  getitem_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:81 in forward, code: conv_v_img_list.append(conv(v_img_list[i]))
        convolution_69: "f32[8, 128, 7, 7]" = torch.ops.aten.convolution.default(getitem_173, arg123_1, arg124_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 128);  getitem_173 = arg123_1 = arg124_1 = None
        convolution_70: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_174, arg125_1, arg126_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 192);  getitem_174 = arg125_1 = arg126_1 = None
        convolution_71: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(getitem_175, arg127_1, arg128_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 192);  getitem_175 = arg127_1 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:82 in forward, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
        cat_39: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([convolution_69, convolution_70, convolution_71], 1);  convolution_69 = convolution_70 = convolution_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:83 in forward, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
        view_327: "f32[8, 8, 64, 49]" = torch.ops.aten.reshape.default(cat_39, [8, 8, 64, 49]);  cat_39 = None
        permute_187: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_327, [0, 1, 3, 2]);  view_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:85 in forward, code: EV_hat = q_img * conv_v_img
        mul_155: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(slice_209, permute_187);  slice_209 = permute_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_15: "f32[8, 8, 50, 64]" = torch.ops.aten.constant_pad_nd.default(mul_155, [0, 0, 1, 0, 0, 0], 0.0);  mul_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:130 in forward, code: x = self.scale * factor_att + crpe
        add_156: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(mul_156, constant_pad_nd_15);  mul_156 = constant_pad_nd_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:131 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
        permute_188: "f32[8, 50, 8, 64]" = torch.ops.aten.permute.default(add_156, [0, 2, 1, 3]);  add_156 = None
        clone_124: "f32[8, 50, 8, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_328: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(clone_124, [8, 50, 512]);  clone_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_329: "f32[400, 512]" = torch.ops.aten.reshape.default(view_328, [400, 512]);  view_328 = None
        permute_189: "f32[512, 512]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[400, 512]" = torch.ops.aten.mm.default(view_329, permute_189);  view_329 = permute_189 = None
        add_tensor_2: "f32[400, 512]" = torch.ops.aten.add.Tensor(mm_default_2, arg142_1);  mm_default_2 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:134 in forward, code: x = self.proj(x)
        view_330: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 50, 512]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:215 in forward, code: x = x + self.drop_path(cur)
        add_157: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(cat_38, view_330);  cat_38 = view_330 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_40 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_176: "f32[8, 50, 1]" = var_mean_40[0]
        getitem_177: "f32[8, 50, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_56: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_157, getitem_177);  getitem_177 = None
        add_158: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_40: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        mul_157: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_40);  sub_56 = rsqrt_40 = None
        mul_158: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_157, arg143_1);  mul_157 = arg143_1 = None
        add_159: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_158, arg144_1);  mul_158 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_331: "f32[400, 512]" = torch.ops.aten.reshape.default(add_159, [400, 512]);  add_159 = None
        permute_190: "f32[512, 2048]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[400, 2048]" = torch.ops.aten.mm.default(view_331, permute_190);  view_331 = permute_190 = None
        add_tensor_1: "f32[400, 2048]" = torch.ops.aten.add.Tensor(mm_default_1, arg146_1);  mm_default_1 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_332: "f32[8, 50, 2048]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 50, 2048]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_159: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_332, 0.5)
        mul_160: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476);  view_332 = None
        erf_15: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
        add_160: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_161: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_159, add_160);  mul_159 = add_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_333: "f32[400, 2048]" = torch.ops.aten.reshape.default(mul_161, [400, 2048]);  mul_161 = None
        permute_191: "f32[2048, 512]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[400, 512]" = torch.ops.aten.mm.default(view_333, permute_191);  view_333 = permute_191 = None
        add_tensor: "f32[400, 512]" = torch.ops.aten.add.Tensor(mm_default, arg148_1);  mm_default = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_334: "f32[8, 50, 512]" = torch.ops.aten.reshape.default(add_tensor, [8, 50, 512]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:220 in forward, code: x = x + self.drop_path(cur)
        add_161: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_157, view_334);  add_157 = view_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_41 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_178: "f32[8, 50, 1]" = var_mean_41[0]
        getitem_179: "f32[8, 50, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_57: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(add_161, getitem_179);  add_161 = getitem_179 = None
        add_162: "f32[8, 50, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_41: "f32[8, 50, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_162: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_41);  sub_57 = rsqrt_41 = None
        mul_163: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_162, arg149_1);  mul_162 = arg149_1 = None
        add_163: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_163, arg150_1);  mul_163 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:659 in forward_head, code: x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
        select_1: "f32[8, 512]" = torch.ops.aten.select.int(add_163, 1, 0);  add_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:660 in forward_head, code: x = self.head_drop(x)
        clone_129: "f32[8, 512]" = torch.ops.aten.clone.default(select_1);  select_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/coat.py:661 in forward_head, code: return x if pre_logits else self.head(x)
        permute_193: "f32[512, 1000]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_65: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg152_1, clone_129, permute_193);  arg152_1 = clone_129 = permute_193 = None
        return (addmm_65,)
        