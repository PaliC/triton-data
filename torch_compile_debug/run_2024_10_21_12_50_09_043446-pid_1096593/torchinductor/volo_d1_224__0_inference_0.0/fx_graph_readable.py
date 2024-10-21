class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64, 64, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64, 64, 3, 3]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[192, 64, 4, 4]", arg17_1: "f32[192]", arg18_1: "f32[192]", arg19_1: "f32[192]", arg20_1: "f32[192, 192]", arg21_1: "f32[486, 192]", arg22_1: "f32[486]", arg23_1: "f32[192, 192]", arg24_1: "f32[192]", arg25_1: "f32[192]", arg26_1: "f32[192]", arg27_1: "f32[576, 192]", arg28_1: "f32[576]", arg29_1: "f32[192, 576]", arg30_1: "f32[192]", arg31_1: "f32[192]", arg32_1: "f32[192]", arg33_1: "f32[192, 192]", arg34_1: "f32[486, 192]", arg35_1: "f32[486]", arg36_1: "f32[192, 192]", arg37_1: "f32[192]", arg38_1: "f32[192]", arg39_1: "f32[192]", arg40_1: "f32[576, 192]", arg41_1: "f32[576]", arg42_1: "f32[192, 576]", arg43_1: "f32[192]", arg44_1: "f32[192]", arg45_1: "f32[192]", arg46_1: "f32[192, 192]", arg47_1: "f32[486, 192]", arg48_1: "f32[486]", arg49_1: "f32[192, 192]", arg50_1: "f32[192]", arg51_1: "f32[192]", arg52_1: "f32[192]", arg53_1: "f32[576, 192]", arg54_1: "f32[576]", arg55_1: "f32[192, 576]", arg56_1: "f32[192]", arg57_1: "f32[192]", arg58_1: "f32[192]", arg59_1: "f32[192, 192]", arg60_1: "f32[486, 192]", arg61_1: "f32[486]", arg62_1: "f32[192, 192]", arg63_1: "f32[192]", arg64_1: "f32[192]", arg65_1: "f32[192]", arg66_1: "f32[576, 192]", arg67_1: "f32[576]", arg68_1: "f32[192, 576]", arg69_1: "f32[192]", arg70_1: "f32[384, 192, 2, 2]", arg71_1: "f32[384]", arg72_1: "f32[1, 14, 14, 384]", arg73_1: "f32[384]", arg74_1: "f32[384]", arg75_1: "f32[1152, 384]", arg76_1: "f32[384, 384]", arg77_1: "f32[384]", arg78_1: "f32[384]", arg79_1: "f32[384]", arg80_1: "f32[1152, 384]", arg81_1: "f32[1152]", arg82_1: "f32[384, 1152]", arg83_1: "f32[384]", arg84_1: "f32[384]", arg85_1: "f32[384]", arg86_1: "f32[1152, 384]", arg87_1: "f32[384, 384]", arg88_1: "f32[384]", arg89_1: "f32[384]", arg90_1: "f32[384]", arg91_1: "f32[1152, 384]", arg92_1: "f32[1152]", arg93_1: "f32[384, 1152]", arg94_1: "f32[384]", arg95_1: "f32[384]", arg96_1: "f32[384]", arg97_1: "f32[1152, 384]", arg98_1: "f32[384, 384]", arg99_1: "f32[384]", arg100_1: "f32[384]", arg101_1: "f32[384]", arg102_1: "f32[1152, 384]", arg103_1: "f32[1152]", arg104_1: "f32[384, 1152]", arg105_1: "f32[384]", arg106_1: "f32[384]", arg107_1: "f32[384]", arg108_1: "f32[1152, 384]", arg109_1: "f32[384, 384]", arg110_1: "f32[384]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[1152, 384]", arg114_1: "f32[1152]", arg115_1: "f32[384, 1152]", arg116_1: "f32[384]", arg117_1: "f32[384]", arg118_1: "f32[384]", arg119_1: "f32[1152, 384]", arg120_1: "f32[384, 384]", arg121_1: "f32[384]", arg122_1: "f32[384]", arg123_1: "f32[384]", arg124_1: "f32[1152, 384]", arg125_1: "f32[1152]", arg126_1: "f32[384, 1152]", arg127_1: "f32[384]", arg128_1: "f32[384]", arg129_1: "f32[384]", arg130_1: "f32[1152, 384]", arg131_1: "f32[384, 384]", arg132_1: "f32[384]", arg133_1: "f32[384]", arg134_1: "f32[384]", arg135_1: "f32[1152, 384]", arg136_1: "f32[1152]", arg137_1: "f32[384, 1152]", arg138_1: "f32[384]", arg139_1: "f32[384]", arg140_1: "f32[384]", arg141_1: "f32[1152, 384]", arg142_1: "f32[384, 384]", arg143_1: "f32[384]", arg144_1: "f32[384]", arg145_1: "f32[384]", arg146_1: "f32[1152, 384]", arg147_1: "f32[1152]", arg148_1: "f32[384, 1152]", arg149_1: "f32[384]", arg150_1: "f32[384]", arg151_1: "f32[384]", arg152_1: "f32[1152, 384]", arg153_1: "f32[384, 384]", arg154_1: "f32[384]", arg155_1: "f32[384]", arg156_1: "f32[384]", arg157_1: "f32[1152, 384]", arg158_1: "f32[1152]", arg159_1: "f32[384, 1152]", arg160_1: "f32[384]", arg161_1: "f32[384]", arg162_1: "f32[384]", arg163_1: "f32[1152, 384]", arg164_1: "f32[384, 384]", arg165_1: "f32[384]", arg166_1: "f32[384]", arg167_1: "f32[384]", arg168_1: "f32[1152, 384]", arg169_1: "f32[1152]", arg170_1: "f32[384, 1152]", arg171_1: "f32[384]", arg172_1: "f32[384]", arg173_1: "f32[384]", arg174_1: "f32[1152, 384]", arg175_1: "f32[384, 384]", arg176_1: "f32[384]", arg177_1: "f32[384]", arg178_1: "f32[384]", arg179_1: "f32[1152, 384]", arg180_1: "f32[1152]", arg181_1: "f32[384, 1152]", arg182_1: "f32[384]", arg183_1: "f32[384]", arg184_1: "f32[384]", arg185_1: "f32[1152, 384]", arg186_1: "f32[384, 384]", arg187_1: "f32[384]", arg188_1: "f32[384]", arg189_1: "f32[384]", arg190_1: "f32[1152, 384]", arg191_1: "f32[1152]", arg192_1: "f32[384, 1152]", arg193_1: "f32[384]", arg194_1: "f32[384]", arg195_1: "f32[384]", arg196_1: "f32[1152, 384]", arg197_1: "f32[384, 384]", arg198_1: "f32[384]", arg199_1: "f32[384]", arg200_1: "f32[384]", arg201_1: "f32[1152, 384]", arg202_1: "f32[1152]", arg203_1: "f32[384, 1152]", arg204_1: "f32[384]", arg205_1: "f32[384]", arg206_1: "f32[384]", arg207_1: "f32[1152, 384]", arg208_1: "f32[384, 384]", arg209_1: "f32[384]", arg210_1: "f32[384]", arg211_1: "f32[384]", arg212_1: "f32[1152, 384]", arg213_1: "f32[1152]", arg214_1: "f32[384, 1152]", arg215_1: "f32[384]", arg216_1: "f32[384]", arg217_1: "f32[384]", arg218_1: "f32[1152, 384]", arg219_1: "f32[384, 384]", arg220_1: "f32[384]", arg221_1: "f32[384]", arg222_1: "f32[384]", arg223_1: "f32[1152, 384]", arg224_1: "f32[1152]", arg225_1: "f32[384, 1152]", arg226_1: "f32[384]", arg227_1: "f32[1, 1, 384]", arg228_1: "f32[384]", arg229_1: "f32[384]", arg230_1: "f32[768, 384]", arg231_1: "f32[384, 384]", arg232_1: "f32[384, 384]", arg233_1: "f32[384]", arg234_1: "f32[384]", arg235_1: "f32[384]", arg236_1: "f32[1152, 384]", arg237_1: "f32[1152]", arg238_1: "f32[384, 1152]", arg239_1: "f32[384]", arg240_1: "f32[384]", arg241_1: "f32[384]", arg242_1: "f32[768, 384]", arg243_1: "f32[384, 384]", arg244_1: "f32[384, 384]", arg245_1: "f32[384]", arg246_1: "f32[384]", arg247_1: "f32[384]", arg248_1: "f32[1152, 384]", arg249_1: "f32[1152]", arg250_1: "f32[384, 1152]", arg251_1: "f32[384]", arg252_1: "f32[384]", arg253_1: "f32[384]", arg254_1: "f32[1000, 384]", arg255_1: "f32[1000]", arg256_1: "f32[1000, 384]", arg257_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:367 in forward, code: x = self.conv(x)
        convolution_5: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_171: "f32[64]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_3: "f32[64]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_158: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        sub_50: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_73);  convolution_5 = unsqueeze_73 = None
        mul_159: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_75);  sub_50 = unsqueeze_75 = None
        unsqueeze_76: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_77: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_160: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_77);  mul_159 = unsqueeze_77 = None
        unsqueeze_78: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_79: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_172: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_79);  mul_160 = unsqueeze_79 = None
        relu_3: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_172);  add_172 = None
        convolution_6: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu_3, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg6_1 = None
        add_173: "f32[64]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_4: "f32[64]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_4: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_161: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_80: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_81: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_161, -1);  mul_161 = None
        unsqueeze_83: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        sub_51: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_81);  convolution_6 = unsqueeze_81 = None
        mul_162: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_83);  sub_51 = unsqueeze_83 = None
        unsqueeze_84: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_85: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_163: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_85);  mul_162 = unsqueeze_85 = None
        unsqueeze_86: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_87: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_174: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_87);  mul_163 = unsqueeze_87 = None
        relu_4: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_174);  add_174 = None
        convolution_7: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu_4, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg11_1 = None
        add_175: "f32[64]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_5: "f32[64]" = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_5: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_164: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_88: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_89: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_164, -1);  mul_164 = None
        unsqueeze_91: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        sub_52: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_89);  convolution_7 = unsqueeze_89 = None
        mul_165: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_91);  sub_52 = unsqueeze_91 = None
        unsqueeze_92: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_93: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_166: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_93);  mul_165 = unsqueeze_93 = None
        unsqueeze_94: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_95: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_176: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_95);  mul_166 = unsqueeze_95 = None
        relu_5: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_176);  add_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:368 in forward, code: x = self.proj(x)  # B, C, H, W
        convolution_8: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_5, arg16_1, arg17_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg16_1 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:792 in forward_features, code: x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C
        permute_161: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(convolution_8, [0, 2, 3, 1]);  convolution_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:135 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_129: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format)
        var_mean_41 = torch.ops.aten.var_mean.correction(clone_129, [3], correction = 0, keepdim = True)
        getitem_186: "f32[8, 28, 28, 1]" = var_mean_41[0]
        getitem_187: "f32[8, 28, 28, 1]" = var_mean_41[1];  var_mean_41 = None
        add_177: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_41: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        sub_53: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_129, getitem_187);  clone_129 = getitem_187 = None
        mul_167: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_41);  sub_53 = rsqrt_41 = None
        mul_168: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_167, arg18_1);  mul_167 = arg18_1 = None
        add_178: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_168, arg19_1);  mul_168 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:74 in forward, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
        permute_162: "f32[192, 192]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        view_253: "f32[6272, 192]" = torch.ops.aten.view.default(add_178, [6272, 192])
        mm_27: "f32[6272, 192]" = torch.ops.aten.mm.default(view_253, permute_162);  view_253 = permute_162 = None
        view_254: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_27, [8, 28, 28, 192]);  mm_27 = None
        permute_163: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_254, [0, 3, 1, 2]);  view_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:77 in forward, code: v = self.unfold(v).reshape(
        iota_32: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_96: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_32, 0);  iota_32 = None
        iota_33: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_97: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        add_179: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_96, unsqueeze_97);  unsqueeze_96 = unsqueeze_97 = None
        iota_34: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_98: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_34, 0);  iota_34 = None
        iota_35: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_99: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
        add_180: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_98, unsqueeze_99);  unsqueeze_98 = unsqueeze_99 = None
        constant_pad_nd_8: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_163, [1, 1, 1, 1], 0.0);  permute_163 = None
        unsqueeze_100: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_179, -1);  add_179 = None
        unsqueeze_101: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        index_4: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(constant_pad_nd_8, [None, None, unsqueeze_101, add_180]);  constant_pad_nd_8 = unsqueeze_101 = add_180 = None
        permute_164: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
        clone_130: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_255: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_130, [8, 1728, 196]);  clone_130 = None
        view_256: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_255, [8, 6, 32, 9, 196]);  view_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:79 in forward, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
        permute_165: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_256, [0, 1, 4, 3, 2]);  view_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:81 in forward, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        permute_166: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_178, [0, 3, 1, 2]);  add_178 = None
        avg_pool2d_4: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_166, [2, 2], [2, 2], [0, 0], True);  permute_166 = None
        permute_167: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_4, [0, 2, 3, 1]);  avg_pool2d_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:82 in forward, code: attn = self.attn(attn).reshape(
        view_257: "f32[1568, 192]" = torch.ops.aten.view.default(permute_167, [1568, 192]);  permute_167 = None
        permute_168: "f32[192, 486]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_61: "f32[1568, 486]" = torch.ops.aten.addmm.default(arg22_1, view_257, permute_168);  arg22_1 = view_257 = permute_168 = None
        view_258: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_61, [8, 14, 14, 486]);  addmm_61 = None
        view_259: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_258, [8, 196, 6, 9, 9]);  view_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:84 in forward, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        permute_169: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3, 4]);  view_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:85 in forward, code: attn = attn * self.scale
        mul_169: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_169, 0.1767766952966369);  permute_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:86 in forward, code: attn = attn.softmax(dim=-1)
        clone_131: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_169, memory_format = torch.contiguous_format);  mul_169 = None
        amax_6: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_131, [-1], True)
        sub_54: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_131, amax_6);  clone_131 = amax_6 = None
        exp_6: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
        sum_7: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:89 in forward, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        expand_17: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(div_6, [8, 6, 196, 9, 9]);  div_6 = None
        view_260: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_17, [9408, 9, 9]);  expand_17 = None
        expand_18: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_165, [8, 6, 196, 9, 32]);  permute_165 = None
        clone_133: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_261: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_133, [9408, 9, 32]);  clone_133 = None
        bmm_8: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_260, view_261);  view_260 = view_261 = None
        view_262: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_8, [8, 6, 196, 9, 32]);  bmm_8 = None
        permute_170: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_262, [0, 1, 4, 3, 2]);  view_262 = None
        clone_134: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_263: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_134, [8, 1728, 196]);  clone_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:90 in forward, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        view_264: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_263, [8, 192, 3, 3, 14, 14]);  view_263 = None
        permute_171: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_264, [0, 1, 2, 4, 3, 5]);  view_264 = None
        iota_36: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_102: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_36, 0);  iota_36 = None
        iota_37: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_103: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        add_181: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_102, unsqueeze_103);  unsqueeze_102 = unsqueeze_103 = None
        unsqueeze_104: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_181, -1);  add_181 = None
        unsqueeze_105: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        iota_38: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_106: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_38, 0);  iota_38 = None
        iota_39: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_107: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
        add_182: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_106, unsqueeze_107);  unsqueeze_106 = unsqueeze_107 = None
        full_default: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_4: "f32[8, 192, 30, 30]" = torch.ops.aten.index_put.default(full_default, [None, None, unsqueeze_105, add_182], permute_171, True);  full_default = unsqueeze_105 = add_182 = permute_171 = None
        constant_pad_nd_9: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(index_put_4, [-1, -1, -1, -1], 0.0);  index_put_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:92 in forward, code: x = self.proj(x.permute(0, 2, 3, 1))
        permute_172: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_9, [0, 2, 3, 1]);  constant_pad_nd_9 = None
        permute_173: "f32[192, 192]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        clone_135: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_265: "f32[6272, 192]" = torch.ops.aten.view.default(clone_135, [6272, 192]);  clone_135 = None
        mm_28: "f32[6272, 192]" = torch.ops.aten.mm.default(view_265, permute_173);  view_265 = permute_173 = None
        view_266: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_28, [8, 28, 28, 192]);  mm_28 = None
        add_183: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_266, arg24_1);  view_266 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:135 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_184: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_161, add_183);  permute_161 = add_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:136 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_137: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_184, memory_format = torch.contiguous_format)
        var_mean_42 = torch.ops.aten.var_mean.correction(clone_137, [3], correction = 0, keepdim = True)
        getitem_188: "f32[8, 28, 28, 1]" = var_mean_42[0]
        getitem_189: "f32[8, 28, 28, 1]" = var_mean_42[1];  var_mean_42 = None
        add_185: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_42: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        sub_55: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_137, getitem_189);  clone_137 = getitem_189 = None
        mul_170: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_42);  sub_55 = rsqrt_42 = None
        mul_171: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_170, arg25_1);  mul_170 = arg25_1 = None
        add_186: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_171, arg26_1);  mul_171 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_267: "f32[6272, 192]" = torch.ops.aten.view.default(add_186, [6272, 192]);  add_186 = None
        permute_174: "f32[192, 576]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_62: "f32[6272, 576]" = torch.ops.aten.addmm.default(arg28_1, view_267, permute_174);  arg28_1 = view_267 = permute_174 = None
        view_268: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_62, [8, 28, 28, 576]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_172: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_268, 0.5)
        mul_173: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_268, 0.7071067811865476);  view_268 = None
        erf_20: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_173);  mul_173 = None
        add_187: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_174: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_172, add_187);  mul_172 = add_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_269: "f32[6272, 576]" = torch.ops.aten.view.default(mul_174, [6272, 576]);  mul_174 = None
        permute_175: "f32[576, 192]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_63: "f32[6272, 192]" = torch.ops.aten.addmm.default(arg30_1, view_269, permute_175);  arg30_1 = view_269 = permute_175 = None
        view_270: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_63, [8, 28, 28, 192]);  addmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:136 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_188: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_184, view_270);  add_184 = view_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:135 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_140: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format)
        var_mean_43 = torch.ops.aten.var_mean.correction(clone_140, [3], correction = 0, keepdim = True)
        getitem_190: "f32[8, 28, 28, 1]" = var_mean_43[0]
        getitem_191: "f32[8, 28, 28, 1]" = var_mean_43[1];  var_mean_43 = None
        add_189: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_43: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
        sub_56: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_140, getitem_191);  clone_140 = getitem_191 = None
        mul_175: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_43);  sub_56 = rsqrt_43 = None
        mul_176: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_175, arg31_1);  mul_175 = arg31_1 = None
        add_190: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_176, arg32_1);  mul_176 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:74 in forward, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
        permute_176: "f32[192, 192]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        view_271: "f32[6272, 192]" = torch.ops.aten.view.default(add_190, [6272, 192])
        mm_29: "f32[6272, 192]" = torch.ops.aten.mm.default(view_271, permute_176);  view_271 = permute_176 = None
        view_272: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_29, [8, 28, 28, 192]);  mm_29 = None
        permute_177: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_272, [0, 3, 1, 2]);  view_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:77 in forward, code: v = self.unfold(v).reshape(
        iota_40: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_108: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_40, 0);  iota_40 = None
        iota_41: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_109: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        add_191: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_108, unsqueeze_109);  unsqueeze_108 = unsqueeze_109 = None
        iota_42: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_110: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_42, 0);  iota_42 = None
        iota_43: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_111: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
        add_192: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_110, unsqueeze_111);  unsqueeze_110 = unsqueeze_111 = None
        constant_pad_nd_10: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_177, [1, 1, 1, 1], 0.0);  permute_177 = None
        unsqueeze_112: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_191, -1);  add_191 = None
        unsqueeze_113: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        index_5: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(constant_pad_nd_10, [None, None, unsqueeze_113, add_192]);  constant_pad_nd_10 = unsqueeze_113 = add_192 = None
        permute_178: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
        clone_141: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
        view_273: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_141, [8, 1728, 196]);  clone_141 = None
        view_274: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_273, [8, 6, 32, 9, 196]);  view_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:79 in forward, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
        permute_179: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_274, [0, 1, 4, 3, 2]);  view_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:81 in forward, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        permute_180: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_190, [0, 3, 1, 2]);  add_190 = None
        avg_pool2d_5: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_180, [2, 2], [2, 2], [0, 0], True);  permute_180 = None
        permute_181: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_5, [0, 2, 3, 1]);  avg_pool2d_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:82 in forward, code: attn = self.attn(attn).reshape(
        view_275: "f32[1568, 192]" = torch.ops.aten.view.default(permute_181, [1568, 192]);  permute_181 = None
        permute_182: "f32[192, 486]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_64: "f32[1568, 486]" = torch.ops.aten.addmm.default(arg35_1, view_275, permute_182);  arg35_1 = view_275 = permute_182 = None
        view_276: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_64, [8, 14, 14, 486]);  addmm_64 = None
        view_277: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_276, [8, 196, 6, 9, 9]);  view_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:84 in forward, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        permute_183: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3, 4]);  view_277 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:85 in forward, code: attn = attn * self.scale
        mul_177: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_183, 0.1767766952966369);  permute_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:86 in forward, code: attn = attn.softmax(dim=-1)
        clone_142: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_177, memory_format = torch.contiguous_format);  mul_177 = None
        amax_7: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_142, [-1], True)
        sub_57: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_142, amax_7);  clone_142 = amax_7 = None
        exp_7: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_8: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:89 in forward, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        expand_19: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(div_7, [8, 6, 196, 9, 9]);  div_7 = None
        view_278: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_19, [9408, 9, 9]);  expand_19 = None
        expand_20: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_179, [8, 6, 196, 9, 32]);  permute_179 = None
        clone_144: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_279: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_144, [9408, 9, 32]);  clone_144 = None
        bmm_9: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_278, view_279);  view_278 = view_279 = None
        view_280: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_9, [8, 6, 196, 9, 32]);  bmm_9 = None
        permute_184: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_280, [0, 1, 4, 3, 2]);  view_280 = None
        clone_145: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_281: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_145, [8, 1728, 196]);  clone_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:90 in forward, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        view_282: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_281, [8, 192, 3, 3, 14, 14]);  view_281 = None
        permute_185: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_282, [0, 1, 2, 4, 3, 5]);  view_282 = None
        iota_44: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_114: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_44, 0);  iota_44 = None
        iota_45: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_115: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        add_193: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_114, unsqueeze_115);  unsqueeze_114 = unsqueeze_115 = None
        unsqueeze_116: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_193, -1);  add_193 = None
        unsqueeze_117: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        iota_46: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_118: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_46, 0);  iota_46 = None
        iota_47: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_119: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
        add_194: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_118, unsqueeze_119);  unsqueeze_118 = unsqueeze_119 = None
        full_default_1: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_5: "f32[8, 192, 30, 30]" = torch.ops.aten.index_put.default(full_default_1, [None, None, unsqueeze_117, add_194], permute_185, True);  full_default_1 = unsqueeze_117 = add_194 = permute_185 = None
        constant_pad_nd_11: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(index_put_5, [-1, -1, -1, -1], 0.0);  index_put_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:92 in forward, code: x = self.proj(x.permute(0, 2, 3, 1))
        permute_186: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_11, [0, 2, 3, 1]);  constant_pad_nd_11 = None
        permute_187: "f32[192, 192]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        clone_146: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
        view_283: "f32[6272, 192]" = torch.ops.aten.view.default(clone_146, [6272, 192]);  clone_146 = None
        mm_30: "f32[6272, 192]" = torch.ops.aten.mm.default(view_283, permute_187);  view_283 = permute_187 = None
        view_284: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_30, [8, 28, 28, 192]);  mm_30 = None
        add_195: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_284, arg37_1);  view_284 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:135 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_196: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_188, add_195);  add_188 = add_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:136 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_148: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_196, memory_format = torch.contiguous_format)
        var_mean_44 = torch.ops.aten.var_mean.correction(clone_148, [3], correction = 0, keepdim = True)
        getitem_192: "f32[8, 28, 28, 1]" = var_mean_44[0]
        getitem_193: "f32[8, 28, 28, 1]" = var_mean_44[1];  var_mean_44 = None
        add_197: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_44: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_58: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_148, getitem_193);  clone_148 = getitem_193 = None
        mul_178: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_44);  sub_58 = rsqrt_44 = None
        mul_179: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_178, arg38_1);  mul_178 = arg38_1 = None
        add_198: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_179, arg39_1);  mul_179 = arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_285: "f32[6272, 192]" = torch.ops.aten.view.default(add_198, [6272, 192]);  add_198 = None
        permute_188: "f32[192, 576]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_65: "f32[6272, 576]" = torch.ops.aten.addmm.default(arg41_1, view_285, permute_188);  arg41_1 = view_285 = permute_188 = None
        view_286: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_65, [8, 28, 28, 576]);  addmm_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_180: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_286, 0.5)
        mul_181: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_286, 0.7071067811865476);  view_286 = None
        erf_21: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_199: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_182: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_180, add_199);  mul_180 = add_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_287: "f32[6272, 576]" = torch.ops.aten.view.default(mul_182, [6272, 576]);  mul_182 = None
        permute_189: "f32[576, 192]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_66: "f32[6272, 192]" = torch.ops.aten.addmm.default(arg43_1, view_287, permute_189);  arg43_1 = view_287 = permute_189 = None
        view_288: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_66, [8, 28, 28, 192]);  addmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:136 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_200: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_196, view_288);  add_196 = view_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:135 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_151: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_200, memory_format = torch.contiguous_format)
        var_mean_45 = torch.ops.aten.var_mean.correction(clone_151, [3], correction = 0, keepdim = True)
        getitem_194: "f32[8, 28, 28, 1]" = var_mean_45[0]
        getitem_195: "f32[8, 28, 28, 1]" = var_mean_45[1];  var_mean_45 = None
        add_201: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_45: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_59: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_151, getitem_195);  clone_151 = getitem_195 = None
        mul_183: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_45);  sub_59 = rsqrt_45 = None
        mul_184: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_183, arg44_1);  mul_183 = arg44_1 = None
        add_202: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_184, arg45_1);  mul_184 = arg45_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:74 in forward, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
        permute_190: "f32[192, 192]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        view_289: "f32[6272, 192]" = torch.ops.aten.view.default(add_202, [6272, 192])
        mm_31: "f32[6272, 192]" = torch.ops.aten.mm.default(view_289, permute_190);  view_289 = permute_190 = None
        view_290: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_31, [8, 28, 28, 192]);  mm_31 = None
        permute_191: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_290, [0, 3, 1, 2]);  view_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:77 in forward, code: v = self.unfold(v).reshape(
        iota_48: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_120: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_48, 0);  iota_48 = None
        iota_49: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_121: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_49, -1);  iota_49 = None
        add_203: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_120, unsqueeze_121);  unsqueeze_120 = unsqueeze_121 = None
        iota_50: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_122: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_50, 0);  iota_50 = None
        iota_51: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_123: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_51, -1);  iota_51 = None
        add_204: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_122, unsqueeze_123);  unsqueeze_122 = unsqueeze_123 = None
        constant_pad_nd_12: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_191, [1, 1, 1, 1], 0.0);  permute_191 = None
        unsqueeze_124: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_203, -1);  add_203 = None
        unsqueeze_125: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        index_6: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(constant_pad_nd_12, [None, None, unsqueeze_125, add_204]);  constant_pad_nd_12 = unsqueeze_125 = add_204 = None
        permute_192: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
        clone_152: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_291: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_152, [8, 1728, 196]);  clone_152 = None
        view_292: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_291, [8, 6, 32, 9, 196]);  view_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:79 in forward, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
        permute_193: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_292, [0, 1, 4, 3, 2]);  view_292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:81 in forward, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        permute_194: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_202, [0, 3, 1, 2]);  add_202 = None
        avg_pool2d_6: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_194, [2, 2], [2, 2], [0, 0], True);  permute_194 = None
        permute_195: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_6, [0, 2, 3, 1]);  avg_pool2d_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:82 in forward, code: attn = self.attn(attn).reshape(
        view_293: "f32[1568, 192]" = torch.ops.aten.view.default(permute_195, [1568, 192]);  permute_195 = None
        permute_196: "f32[192, 486]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_67: "f32[1568, 486]" = torch.ops.aten.addmm.default(arg48_1, view_293, permute_196);  arg48_1 = view_293 = permute_196 = None
        view_294: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_67, [8, 14, 14, 486]);  addmm_67 = None
        view_295: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_294, [8, 196, 6, 9, 9]);  view_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:84 in forward, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        permute_197: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3, 4]);  view_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:85 in forward, code: attn = attn * self.scale
        mul_185: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_197, 0.1767766952966369);  permute_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:86 in forward, code: attn = attn.softmax(dim=-1)
        clone_153: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_185, memory_format = torch.contiguous_format);  mul_185 = None
        amax_8: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_153, [-1], True)
        sub_60: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_153, amax_8);  clone_153 = amax_8 = None
        exp_8: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_9: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:89 in forward, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        expand_21: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(div_8, [8, 6, 196, 9, 9]);  div_8 = None
        view_296: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_21, [9408, 9, 9]);  expand_21 = None
        expand_22: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_193, [8, 6, 196, 9, 32]);  permute_193 = None
        clone_155: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_297: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_155, [9408, 9, 32]);  clone_155 = None
        bmm_10: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_296, view_297);  view_296 = view_297 = None
        view_298: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_10, [8, 6, 196, 9, 32]);  bmm_10 = None
        permute_198: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_298, [0, 1, 4, 3, 2]);  view_298 = None
        clone_156: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_299: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_156, [8, 1728, 196]);  clone_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:90 in forward, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        view_300: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_299, [8, 192, 3, 3, 14, 14]);  view_299 = None
        permute_199: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_300, [0, 1, 2, 4, 3, 5]);  view_300 = None
        iota_52: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_126: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_52, 0);  iota_52 = None
        iota_53: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_127: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_53, -1);  iota_53 = None
        add_205: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_126, unsqueeze_127);  unsqueeze_126 = unsqueeze_127 = None
        unsqueeze_128: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_205, -1);  add_205 = None
        unsqueeze_129: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        iota_54: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_130: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_54, 0);  iota_54 = None
        iota_55: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_131: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_55, -1);  iota_55 = None
        add_206: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_130, unsqueeze_131);  unsqueeze_130 = unsqueeze_131 = None
        full_default_2: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_6: "f32[8, 192, 30, 30]" = torch.ops.aten.index_put.default(full_default_2, [None, None, unsqueeze_129, add_206], permute_199, True);  full_default_2 = unsqueeze_129 = add_206 = permute_199 = None
        constant_pad_nd_13: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(index_put_6, [-1, -1, -1, -1], 0.0);  index_put_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:92 in forward, code: x = self.proj(x.permute(0, 2, 3, 1))
        permute_200: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_13, [0, 2, 3, 1]);  constant_pad_nd_13 = None
        permute_201: "f32[192, 192]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        clone_157: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_301: "f32[6272, 192]" = torch.ops.aten.view.default(clone_157, [6272, 192]);  clone_157 = None
        mm_32: "f32[6272, 192]" = torch.ops.aten.mm.default(view_301, permute_201);  view_301 = permute_201 = None
        view_302: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_32, [8, 28, 28, 192]);  mm_32 = None
        add_207: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_302, arg50_1);  view_302 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:135 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_208: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_200, add_207);  add_200 = add_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:136 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_159: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_208, memory_format = torch.contiguous_format)
        var_mean_46 = torch.ops.aten.var_mean.correction(clone_159, [3], correction = 0, keepdim = True)
        getitem_196: "f32[8, 28, 28, 1]" = var_mean_46[0]
        getitem_197: "f32[8, 28, 28, 1]" = var_mean_46[1];  var_mean_46 = None
        add_209: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_46: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_61: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_159, getitem_197);  clone_159 = getitem_197 = None
        mul_186: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_46);  sub_61 = rsqrt_46 = None
        mul_187: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_186, arg51_1);  mul_186 = arg51_1 = None
        add_210: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_187, arg52_1);  mul_187 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_303: "f32[6272, 192]" = torch.ops.aten.view.default(add_210, [6272, 192]);  add_210 = None
        permute_202: "f32[192, 576]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_68: "f32[6272, 576]" = torch.ops.aten.addmm.default(arg54_1, view_303, permute_202);  arg54_1 = view_303 = permute_202 = None
        view_304: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_68, [8, 28, 28, 576]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_188: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_304, 0.5)
        mul_189: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_304, 0.7071067811865476);  view_304 = None
        erf_22: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_189);  mul_189 = None
        add_211: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_190: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_188, add_211);  mul_188 = add_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_305: "f32[6272, 576]" = torch.ops.aten.view.default(mul_190, [6272, 576]);  mul_190 = None
        permute_203: "f32[576, 192]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_69: "f32[6272, 192]" = torch.ops.aten.addmm.default(arg56_1, view_305, permute_203);  arg56_1 = view_305 = permute_203 = None
        view_306: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_69, [8, 28, 28, 192]);  addmm_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:136 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_212: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_208, view_306);  add_208 = view_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:135 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_162: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_212, memory_format = torch.contiguous_format)
        var_mean_47 = torch.ops.aten.var_mean.correction(clone_162, [3], correction = 0, keepdim = True)
        getitem_198: "f32[8, 28, 28, 1]" = var_mean_47[0]
        getitem_199: "f32[8, 28, 28, 1]" = var_mean_47[1];  var_mean_47 = None
        add_213: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_47: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_62: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_162, getitem_199);  clone_162 = getitem_199 = None
        mul_191: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_47);  sub_62 = rsqrt_47 = None
        mul_192: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_191, arg57_1);  mul_191 = arg57_1 = None
        add_214: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_192, arg58_1);  mul_192 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:74 in forward, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
        permute_204: "f32[192, 192]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        view_307: "f32[6272, 192]" = torch.ops.aten.view.default(add_214, [6272, 192])
        mm_33: "f32[6272, 192]" = torch.ops.aten.mm.default(view_307, permute_204);  view_307 = permute_204 = None
        view_308: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_33, [8, 28, 28, 192]);  mm_33 = None
        permute_205: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_308, [0, 3, 1, 2]);  view_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:77 in forward, code: v = self.unfold(v).reshape(
        iota_56: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_132: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_56, 0);  iota_56 = None
        iota_57: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_133: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_57, -1);  iota_57 = None
        add_215: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_132, unsqueeze_133);  unsqueeze_132 = unsqueeze_133 = None
        iota_58: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_134: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_58, 0);  iota_58 = None
        iota_59: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_135: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_59, -1);  iota_59 = None
        add_216: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_134, unsqueeze_135);  unsqueeze_134 = unsqueeze_135 = None
        constant_pad_nd_14: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_205, [1, 1, 1, 1], 0.0);  permute_205 = None
        unsqueeze_136: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_215, -1);  add_215 = None
        unsqueeze_137: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        index_7: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(constant_pad_nd_14, [None, None, unsqueeze_137, add_216]);  constant_pad_nd_14 = unsqueeze_137 = add_216 = None
        permute_206: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
        clone_163: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        view_309: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_163, [8, 1728, 196]);  clone_163 = None
        view_310: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_309, [8, 6, 32, 9, 196]);  view_309 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:79 in forward, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
        permute_207: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_310, [0, 1, 4, 3, 2]);  view_310 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:81 in forward, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        permute_208: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_214, [0, 3, 1, 2]);  add_214 = None
        avg_pool2d_7: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_208, [2, 2], [2, 2], [0, 0], True);  permute_208 = None
        permute_209: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_7, [0, 2, 3, 1]);  avg_pool2d_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:82 in forward, code: attn = self.attn(attn).reshape(
        view_311: "f32[1568, 192]" = torch.ops.aten.view.default(permute_209, [1568, 192]);  permute_209 = None
        permute_210: "f32[192, 486]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_70: "f32[1568, 486]" = torch.ops.aten.addmm.default(arg61_1, view_311, permute_210);  arg61_1 = view_311 = permute_210 = None
        view_312: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_70, [8, 14, 14, 486]);  addmm_70 = None
        view_313: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_312, [8, 196, 6, 9, 9]);  view_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:84 in forward, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        permute_211: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3, 4]);  view_313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:85 in forward, code: attn = attn * self.scale
        mul_193: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_211, 0.1767766952966369);  permute_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:86 in forward, code: attn = attn.softmax(dim=-1)
        clone_164: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_193, memory_format = torch.contiguous_format);  mul_193 = None
        amax_9: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_164, [-1], True)
        sub_63: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_164, amax_9);  clone_164 = amax_9 = None
        exp_9: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        sum_10: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:89 in forward, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        expand_23: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(div_9, [8, 6, 196, 9, 9]);  div_9 = None
        view_314: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_23, [9408, 9, 9]);  expand_23 = None
        expand_24: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_207, [8, 6, 196, 9, 32]);  permute_207 = None
        clone_166: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_315: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_166, [9408, 9, 32]);  clone_166 = None
        bmm_11: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_314, view_315);  view_314 = view_315 = None
        view_316: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_11, [8, 6, 196, 9, 32]);  bmm_11 = None
        permute_212: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_316, [0, 1, 4, 3, 2]);  view_316 = None
        clone_167: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_317: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_167, [8, 1728, 196]);  clone_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:90 in forward, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        view_318: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_317, [8, 192, 3, 3, 14, 14]);  view_317 = None
        permute_213: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_318, [0, 1, 2, 4, 3, 5]);  view_318 = None
        iota_60: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_138: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_60, 0);  iota_60 = None
        iota_61: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_139: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_61, -1);  iota_61 = None
        add_217: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_138, unsqueeze_139);  unsqueeze_138 = unsqueeze_139 = None
        unsqueeze_140: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_217, -1);  add_217 = None
        unsqueeze_141: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        iota_62: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_142: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_62, 0);  iota_62 = None
        iota_63: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_143: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_63, -1);  iota_63 = None
        add_218: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_142, unsqueeze_143);  unsqueeze_142 = unsqueeze_143 = None
        full_default_3: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_7: "f32[8, 192, 30, 30]" = torch.ops.aten.index_put.default(full_default_3, [None, None, unsqueeze_141, add_218], permute_213, True);  full_default_3 = unsqueeze_141 = add_218 = permute_213 = None
        constant_pad_nd_15: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(index_put_7, [-1, -1, -1, -1], 0.0);  index_put_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:92 in forward, code: x = self.proj(x.permute(0, 2, 3, 1))
        permute_214: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_15, [0, 2, 3, 1]);  constant_pad_nd_15 = None
        permute_215: "f32[192, 192]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        clone_168: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_319: "f32[6272, 192]" = torch.ops.aten.view.default(clone_168, [6272, 192]);  clone_168 = None
        mm_34: "f32[6272, 192]" = torch.ops.aten.mm.default(view_319, permute_215);  view_319 = permute_215 = None
        view_320: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_34, [8, 28, 28, 192]);  mm_34 = None
        add_219: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_320, arg63_1);  view_320 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:135 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_220: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_212, add_219);  add_212 = add_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:136 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_170: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_220, memory_format = torch.contiguous_format)
        var_mean_48 = torch.ops.aten.var_mean.correction(clone_170, [3], correction = 0, keepdim = True)
        getitem_200: "f32[8, 28, 28, 1]" = var_mean_48[0]
        getitem_201: "f32[8, 28, 28, 1]" = var_mean_48[1];  var_mean_48 = None
        add_221: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_48: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        sub_64: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_170, getitem_201);  clone_170 = getitem_201 = None
        mul_194: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_48);  sub_64 = rsqrt_48 = None
        mul_195: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_194, arg64_1);  mul_194 = arg64_1 = None
        add_222: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_195, arg65_1);  mul_195 = arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_321: "f32[6272, 192]" = torch.ops.aten.view.default(add_222, [6272, 192]);  add_222 = None
        permute_216: "f32[192, 576]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_71: "f32[6272, 576]" = torch.ops.aten.addmm.default(arg67_1, view_321, permute_216);  arg67_1 = view_321 = permute_216 = None
        view_322: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_71, [8, 28, 28, 576]);  addmm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_196: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_322, 0.5)
        mul_197: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_322, 0.7071067811865476);  view_322 = None
        erf_23: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
        add_223: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_198: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_196, add_223);  mul_196 = add_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_323: "f32[6272, 576]" = torch.ops.aten.view.default(mul_198, [6272, 576]);  mul_198 = None
        permute_217: "f32[576, 192]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_72: "f32[6272, 192]" = torch.ops.aten.addmm.default(arg69_1, view_323, permute_217);  arg69_1 = view_323 = permute_217 = None
        view_324: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_72, [8, 28, 28, 192]);  addmm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:136 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_224: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_220, view_324);  add_220 = view_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:381 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_218: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_224, [0, 3, 1, 2]);  add_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:382 in forward, code: x = self.proj(x)  # B, C, H, W
        convolution_9: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(permute_218, arg70_1, arg71_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_218 = arg70_1 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:383 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_219: "f32[8, 14, 14, 384]" = torch.ops.aten.permute.default(convolution_9, [0, 2, 3, 1]);  convolution_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:637 in forward_tokens, code: x = x + self.pos_embed
        add_225: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(permute_219, arg72_1);  permute_219 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_174: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_225, memory_format = torch.contiguous_format)
        var_mean_49 = torch.ops.aten.var_mean.correction(clone_174, [3], correction = 0, keepdim = True)
        getitem_202: "f32[8, 14, 14, 1]" = var_mean_49[0]
        getitem_203: "f32[8, 14, 14, 1]" = var_mean_49[1];  var_mean_49 = None
        add_226: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-05);  getitem_202 = None
        rsqrt_49: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_65: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_174, getitem_203);  clone_174 = getitem_203 = None
        mul_199: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_49);  sub_65 = rsqrt_49 = None
        mul_200: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_199, arg73_1);  mul_199 = arg73_1 = None
        add_227: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_200, arg74_1);  mul_200 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_220: "f32[384, 1152]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        view_325: "f32[1568, 384]" = torch.ops.aten.view.default(add_227, [1568, 384]);  add_227 = None
        mm_35: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_325, permute_220);  view_325 = permute_220 = None
        view_326: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_35, [8, 14, 14, 1152]);  mm_35 = None
        view_327: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_326, [8, 196, 3, 12, 32]);  view_326 = None
        permute_221: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_327, [2, 0, 3, 1, 4]);  view_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_221);  permute_221 = None
        getitem_204: "f32[8, 12, 196, 32]" = unbind_16[0]
        getitem_205: "f32[8, 12, 196, 32]" = unbind_16[1]
        getitem_206: "f32[8, 12, 196, 32]" = unbind_16[2];  unbind_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_204, getitem_205, getitem_206, None, False);  getitem_204 = getitem_205 = getitem_206 = None
        getitem_207: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_222: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None
        view_328: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_222, [8, 14, 14, 384]);  permute_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_329: "f32[1568, 384]" = torch.ops.aten.view.default(view_328, [1568, 384]);  view_328 = None
        permute_223: "f32[384, 384]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_73: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg77_1, view_329, permute_223);  arg77_1 = view_329 = permute_223 = None
        view_330: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_73, [8, 14, 14, 384]);  addmm_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_228: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_225, view_330);  add_225 = view_330 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_176: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_228, memory_format = torch.contiguous_format)
        var_mean_50 = torch.ops.aten.var_mean.correction(clone_176, [3], correction = 0, keepdim = True)
        getitem_211: "f32[8, 14, 14, 1]" = var_mean_50[0]
        getitem_212: "f32[8, 14, 14, 1]" = var_mean_50[1];  var_mean_50 = None
        add_229: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_211, 1e-05);  getitem_211 = None
        rsqrt_50: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_66: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_176, getitem_212);  clone_176 = getitem_212 = None
        mul_201: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_50);  sub_66 = rsqrt_50 = None
        mul_202: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_201, arg78_1);  mul_201 = arg78_1 = None
        add_230: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_202, arg79_1);  mul_202 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_331: "f32[1568, 384]" = torch.ops.aten.view.default(add_230, [1568, 384]);  add_230 = None
        permute_224: "f32[384, 1152]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_74: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg81_1, view_331, permute_224);  arg81_1 = view_331 = permute_224 = None
        view_332: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_74, [8, 14, 14, 1152]);  addmm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_203: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.5)
        mul_204: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476);  view_332 = None
        erf_24: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_204);  mul_204 = None
        add_231: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_205: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_203, add_231);  mul_203 = add_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_333: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_205, [1568, 1152]);  mul_205 = None
        permute_225: "f32[1152, 384]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_75: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg83_1, view_333, permute_225);  arg83_1 = view_333 = permute_225 = None
        view_334: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_75, [8, 14, 14, 384]);  addmm_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_232: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_228, view_334);  add_228 = view_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_179: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_232, memory_format = torch.contiguous_format)
        var_mean_51 = torch.ops.aten.var_mean.correction(clone_179, [3], correction = 0, keepdim = True)
        getitem_213: "f32[8, 14, 14, 1]" = var_mean_51[0]
        getitem_214: "f32[8, 14, 14, 1]" = var_mean_51[1];  var_mean_51 = None
        add_233: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_213, 1e-05);  getitem_213 = None
        rsqrt_51: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        sub_67: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_179, getitem_214);  clone_179 = getitem_214 = None
        mul_206: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_51);  sub_67 = rsqrt_51 = None
        mul_207: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_206, arg84_1);  mul_206 = arg84_1 = None
        add_234: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_207, arg85_1);  mul_207 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_226: "f32[384, 1152]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        view_335: "f32[1568, 384]" = torch.ops.aten.view.default(add_234, [1568, 384]);  add_234 = None
        mm_36: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_335, permute_226);  view_335 = permute_226 = None
        view_336: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_36, [8, 14, 14, 1152]);  mm_36 = None
        view_337: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_336, [8, 196, 3, 12, 32]);  view_336 = None
        permute_227: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_337, [2, 0, 3, 1, 4]);  view_337 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_227);  permute_227 = None
        getitem_215: "f32[8, 12, 196, 32]" = unbind_17[0]
        getitem_216: "f32[8, 12, 196, 32]" = unbind_17[1]
        getitem_217: "f32[8, 12, 196, 32]" = unbind_17[2];  unbind_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_215, getitem_216, getitem_217, None, False);  getitem_215 = getitem_216 = getitem_217 = None
        getitem_218: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_228: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
        view_338: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_228, [8, 14, 14, 384]);  permute_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_339: "f32[1568, 384]" = torch.ops.aten.view.default(view_338, [1568, 384]);  view_338 = None
        permute_229: "f32[384, 384]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_76: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg88_1, view_339, permute_229);  arg88_1 = view_339 = permute_229 = None
        view_340: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_76, [8, 14, 14, 384]);  addmm_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_235: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_232, view_340);  add_232 = view_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_181: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_235, memory_format = torch.contiguous_format)
        var_mean_52 = torch.ops.aten.var_mean.correction(clone_181, [3], correction = 0, keepdim = True)
        getitem_222: "f32[8, 14, 14, 1]" = var_mean_52[0]
        getitem_223: "f32[8, 14, 14, 1]" = var_mean_52[1];  var_mean_52 = None
        add_236: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_52: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_68: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_181, getitem_223);  clone_181 = getitem_223 = None
        mul_208: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_52);  sub_68 = rsqrt_52 = None
        mul_209: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_208, arg89_1);  mul_208 = arg89_1 = None
        add_237: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_209, arg90_1);  mul_209 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_341: "f32[1568, 384]" = torch.ops.aten.view.default(add_237, [1568, 384]);  add_237 = None
        permute_230: "f32[384, 1152]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_77: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg92_1, view_341, permute_230);  arg92_1 = view_341 = permute_230 = None
        view_342: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_77, [8, 14, 14, 1152]);  addmm_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_210: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_342, 0.5)
        mul_211: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_342, 0.7071067811865476);  view_342 = None
        erf_25: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
        add_238: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_212: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_210, add_238);  mul_210 = add_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_343: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_212, [1568, 1152]);  mul_212 = None
        permute_231: "f32[1152, 384]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_78: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg94_1, view_343, permute_231);  arg94_1 = view_343 = permute_231 = None
        view_344: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_78, [8, 14, 14, 384]);  addmm_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_239: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_235, view_344);  add_235 = view_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_184: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_184, [3], correction = 0, keepdim = True)
        getitem_224: "f32[8, 14, 14, 1]" = var_mean_53[0]
        getitem_225: "f32[8, 14, 14, 1]" = var_mean_53[1];  var_mean_53 = None
        add_240: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_53: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_69: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_184, getitem_225);  clone_184 = getitem_225 = None
        mul_213: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_53);  sub_69 = rsqrt_53 = None
        mul_214: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_213, arg95_1);  mul_213 = arg95_1 = None
        add_241: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_214, arg96_1);  mul_214 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_232: "f32[384, 1152]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        view_345: "f32[1568, 384]" = torch.ops.aten.view.default(add_241, [1568, 384]);  add_241 = None
        mm_37: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_345, permute_232);  view_345 = permute_232 = None
        view_346: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_37, [8, 14, 14, 1152]);  mm_37 = None
        view_347: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_346, [8, 196, 3, 12, 32]);  view_346 = None
        permute_233: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_347, [2, 0, 3, 1, 4]);  view_347 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_18 = torch.ops.aten.unbind.int(permute_233);  permute_233 = None
        getitem_226: "f32[8, 12, 196, 32]" = unbind_18[0]
        getitem_227: "f32[8, 12, 196, 32]" = unbind_18[1]
        getitem_228: "f32[8, 12, 196, 32]" = unbind_18[2];  unbind_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_226, getitem_227, getitem_228, None, False);  getitem_226 = getitem_227 = getitem_228 = None
        getitem_229: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_234: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_229, [0, 2, 1, 3]);  getitem_229 = None
        view_348: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_234, [8, 14, 14, 384]);  permute_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_349: "f32[1568, 384]" = torch.ops.aten.view.default(view_348, [1568, 384]);  view_348 = None
        permute_235: "f32[384, 384]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_79: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg99_1, view_349, permute_235);  arg99_1 = view_349 = permute_235 = None
        view_350: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_79, [8, 14, 14, 384]);  addmm_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_242: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_239, view_350);  add_239 = view_350 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_186: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_242, memory_format = torch.contiguous_format)
        var_mean_54 = torch.ops.aten.var_mean.correction(clone_186, [3], correction = 0, keepdim = True)
        getitem_233: "f32[8, 14, 14, 1]" = var_mean_54[0]
        getitem_234: "f32[8, 14, 14, 1]" = var_mean_54[1];  var_mean_54 = None
        add_243: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_233, 1e-05);  getitem_233 = None
        rsqrt_54: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        sub_70: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_186, getitem_234);  clone_186 = getitem_234 = None
        mul_215: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_54);  sub_70 = rsqrt_54 = None
        mul_216: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_215, arg100_1);  mul_215 = arg100_1 = None
        add_244: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_216, arg101_1);  mul_216 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_351: "f32[1568, 384]" = torch.ops.aten.view.default(add_244, [1568, 384]);  add_244 = None
        permute_236: "f32[384, 1152]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_80: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg103_1, view_351, permute_236);  arg103_1 = view_351 = permute_236 = None
        view_352: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_80, [8, 14, 14, 1152]);  addmm_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_217: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_352, 0.5)
        mul_218: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_352, 0.7071067811865476);  view_352 = None
        erf_26: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
        add_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_219: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_217, add_245);  mul_217 = add_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_353: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_219, [1568, 1152]);  mul_219 = None
        permute_237: "f32[1152, 384]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_81: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg105_1, view_353, permute_237);  arg105_1 = view_353 = permute_237 = None
        view_354: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_81, [8, 14, 14, 384]);  addmm_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_246: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_242, view_354);  add_242 = view_354 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_189: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_246, memory_format = torch.contiguous_format)
        var_mean_55 = torch.ops.aten.var_mean.correction(clone_189, [3], correction = 0, keepdim = True)
        getitem_235: "f32[8, 14, 14, 1]" = var_mean_55[0]
        getitem_236: "f32[8, 14, 14, 1]" = var_mean_55[1];  var_mean_55 = None
        add_247: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_235, 1e-05);  getitem_235 = None
        rsqrt_55: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_71: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_189, getitem_236);  clone_189 = getitem_236 = None
        mul_220: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_55);  sub_71 = rsqrt_55 = None
        mul_221: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_220, arg106_1);  mul_220 = arg106_1 = None
        add_248: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_221, arg107_1);  mul_221 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_238: "f32[384, 1152]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        view_355: "f32[1568, 384]" = torch.ops.aten.view.default(add_248, [1568, 384]);  add_248 = None
        mm_38: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_355, permute_238);  view_355 = permute_238 = None
        view_356: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_38, [8, 14, 14, 1152]);  mm_38 = None
        view_357: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_356, [8, 196, 3, 12, 32]);  view_356 = None
        permute_239: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_357, [2, 0, 3, 1, 4]);  view_357 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_19 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
        getitem_237: "f32[8, 12, 196, 32]" = unbind_19[0]
        getitem_238: "f32[8, 12, 196, 32]" = unbind_19[1]
        getitem_239: "f32[8, 12, 196, 32]" = unbind_19[2];  unbind_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_237, getitem_238, getitem_239, None, False);  getitem_237 = getitem_238 = getitem_239 = None
        getitem_240: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_240: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_240, [0, 2, 1, 3]);  getitem_240 = None
        view_358: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_240, [8, 14, 14, 384]);  permute_240 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_359: "f32[1568, 384]" = torch.ops.aten.view.default(view_358, [1568, 384]);  view_358 = None
        permute_241: "f32[384, 384]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_82: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg110_1, view_359, permute_241);  arg110_1 = view_359 = permute_241 = None
        view_360: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_82, [8, 14, 14, 384]);  addmm_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_249: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_246, view_360);  add_246 = view_360 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_191: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_249, memory_format = torch.contiguous_format)
        var_mean_56 = torch.ops.aten.var_mean.correction(clone_191, [3], correction = 0, keepdim = True)
        getitem_244: "f32[8, 14, 14, 1]" = var_mean_56[0]
        getitem_245: "f32[8, 14, 14, 1]" = var_mean_56[1];  var_mean_56 = None
        add_250: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-05);  getitem_244 = None
        rsqrt_56: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_72: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_191, getitem_245);  clone_191 = getitem_245 = None
        mul_222: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_56);  sub_72 = rsqrt_56 = None
        mul_223: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_222, arg111_1);  mul_222 = arg111_1 = None
        add_251: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_223, arg112_1);  mul_223 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_361: "f32[1568, 384]" = torch.ops.aten.view.default(add_251, [1568, 384]);  add_251 = None
        permute_242: "f32[384, 1152]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_83: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg114_1, view_361, permute_242);  arg114_1 = view_361 = permute_242 = None
        view_362: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_83, [8, 14, 14, 1152]);  addmm_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_224: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_362, 0.5)
        mul_225: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_362, 0.7071067811865476);  view_362 = None
        erf_27: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_225);  mul_225 = None
        add_252: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_226: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_224, add_252);  mul_224 = add_252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_363: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_226, [1568, 1152]);  mul_226 = None
        permute_243: "f32[1152, 384]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_84: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg116_1, view_363, permute_243);  arg116_1 = view_363 = permute_243 = None
        view_364: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_84, [8, 14, 14, 384]);  addmm_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_253: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_249, view_364);  add_249 = view_364 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_194: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_253, memory_format = torch.contiguous_format)
        var_mean_57 = torch.ops.aten.var_mean.correction(clone_194, [3], correction = 0, keepdim = True)
        getitem_246: "f32[8, 14, 14, 1]" = var_mean_57[0]
        getitem_247: "f32[8, 14, 14, 1]" = var_mean_57[1];  var_mean_57 = None
        add_254: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_57: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        sub_73: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_194, getitem_247);  clone_194 = getitem_247 = None
        mul_227: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_57);  sub_73 = rsqrt_57 = None
        mul_228: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_227, arg117_1);  mul_227 = arg117_1 = None
        add_255: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_228, arg118_1);  mul_228 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_244: "f32[384, 1152]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        view_365: "f32[1568, 384]" = torch.ops.aten.view.default(add_255, [1568, 384]);  add_255 = None
        mm_39: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_365, permute_244);  view_365 = permute_244 = None
        view_366: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_39, [8, 14, 14, 1152]);  mm_39 = None
        view_367: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_366, [8, 196, 3, 12, 32]);  view_366 = None
        permute_245: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_367, [2, 0, 3, 1, 4]);  view_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_20 = torch.ops.aten.unbind.int(permute_245);  permute_245 = None
        getitem_248: "f32[8, 12, 196, 32]" = unbind_20[0]
        getitem_249: "f32[8, 12, 196, 32]" = unbind_20[1]
        getitem_250: "f32[8, 12, 196, 32]" = unbind_20[2];  unbind_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_248, getitem_249, getitem_250, None, False);  getitem_248 = getitem_249 = getitem_250 = None
        getitem_251: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_246: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_251, [0, 2, 1, 3]);  getitem_251 = None
        view_368: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_246, [8, 14, 14, 384]);  permute_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_369: "f32[1568, 384]" = torch.ops.aten.view.default(view_368, [1568, 384]);  view_368 = None
        permute_247: "f32[384, 384]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_85: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg121_1, view_369, permute_247);  arg121_1 = view_369 = permute_247 = None
        view_370: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_85, [8, 14, 14, 384]);  addmm_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_256: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_253, view_370);  add_253 = view_370 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_196: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_256, memory_format = torch.contiguous_format)
        var_mean_58 = torch.ops.aten.var_mean.correction(clone_196, [3], correction = 0, keepdim = True)
        getitem_255: "f32[8, 14, 14, 1]" = var_mean_58[0]
        getitem_256: "f32[8, 14, 14, 1]" = var_mean_58[1];  var_mean_58 = None
        add_257: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_255, 1e-05);  getitem_255 = None
        rsqrt_58: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        sub_74: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_196, getitem_256);  clone_196 = getitem_256 = None
        mul_229: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_58);  sub_74 = rsqrt_58 = None
        mul_230: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_229, arg122_1);  mul_229 = arg122_1 = None
        add_258: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_230, arg123_1);  mul_230 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_371: "f32[1568, 384]" = torch.ops.aten.view.default(add_258, [1568, 384]);  add_258 = None
        permute_248: "f32[384, 1152]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_86: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg125_1, view_371, permute_248);  arg125_1 = view_371 = permute_248 = None
        view_372: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_86, [8, 14, 14, 1152]);  addmm_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_231: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_372, 0.5)
        mul_232: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_372, 0.7071067811865476);  view_372 = None
        erf_28: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_259: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_233: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_231, add_259);  mul_231 = add_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_373: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_233, [1568, 1152]);  mul_233 = None
        permute_249: "f32[1152, 384]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_87: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg127_1, view_373, permute_249);  arg127_1 = view_373 = permute_249 = None
        view_374: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_87, [8, 14, 14, 384]);  addmm_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_260: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_256, view_374);  add_256 = view_374 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_199: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
        var_mean_59 = torch.ops.aten.var_mean.correction(clone_199, [3], correction = 0, keepdim = True)
        getitem_257: "f32[8, 14, 14, 1]" = var_mean_59[0]
        getitem_258: "f32[8, 14, 14, 1]" = var_mean_59[1];  var_mean_59 = None
        add_261: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_257, 1e-05);  getitem_257 = None
        rsqrt_59: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_75: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_199, getitem_258);  clone_199 = getitem_258 = None
        mul_234: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_59);  sub_75 = rsqrt_59 = None
        mul_235: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_234, arg128_1);  mul_234 = arg128_1 = None
        add_262: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_235, arg129_1);  mul_235 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_250: "f32[384, 1152]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        view_375: "f32[1568, 384]" = torch.ops.aten.view.default(add_262, [1568, 384]);  add_262 = None
        mm_40: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_375, permute_250);  view_375 = permute_250 = None
        view_376: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_40, [8, 14, 14, 1152]);  mm_40 = None
        view_377: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_376, [8, 196, 3, 12, 32]);  view_376 = None
        permute_251: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_377, [2, 0, 3, 1, 4]);  view_377 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_21 = torch.ops.aten.unbind.int(permute_251);  permute_251 = None
        getitem_259: "f32[8, 12, 196, 32]" = unbind_21[0]
        getitem_260: "f32[8, 12, 196, 32]" = unbind_21[1]
        getitem_261: "f32[8, 12, 196, 32]" = unbind_21[2];  unbind_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_259, getitem_260, getitem_261, None, False);  getitem_259 = getitem_260 = getitem_261 = None
        getitem_262: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_252: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_262, [0, 2, 1, 3]);  getitem_262 = None
        view_378: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_252, [8, 14, 14, 384]);  permute_252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_379: "f32[1568, 384]" = torch.ops.aten.view.default(view_378, [1568, 384]);  view_378 = None
        permute_253: "f32[384, 384]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_88: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg132_1, view_379, permute_253);  arg132_1 = view_379 = permute_253 = None
        view_380: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_88, [8, 14, 14, 384]);  addmm_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_263: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_260, view_380);  add_260 = view_380 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_201: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_263, memory_format = torch.contiguous_format)
        var_mean_60 = torch.ops.aten.var_mean.correction(clone_201, [3], correction = 0, keepdim = True)
        getitem_266: "f32[8, 14, 14, 1]" = var_mean_60[0]
        getitem_267: "f32[8, 14, 14, 1]" = var_mean_60[1];  var_mean_60 = None
        add_264: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-05);  getitem_266 = None
        rsqrt_60: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_76: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_201, getitem_267);  clone_201 = getitem_267 = None
        mul_236: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_60);  sub_76 = rsqrt_60 = None
        mul_237: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_236, arg133_1);  mul_236 = arg133_1 = None
        add_265: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_237, arg134_1);  mul_237 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_381: "f32[1568, 384]" = torch.ops.aten.view.default(add_265, [1568, 384]);  add_265 = None
        permute_254: "f32[384, 1152]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_89: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg136_1, view_381, permute_254);  arg136_1 = view_381 = permute_254 = None
        view_382: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_89, [8, 14, 14, 1152]);  addmm_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_238: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_382, 0.5)
        mul_239: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_382, 0.7071067811865476);  view_382 = None
        erf_29: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_239);  mul_239 = None
        add_266: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_240: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_238, add_266);  mul_238 = add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_383: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_240, [1568, 1152]);  mul_240 = None
        permute_255: "f32[1152, 384]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_90: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg138_1, view_383, permute_255);  arg138_1 = view_383 = permute_255 = None
        view_384: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_90, [8, 14, 14, 384]);  addmm_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_267: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_263, view_384);  add_263 = view_384 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_204: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_267, memory_format = torch.contiguous_format)
        var_mean_61 = torch.ops.aten.var_mean.correction(clone_204, [3], correction = 0, keepdim = True)
        getitem_268: "f32[8, 14, 14, 1]" = var_mean_61[0]
        getitem_269: "f32[8, 14, 14, 1]" = var_mean_61[1];  var_mean_61 = None
        add_268: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
        rsqrt_61: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_77: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_204, getitem_269);  clone_204 = getitem_269 = None
        mul_241: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_61);  sub_77 = rsqrt_61 = None
        mul_242: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_241, arg139_1);  mul_241 = arg139_1 = None
        add_269: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_242, arg140_1);  mul_242 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_256: "f32[384, 1152]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        view_385: "f32[1568, 384]" = torch.ops.aten.view.default(add_269, [1568, 384]);  add_269 = None
        mm_41: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_385, permute_256);  view_385 = permute_256 = None
        view_386: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_41, [8, 14, 14, 1152]);  mm_41 = None
        view_387: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_386, [8, 196, 3, 12, 32]);  view_386 = None
        permute_257: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_387, [2, 0, 3, 1, 4]);  view_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_22 = torch.ops.aten.unbind.int(permute_257);  permute_257 = None
        getitem_270: "f32[8, 12, 196, 32]" = unbind_22[0]
        getitem_271: "f32[8, 12, 196, 32]" = unbind_22[1]
        getitem_272: "f32[8, 12, 196, 32]" = unbind_22[2];  unbind_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_270, getitem_271, getitem_272, None, False);  getitem_270 = getitem_271 = getitem_272 = None
        getitem_273: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_258: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_273, [0, 2, 1, 3]);  getitem_273 = None
        view_388: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_258, [8, 14, 14, 384]);  permute_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_389: "f32[1568, 384]" = torch.ops.aten.view.default(view_388, [1568, 384]);  view_388 = None
        permute_259: "f32[384, 384]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_91: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg143_1, view_389, permute_259);  arg143_1 = view_389 = permute_259 = None
        view_390: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_91, [8, 14, 14, 384]);  addmm_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_270: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_267, view_390);  add_267 = view_390 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_206: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_270, memory_format = torch.contiguous_format)
        var_mean_62 = torch.ops.aten.var_mean.correction(clone_206, [3], correction = 0, keepdim = True)
        getitem_277: "f32[8, 14, 14, 1]" = var_mean_62[0]
        getitem_278: "f32[8, 14, 14, 1]" = var_mean_62[1];  var_mean_62 = None
        add_271: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_277, 1e-05);  getitem_277 = None
        rsqrt_62: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_78: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_206, getitem_278);  clone_206 = getitem_278 = None
        mul_243: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_62);  sub_78 = rsqrt_62 = None
        mul_244: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_243, arg144_1);  mul_243 = arg144_1 = None
        add_272: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_244, arg145_1);  mul_244 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_391: "f32[1568, 384]" = torch.ops.aten.view.default(add_272, [1568, 384]);  add_272 = None
        permute_260: "f32[384, 1152]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_92: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg147_1, view_391, permute_260);  arg147_1 = view_391 = permute_260 = None
        view_392: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_92, [8, 14, 14, 1152]);  addmm_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_392, 0.5)
        mul_246: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_392, 0.7071067811865476);  view_392 = None
        erf_30: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_246);  mul_246 = None
        add_273: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_247: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_245, add_273);  mul_245 = add_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_393: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_247, [1568, 1152]);  mul_247 = None
        permute_261: "f32[1152, 384]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_93: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg149_1, view_393, permute_261);  arg149_1 = view_393 = permute_261 = None
        view_394: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_93, [8, 14, 14, 384]);  addmm_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_274: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_270, view_394);  add_270 = view_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_209: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_274, memory_format = torch.contiguous_format)
        var_mean_63 = torch.ops.aten.var_mean.correction(clone_209, [3], correction = 0, keepdim = True)
        getitem_279: "f32[8, 14, 14, 1]" = var_mean_63[0]
        getitem_280: "f32[8, 14, 14, 1]" = var_mean_63[1];  var_mean_63 = None
        add_275: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_279, 1e-05);  getitem_279 = None
        rsqrt_63: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        sub_79: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_209, getitem_280);  clone_209 = getitem_280 = None
        mul_248: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_63);  sub_79 = rsqrt_63 = None
        mul_249: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_248, arg150_1);  mul_248 = arg150_1 = None
        add_276: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_249, arg151_1);  mul_249 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_262: "f32[384, 1152]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        view_395: "f32[1568, 384]" = torch.ops.aten.view.default(add_276, [1568, 384]);  add_276 = None
        mm_42: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_395, permute_262);  view_395 = permute_262 = None
        view_396: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_42, [8, 14, 14, 1152]);  mm_42 = None
        view_397: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_396, [8, 196, 3, 12, 32]);  view_396 = None
        permute_263: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_397, [2, 0, 3, 1, 4]);  view_397 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_23 = torch.ops.aten.unbind.int(permute_263);  permute_263 = None
        getitem_281: "f32[8, 12, 196, 32]" = unbind_23[0]
        getitem_282: "f32[8, 12, 196, 32]" = unbind_23[1]
        getitem_283: "f32[8, 12, 196, 32]" = unbind_23[2];  unbind_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_281, getitem_282, getitem_283, None, False);  getitem_281 = getitem_282 = getitem_283 = None
        getitem_284: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_264: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_284, [0, 2, 1, 3]);  getitem_284 = None
        view_398: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_264, [8, 14, 14, 384]);  permute_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_399: "f32[1568, 384]" = torch.ops.aten.view.default(view_398, [1568, 384]);  view_398 = None
        permute_265: "f32[384, 384]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_94: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg154_1, view_399, permute_265);  arg154_1 = view_399 = permute_265 = None
        view_400: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_94, [8, 14, 14, 384]);  addmm_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_277: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_274, view_400);  add_274 = view_400 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_211: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_277, memory_format = torch.contiguous_format)
        var_mean_64 = torch.ops.aten.var_mean.correction(clone_211, [3], correction = 0, keepdim = True)
        getitem_288: "f32[8, 14, 14, 1]" = var_mean_64[0]
        getitem_289: "f32[8, 14, 14, 1]" = var_mean_64[1];  var_mean_64 = None
        add_278: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_64: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        sub_80: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_211, getitem_289);  clone_211 = getitem_289 = None
        mul_250: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_64);  sub_80 = rsqrt_64 = None
        mul_251: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_250, arg155_1);  mul_250 = arg155_1 = None
        add_279: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_251, arg156_1);  mul_251 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_401: "f32[1568, 384]" = torch.ops.aten.view.default(add_279, [1568, 384]);  add_279 = None
        permute_266: "f32[384, 1152]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_95: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg158_1, view_401, permute_266);  arg158_1 = view_401 = permute_266 = None
        view_402: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_95, [8, 14, 14, 1152]);  addmm_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_252: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_402, 0.5)
        mul_253: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_402, 0.7071067811865476);  view_402 = None
        erf_31: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_253);  mul_253 = None
        add_280: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_254: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_252, add_280);  mul_252 = add_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_403: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_254, [1568, 1152]);  mul_254 = None
        permute_267: "f32[1152, 384]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_96: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg160_1, view_403, permute_267);  arg160_1 = view_403 = permute_267 = None
        view_404: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_96, [8, 14, 14, 384]);  addmm_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_281: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_277, view_404);  add_277 = view_404 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_214: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_281, memory_format = torch.contiguous_format)
        var_mean_65 = torch.ops.aten.var_mean.correction(clone_214, [3], correction = 0, keepdim = True)
        getitem_290: "f32[8, 14, 14, 1]" = var_mean_65[0]
        getitem_291: "f32[8, 14, 14, 1]" = var_mean_65[1];  var_mean_65 = None
        add_282: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-05);  getitem_290 = None
        rsqrt_65: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_81: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_214, getitem_291);  clone_214 = getitem_291 = None
        mul_255: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_65);  sub_81 = rsqrt_65 = None
        mul_256: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_255, arg161_1);  mul_255 = arg161_1 = None
        add_283: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_256, arg162_1);  mul_256 = arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_268: "f32[384, 1152]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        view_405: "f32[1568, 384]" = torch.ops.aten.view.default(add_283, [1568, 384]);  add_283 = None
        mm_43: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_405, permute_268);  view_405 = permute_268 = None
        view_406: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_43, [8, 14, 14, 1152]);  mm_43 = None
        view_407: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_406, [8, 196, 3, 12, 32]);  view_406 = None
        permute_269: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_407, [2, 0, 3, 1, 4]);  view_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_24 = torch.ops.aten.unbind.int(permute_269);  permute_269 = None
        getitem_292: "f32[8, 12, 196, 32]" = unbind_24[0]
        getitem_293: "f32[8, 12, 196, 32]" = unbind_24[1]
        getitem_294: "f32[8, 12, 196, 32]" = unbind_24[2];  unbind_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_292, getitem_293, getitem_294, None, False);  getitem_292 = getitem_293 = getitem_294 = None
        getitem_295: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_270: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_295, [0, 2, 1, 3]);  getitem_295 = None
        view_408: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_270, [8, 14, 14, 384]);  permute_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_409: "f32[1568, 384]" = torch.ops.aten.view.default(view_408, [1568, 384]);  view_408 = None
        permute_271: "f32[384, 384]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_97: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg165_1, view_409, permute_271);  arg165_1 = view_409 = permute_271 = None
        view_410: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_97, [8, 14, 14, 384]);  addmm_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_284: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_281, view_410);  add_281 = view_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_216: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_284, memory_format = torch.contiguous_format)
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_216, [3], correction = 0, keepdim = True)
        getitem_299: "f32[8, 14, 14, 1]" = var_mean_66[0]
        getitem_300: "f32[8, 14, 14, 1]" = var_mean_66[1];  var_mean_66 = None
        add_285: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_299, 1e-05);  getitem_299 = None
        rsqrt_66: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        sub_82: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_216, getitem_300);  clone_216 = getitem_300 = None
        mul_257: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_66);  sub_82 = rsqrt_66 = None
        mul_258: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_257, arg166_1);  mul_257 = arg166_1 = None
        add_286: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_258, arg167_1);  mul_258 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_411: "f32[1568, 384]" = torch.ops.aten.view.default(add_286, [1568, 384]);  add_286 = None
        permute_272: "f32[384, 1152]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_98: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg169_1, view_411, permute_272);  arg169_1 = view_411 = permute_272 = None
        view_412: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_98, [8, 14, 14, 1152]);  addmm_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_259: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_412, 0.5)
        mul_260: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_412, 0.7071067811865476);  view_412 = None
        erf_32: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_260);  mul_260 = None
        add_287: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_261: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_259, add_287);  mul_259 = add_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_413: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_261, [1568, 1152]);  mul_261 = None
        permute_273: "f32[1152, 384]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_99: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg171_1, view_413, permute_273);  arg171_1 = view_413 = permute_273 = None
        view_414: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_99, [8, 14, 14, 384]);  addmm_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_288: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_284, view_414);  add_284 = view_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_219: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_219, [3], correction = 0, keepdim = True)
        getitem_301: "f32[8, 14, 14, 1]" = var_mean_67[0]
        getitem_302: "f32[8, 14, 14, 1]" = var_mean_67[1];  var_mean_67 = None
        add_289: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_301, 1e-05);  getitem_301 = None
        rsqrt_67: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        sub_83: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_219, getitem_302);  clone_219 = getitem_302 = None
        mul_262: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_67);  sub_83 = rsqrt_67 = None
        mul_263: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_262, arg172_1);  mul_262 = arg172_1 = None
        add_290: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_263, arg173_1);  mul_263 = arg173_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_274: "f32[384, 1152]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        view_415: "f32[1568, 384]" = torch.ops.aten.view.default(add_290, [1568, 384]);  add_290 = None
        mm_44: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_415, permute_274);  view_415 = permute_274 = None
        view_416: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_44, [8, 14, 14, 1152]);  mm_44 = None
        view_417: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_416, [8, 196, 3, 12, 32]);  view_416 = None
        permute_275: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_417, [2, 0, 3, 1, 4]);  view_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_25 = torch.ops.aten.unbind.int(permute_275);  permute_275 = None
        getitem_303: "f32[8, 12, 196, 32]" = unbind_25[0]
        getitem_304: "f32[8, 12, 196, 32]" = unbind_25[1]
        getitem_305: "f32[8, 12, 196, 32]" = unbind_25[2];  unbind_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_303, getitem_304, getitem_305, None, False);  getitem_303 = getitem_304 = getitem_305 = None
        getitem_306: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_276: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_306, [0, 2, 1, 3]);  getitem_306 = None
        view_418: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_276, [8, 14, 14, 384]);  permute_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_419: "f32[1568, 384]" = torch.ops.aten.view.default(view_418, [1568, 384]);  view_418 = None
        permute_277: "f32[384, 384]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_100: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg176_1, view_419, permute_277);  arg176_1 = view_419 = permute_277 = None
        view_420: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_100, [8, 14, 14, 384]);  addmm_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_291: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_288, view_420);  add_288 = view_420 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_221: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_291, memory_format = torch.contiguous_format)
        var_mean_68 = torch.ops.aten.var_mean.correction(clone_221, [3], correction = 0, keepdim = True)
        getitem_310: "f32[8, 14, 14, 1]" = var_mean_68[0]
        getitem_311: "f32[8, 14, 14, 1]" = var_mean_68[1];  var_mean_68 = None
        add_292: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_68: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_84: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_221, getitem_311);  clone_221 = getitem_311 = None
        mul_264: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_68);  sub_84 = rsqrt_68 = None
        mul_265: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_264, arg177_1);  mul_264 = arg177_1 = None
        add_293: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_265, arg178_1);  mul_265 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_421: "f32[1568, 384]" = torch.ops.aten.view.default(add_293, [1568, 384]);  add_293 = None
        permute_278: "f32[384, 1152]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_101: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg180_1, view_421, permute_278);  arg180_1 = view_421 = permute_278 = None
        view_422: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_101, [8, 14, 14, 1152]);  addmm_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_266: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_422, 0.5)
        mul_267: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_422, 0.7071067811865476);  view_422 = None
        erf_33: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_267);  mul_267 = None
        add_294: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_268: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_266, add_294);  mul_266 = add_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_423: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_268, [1568, 1152]);  mul_268 = None
        permute_279: "f32[1152, 384]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_102: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg182_1, view_423, permute_279);  arg182_1 = view_423 = permute_279 = None
        view_424: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_102, [8, 14, 14, 384]);  addmm_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_295: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_291, view_424);  add_291 = view_424 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_224: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_295, memory_format = torch.contiguous_format)
        var_mean_69 = torch.ops.aten.var_mean.correction(clone_224, [3], correction = 0, keepdim = True)
        getitem_312: "f32[8, 14, 14, 1]" = var_mean_69[0]
        getitem_313: "f32[8, 14, 14, 1]" = var_mean_69[1];  var_mean_69 = None
        add_296: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_312, 1e-05);  getitem_312 = None
        rsqrt_69: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_85: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_224, getitem_313);  clone_224 = getitem_313 = None
        mul_269: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_69);  sub_85 = rsqrt_69 = None
        mul_270: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_269, arg183_1);  mul_269 = arg183_1 = None
        add_297: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_270, arg184_1);  mul_270 = arg184_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_280: "f32[384, 1152]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        view_425: "f32[1568, 384]" = torch.ops.aten.view.default(add_297, [1568, 384]);  add_297 = None
        mm_45: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_425, permute_280);  view_425 = permute_280 = None
        view_426: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_45, [8, 14, 14, 1152]);  mm_45 = None
        view_427: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_426, [8, 196, 3, 12, 32]);  view_426 = None
        permute_281: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_427, [2, 0, 3, 1, 4]);  view_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_26 = torch.ops.aten.unbind.int(permute_281);  permute_281 = None
        getitem_314: "f32[8, 12, 196, 32]" = unbind_26[0]
        getitem_315: "f32[8, 12, 196, 32]" = unbind_26[1]
        getitem_316: "f32[8, 12, 196, 32]" = unbind_26[2];  unbind_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_314, getitem_315, getitem_316, None, False);  getitem_314 = getitem_315 = getitem_316 = None
        getitem_317: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_24[0];  _scaled_dot_product_efficient_attention_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_282: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_317, [0, 2, 1, 3]);  getitem_317 = None
        view_428: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_282, [8, 14, 14, 384]);  permute_282 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_429: "f32[1568, 384]" = torch.ops.aten.view.default(view_428, [1568, 384]);  view_428 = None
        permute_283: "f32[384, 384]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_103: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg187_1, view_429, permute_283);  arg187_1 = view_429 = permute_283 = None
        view_430: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_103, [8, 14, 14, 384]);  addmm_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_298: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_295, view_430);  add_295 = view_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_226: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_298, memory_format = torch.contiguous_format)
        var_mean_70 = torch.ops.aten.var_mean.correction(clone_226, [3], correction = 0, keepdim = True)
        getitem_321: "f32[8, 14, 14, 1]" = var_mean_70[0]
        getitem_322: "f32[8, 14, 14, 1]" = var_mean_70[1];  var_mean_70 = None
        add_299: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_321, 1e-05);  getitem_321 = None
        rsqrt_70: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_86: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_226, getitem_322);  clone_226 = getitem_322 = None
        mul_271: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_70);  sub_86 = rsqrt_70 = None
        mul_272: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_271, arg188_1);  mul_271 = arg188_1 = None
        add_300: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_272, arg189_1);  mul_272 = arg189_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_431: "f32[1568, 384]" = torch.ops.aten.view.default(add_300, [1568, 384]);  add_300 = None
        permute_284: "f32[384, 1152]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_104: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg191_1, view_431, permute_284);  arg191_1 = view_431 = permute_284 = None
        view_432: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_104, [8, 14, 14, 1152]);  addmm_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_273: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_432, 0.5)
        mul_274: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_432, 0.7071067811865476);  view_432 = None
        erf_34: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_274);  mul_274 = None
        add_301: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_275: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_273, add_301);  mul_273 = add_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_433: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_275, [1568, 1152]);  mul_275 = None
        permute_285: "f32[1152, 384]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_105: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg193_1, view_433, permute_285);  arg193_1 = view_433 = permute_285 = None
        view_434: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_105, [8, 14, 14, 384]);  addmm_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_302: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_298, view_434);  add_298 = view_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_229: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_302, memory_format = torch.contiguous_format)
        var_mean_71 = torch.ops.aten.var_mean.correction(clone_229, [3], correction = 0, keepdim = True)
        getitem_323: "f32[8, 14, 14, 1]" = var_mean_71[0]
        getitem_324: "f32[8, 14, 14, 1]" = var_mean_71[1];  var_mean_71 = None
        add_303: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_323, 1e-05);  getitem_323 = None
        rsqrt_71: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_87: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_229, getitem_324);  clone_229 = getitem_324 = None
        mul_276: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_71);  sub_87 = rsqrt_71 = None
        mul_277: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_276, arg194_1);  mul_276 = arg194_1 = None
        add_304: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_277, arg195_1);  mul_277 = arg195_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_286: "f32[384, 1152]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        view_435: "f32[1568, 384]" = torch.ops.aten.view.default(add_304, [1568, 384]);  add_304 = None
        mm_46: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_435, permute_286);  view_435 = permute_286 = None
        view_436: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_46, [8, 14, 14, 1152]);  mm_46 = None
        view_437: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_436, [8, 196, 3, 12, 32]);  view_436 = None
        permute_287: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_437, [2, 0, 3, 1, 4]);  view_437 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_27 = torch.ops.aten.unbind.int(permute_287);  permute_287 = None
        getitem_325: "f32[8, 12, 196, 32]" = unbind_27[0]
        getitem_326: "f32[8, 12, 196, 32]" = unbind_27[1]
        getitem_327: "f32[8, 12, 196, 32]" = unbind_27[2];  unbind_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_325, getitem_326, getitem_327, None, False);  getitem_325 = getitem_326 = getitem_327 = None
        getitem_328: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_25[0];  _scaled_dot_product_efficient_attention_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_288: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_328, [0, 2, 1, 3]);  getitem_328 = None
        view_438: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_288, [8, 14, 14, 384]);  permute_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_439: "f32[1568, 384]" = torch.ops.aten.view.default(view_438, [1568, 384]);  view_438 = None
        permute_289: "f32[384, 384]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_106: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg198_1, view_439, permute_289);  arg198_1 = view_439 = permute_289 = None
        view_440: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_106, [8, 14, 14, 384]);  addmm_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_305: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_302, view_440);  add_302 = view_440 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_231: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_305, memory_format = torch.contiguous_format)
        var_mean_72 = torch.ops.aten.var_mean.correction(clone_231, [3], correction = 0, keepdim = True)
        getitem_332: "f32[8, 14, 14, 1]" = var_mean_72[0]
        getitem_333: "f32[8, 14, 14, 1]" = var_mean_72[1];  var_mean_72 = None
        add_306: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_332, 1e-05);  getitem_332 = None
        rsqrt_72: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_88: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_231, getitem_333);  clone_231 = getitem_333 = None
        mul_278: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_72);  sub_88 = rsqrt_72 = None
        mul_279: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_278, arg199_1);  mul_278 = arg199_1 = None
        add_307: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_279, arg200_1);  mul_279 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_441: "f32[1568, 384]" = torch.ops.aten.view.default(add_307, [1568, 384]);  add_307 = None
        permute_290: "f32[384, 1152]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_107: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg202_1, view_441, permute_290);  arg202_1 = view_441 = permute_290 = None
        view_442: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_107, [8, 14, 14, 1152]);  addmm_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_280: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_442, 0.5)
        mul_281: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_442, 0.7071067811865476);  view_442 = None
        erf_35: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_281);  mul_281 = None
        add_308: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_282: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_280, add_308);  mul_280 = add_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_443: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_282, [1568, 1152]);  mul_282 = None
        permute_291: "f32[1152, 384]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_108: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg204_1, view_443, permute_291);  arg204_1 = view_443 = permute_291 = None
        view_444: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_108, [8, 14, 14, 384]);  addmm_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_309: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_305, view_444);  add_305 = view_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_234: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format)
        var_mean_73 = torch.ops.aten.var_mean.correction(clone_234, [3], correction = 0, keepdim = True)
        getitem_334: "f32[8, 14, 14, 1]" = var_mean_73[0]
        getitem_335: "f32[8, 14, 14, 1]" = var_mean_73[1];  var_mean_73 = None
        add_310: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_334, 1e-05);  getitem_334 = None
        rsqrt_73: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        sub_89: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_234, getitem_335);  clone_234 = getitem_335 = None
        mul_283: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_73);  sub_89 = rsqrt_73 = None
        mul_284: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_283, arg205_1);  mul_283 = arg205_1 = None
        add_311: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_284, arg206_1);  mul_284 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_292: "f32[384, 1152]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        view_445: "f32[1568, 384]" = torch.ops.aten.view.default(add_311, [1568, 384]);  add_311 = None
        mm_47: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_445, permute_292);  view_445 = permute_292 = None
        view_446: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_47, [8, 14, 14, 1152]);  mm_47 = None
        view_447: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_446, [8, 196, 3, 12, 32]);  view_446 = None
        permute_293: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_447, [2, 0, 3, 1, 4]);  view_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_28 = torch.ops.aten.unbind.int(permute_293);  permute_293 = None
        getitem_336: "f32[8, 12, 196, 32]" = unbind_28[0]
        getitem_337: "f32[8, 12, 196, 32]" = unbind_28[1]
        getitem_338: "f32[8, 12, 196, 32]" = unbind_28[2];  unbind_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_336, getitem_337, getitem_338, None, False);  getitem_336 = getitem_337 = getitem_338 = None
        getitem_339: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_26[0];  _scaled_dot_product_efficient_attention_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_294: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_339, [0, 2, 1, 3]);  getitem_339 = None
        view_448: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_294, [8, 14, 14, 384]);  permute_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_449: "f32[1568, 384]" = torch.ops.aten.view.default(view_448, [1568, 384]);  view_448 = None
        permute_295: "f32[384, 384]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_109: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg209_1, view_449, permute_295);  arg209_1 = view_449 = permute_295 = None
        view_450: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_109, [8, 14, 14, 384]);  addmm_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_312: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_309, view_450);  add_309 = view_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_236: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_312, memory_format = torch.contiguous_format)
        var_mean_74 = torch.ops.aten.var_mean.correction(clone_236, [3], correction = 0, keepdim = True)
        getitem_343: "f32[8, 14, 14, 1]" = var_mean_74[0]
        getitem_344: "f32[8, 14, 14, 1]" = var_mean_74[1];  var_mean_74 = None
        add_313: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_343, 1e-05);  getitem_343 = None
        rsqrt_74: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_90: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_236, getitem_344);  clone_236 = getitem_344 = None
        mul_285: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_74);  sub_90 = rsqrt_74 = None
        mul_286: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_285, arg210_1);  mul_285 = arg210_1 = None
        add_314: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_286, arg211_1);  mul_286 = arg211_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_451: "f32[1568, 384]" = torch.ops.aten.view.default(add_314, [1568, 384]);  add_314 = None
        permute_296: "f32[384, 1152]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_110: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg213_1, view_451, permute_296);  arg213_1 = view_451 = permute_296 = None
        view_452: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_110, [8, 14, 14, 1152]);  addmm_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_287: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_452, 0.5)
        mul_288: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_452, 0.7071067811865476);  view_452 = None
        erf_36: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_288);  mul_288 = None
        add_315: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_289: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_287, add_315);  mul_287 = add_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_453: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_289, [1568, 1152]);  mul_289 = None
        permute_297: "f32[1152, 384]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_111: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg215_1, view_453, permute_297);  arg215_1 = view_453 = permute_297 = None
        view_454: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_111, [8, 14, 14, 384]);  addmm_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_316: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_312, view_454);  add_312 = view_454 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        clone_239: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_316, memory_format = torch.contiguous_format)
        var_mean_75 = torch.ops.aten.var_mean.correction(clone_239, [3], correction = 0, keepdim = True)
        getitem_345: "f32[8, 14, 14, 1]" = var_mean_75[0]
        getitem_346: "f32[8, 14, 14, 1]" = var_mean_75[1];  var_mean_75 = None
        add_317: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_345, 1e-05);  getitem_345 = None
        rsqrt_75: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        sub_91: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_239, getitem_346);  clone_239 = getitem_346 = None
        mul_290: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_75);  sub_91 = rsqrt_75 = None
        mul_291: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_290, arg216_1);  mul_290 = arg216_1 = None
        add_318: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_291, arg217_1);  mul_291 = arg217_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:165 in forward, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        permute_298: "f32[384, 1152]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        view_455: "f32[1568, 384]" = torch.ops.aten.view.default(add_318, [1568, 384]);  add_318 = None
        mm_48: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_455, permute_298);  view_455 = permute_298 = None
        view_456: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_48, [8, 14, 14, 1152]);  mm_48 = None
        view_457: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_456, [8, 196, 3, 12, 32]);  view_456 = None
        permute_299: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_457, [2, 0, 3, 1, 4]);  view_457 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:166 in forward, code: q, k, v = qkv.unbind(0)
        unbind_29 = torch.ops.aten.unbind.int(permute_299);  permute_299 = None
        getitem_347: "f32[8, 12, 196, 32]" = unbind_29[0]
        getitem_348: "f32[8, 12, 196, 32]" = unbind_29[1]
        getitem_349: "f32[8, 12, 196, 32]" = unbind_29[2];  unbind_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:169 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_347, getitem_348, getitem_349, None, False);  getitem_347 = getitem_348 = getitem_349 = None
        getitem_350: "f32[8, 12, 196, 32]" = _scaled_dot_product_efficient_attention_27[0];  _scaled_dot_product_efficient_attention_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:180 in forward, code: x = x.transpose(1, 2).reshape(B, H, W, C)
        permute_300: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(getitem_350, [0, 2, 1, 3]);  getitem_350 = None
        view_458: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(permute_300, [8, 14, 14, 384]);  permute_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:181 in forward, code: x = self.proj(x)
        view_459: "f32[1568, 384]" = torch.ops.aten.view.default(view_458, [1568, 384]);  view_458 = None
        permute_301: "f32[384, 384]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_112: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg220_1, view_459, permute_301);  arg220_1 = view_459 = permute_301 = None
        view_460: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_112, [8, 14, 14, 384]);  addmm_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:210 in forward, code: x = x + self.drop_path1(self.attn(self.norm1(x)))
        add_319: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_316, view_460);  add_316 = view_460 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        clone_241: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_319, memory_format = torch.contiguous_format)
        var_mean_76 = torch.ops.aten.var_mean.correction(clone_241, [3], correction = 0, keepdim = True)
        getitem_354: "f32[8, 14, 14, 1]" = var_mean_76[0]
        getitem_355: "f32[8, 14, 14, 1]" = var_mean_76[1];  var_mean_76 = None
        add_320: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_354, 1e-05);  getitem_354 = None
        rsqrt_76: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        sub_92: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_241, getitem_355);  clone_241 = getitem_355 = None
        mul_292: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_76);  sub_92 = rsqrt_76 = None
        mul_293: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_292, arg221_1);  mul_292 = arg221_1 = None
        add_321: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_293, arg222_1);  mul_293 = arg222_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_461: "f32[1568, 384]" = torch.ops.aten.view.default(add_321, [1568, 384]);  add_321 = None
        permute_302: "f32[384, 1152]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_113: "f32[1568, 1152]" = torch.ops.aten.addmm.default(arg224_1, view_461, permute_302);  arg224_1 = view_461 = permute_302 = None
        view_462: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_113, [8, 14, 14, 1152]);  addmm_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_294: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_462, 0.5)
        mul_295: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_462, 0.7071067811865476);  view_462 = None
        erf_37: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_295);  mul_295 = None
        add_322: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_296: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_294, add_322);  mul_294 = add_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_463: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_296, [1568, 1152]);  mul_296 = None
        permute_303: "f32[1152, 384]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_114: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg226_1, view_463, permute_303);  arg226_1 = view_463 = permute_303 = None
        view_464: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_114, [8, 14, 14, 384]);  addmm_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:211 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_323: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_319, view_464);  add_319 = view_464 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:645 in forward_tokens, code: x = x.reshape(B, -1, C)
        view_465: "f32[8, 196, 384]" = torch.ops.aten.view.default(add_323, [8, 196, 384]);  add_323 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:650 in forward_cls, code: cls_tokens = self.cls_token.expand(B, -1, -1)
        expand_25: "f32[8, 1, 384]" = torch.ops.aten.expand.default(arg227_1, [8, -1, -1]);  arg227_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:651 in forward_cls, code: x = torch.cat([cls_tokens, x], dim=1)
        cat_3: "f32[8, 197, 384]" = torch.ops.aten.cat.default([expand_25, view_465], 1);  expand_25 = view_465 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:295 in forward, code: cls_embed = x[:, :1]
        slice_35: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(cat_3, 1, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:296 in forward, code: cls_embed = cls_embed + self.drop_path1(self.attn(self.norm1(x)))
        var_mean_77 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
        getitem_356: "f32[8, 197, 1]" = var_mean_77[0]
        getitem_357: "f32[8, 197, 1]" = var_mean_77[1];  var_mean_77 = None
        add_324: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_356, 1e-05);  getitem_356 = None
        rsqrt_77: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_93: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_3, getitem_357);  getitem_357 = None
        mul_297: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_77);  sub_93 = rsqrt_77 = None
        mul_298: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_297, arg228_1);  mul_297 = arg228_1 = None
        add_325: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_298, arg229_1);  mul_298 = arg229_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:244 in forward, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_304: "f32[384, 768]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        view_466: "f32[1576, 384]" = torch.ops.aten.view.default(add_325, [1576, 384])
        mm_49: "f32[1576, 768]" = torch.ops.aten.mm.default(view_466, permute_304);  view_466 = permute_304 = None
        view_467: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_49, [8, 197, 768]);  mm_49 = None
        view_468: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.view.default(view_467, [8, 197, 2, 12, 32]);  view_467 = None
        permute_305: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.permute.default(view_468, [2, 0, 3, 1, 4]);  view_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:245 in forward, code: k, v = kv.unbind(0)
        unbind_30 = torch.ops.aten.unbind.int(permute_305);  permute_305 = None
        getitem_358: "f32[8, 12, 197, 32]" = unbind_30[0]
        getitem_359: "f32[8, 12, 197, 32]" = unbind_30[1];  unbind_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:246 in forward, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim) * self.scale
        slice_37: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_325, 1, 0, 1);  add_325 = None
        permute_306: "f32[384, 384]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        view_469: "f32[8, 384]" = torch.ops.aten.view.default(slice_37, [8, 384]);  slice_37 = None
        mm_50: "f32[8, 384]" = torch.ops.aten.mm.default(view_469, permute_306);  view_469 = permute_306 = None
        view_470: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_50, [8, 1, 384]);  mm_50 = None
        view_471: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(view_470, [8, 12, 1, 32]);  view_470 = None
        mul_299: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_471, 0.1767766952966369);  view_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:248 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_307: "f32[8, 12, 32, 197]" = torch.ops.aten.permute.default(getitem_358, [0, 1, 3, 2]);  getitem_358 = None
        expand_26: "f32[8, 12, 1, 32]" = torch.ops.aten.expand.default(mul_299, [8, 12, 1, 32]);  mul_299 = None
        view_472: "f32[96, 1, 32]" = torch.ops.aten.view.default(expand_26, [96, 1, 32]);  expand_26 = None
        expand_27: "f32[8, 12, 32, 197]" = torch.ops.aten.expand.default(permute_307, [8, 12, 32, 197]);  permute_307 = None
        clone_244: "f32[8, 12, 32, 197]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_473: "f32[96, 32, 197]" = torch.ops.aten.view.default(clone_244, [96, 32, 197]);  clone_244 = None
        bmm_12: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_472, view_473);  view_472 = view_473 = None
        view_474: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_12, [8, 12, 1, 197]);  bmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:249 in forward, code: attn = attn.softmax(dim=-1)
        amax_10: "f32[8, 12, 1, 1]" = torch.ops.aten.amax.default(view_474, [-1], True)
        sub_94: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(view_474, amax_10);  view_474 = amax_10 = None
        exp_10: "f32[8, 12, 1, 197]" = torch.ops.aten.exp.default(sub_94);  sub_94 = None
        sum_11: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[8, 12, 1, 197]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:252 in forward, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        expand_28: "f32[8, 12, 1, 197]" = torch.ops.aten.expand.default(div_10, [8, 12, 1, 197]);  div_10 = None
        view_475: "f32[96, 1, 197]" = torch.ops.aten.view.default(expand_28, [96, 1, 197]);  expand_28 = None
        expand_29: "f32[8, 12, 197, 32]" = torch.ops.aten.expand.default(getitem_359, [8, 12, 197, 32]);  getitem_359 = None
        clone_246: "f32[8, 12, 197, 32]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_476: "f32[96, 197, 32]" = torch.ops.aten.view.default(clone_246, [96, 197, 32]);  clone_246 = None
        bmm_13: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_475, view_476);  view_475 = view_476 = None
        view_477: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_13, [8, 12, 1, 32]);  bmm_13 = None
        permute_308: "f32[8, 1, 12, 32]" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
        view_478: "f32[8, 1, 384]" = torch.ops.aten.view.default(permute_308, [8, 1, 384]);  permute_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:253 in forward, code: cls_embed = self.proj(cls_embed)
        view_479: "f32[8, 384]" = torch.ops.aten.view.default(view_478, [8, 384]);  view_478 = None
        permute_309: "f32[384, 384]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_115: "f32[8, 384]" = torch.ops.aten.addmm.default(arg233_1, view_479, permute_309);  arg233_1 = view_479 = permute_309 = None
        view_480: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_115, [8, 1, 384]);  addmm_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:296 in forward, code: cls_embed = cls_embed + self.drop_path1(self.attn(self.norm1(x)))
        add_326: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_35, view_480);  slice_35 = view_480 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:297 in forward, code: cls_embed = cls_embed + self.drop_path2(self.mlp(self.norm2(cls_embed)))
        var_mean_78 = torch.ops.aten.var_mean.correction(add_326, [2], correction = 0, keepdim = True)
        getitem_360: "f32[8, 1, 1]" = var_mean_78[0]
        getitem_361: "f32[8, 1, 1]" = var_mean_78[1];  var_mean_78 = None
        add_327: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_360, 1e-05);  getitem_360 = None
        rsqrt_78: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        sub_95: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_326, getitem_361);  getitem_361 = None
        mul_300: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_78);  sub_95 = rsqrt_78 = None
        mul_301: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_300, arg234_1);  mul_300 = arg234_1 = None
        add_328: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(mul_301, arg235_1);  mul_301 = arg235_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_481: "f32[8, 384]" = torch.ops.aten.view.default(add_328, [8, 384]);  add_328 = None
        permute_310: "f32[384, 1152]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_116: "f32[8, 1152]" = torch.ops.aten.addmm.default(arg237_1, view_481, permute_310);  arg237_1 = view_481 = permute_310 = None
        view_482: "f32[8, 1, 1152]" = torch.ops.aten.view.default(addmm_116, [8, 1, 1152]);  addmm_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_302: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_482, 0.5)
        mul_303: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_482, 0.7071067811865476);  view_482 = None
        erf_38: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
        add_329: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_304: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_302, add_329);  mul_302 = add_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_483: "f32[8, 1152]" = torch.ops.aten.view.default(mul_304, [8, 1152]);  mul_304 = None
        permute_311: "f32[1152, 384]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_117: "f32[8, 384]" = torch.ops.aten.addmm.default(arg239_1, view_483, permute_311);  arg239_1 = view_483 = permute_311 = None
        view_484: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_117, [8, 1, 384]);  addmm_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:297 in forward, code: cls_embed = cls_embed + self.drop_path2(self.mlp(self.norm2(cls_embed)))
        add_330: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(add_326, view_484);  add_326 = view_484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:298 in forward, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
        slice_40: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(cat_3, 1, 1, 9223372036854775807);  cat_3 = None
        cat_4: "f32[8, 197, 384]" = torch.ops.aten.cat.default([add_330, slice_40], 1);  add_330 = slice_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:295 in forward, code: cls_embed = x[:, :1]
        slice_42: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(cat_4, 1, 0, 1)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:296 in forward, code: cls_embed = cls_embed + self.drop_path1(self.attn(self.norm1(x)))
        var_mean_79 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
        getitem_362: "f32[8, 197, 1]" = var_mean_79[0]
        getitem_363: "f32[8, 197, 1]" = var_mean_79[1];  var_mean_79 = None
        add_331: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_362, 1e-05);  getitem_362 = None
        rsqrt_79: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_96: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_4, getitem_363);  getitem_363 = None
        mul_305: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_79);  sub_96 = rsqrt_79 = None
        mul_306: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_305, arg240_1);  mul_305 = arg240_1 = None
        add_332: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_306, arg241_1);  mul_306 = arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:244 in forward, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_312: "f32[384, 768]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        view_485: "f32[1576, 384]" = torch.ops.aten.view.default(add_332, [1576, 384])
        mm_51: "f32[1576, 768]" = torch.ops.aten.mm.default(view_485, permute_312);  view_485 = permute_312 = None
        view_486: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_51, [8, 197, 768]);  mm_51 = None
        view_487: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.view.default(view_486, [8, 197, 2, 12, 32]);  view_486 = None
        permute_313: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.permute.default(view_487, [2, 0, 3, 1, 4]);  view_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:245 in forward, code: k, v = kv.unbind(0)
        unbind_31 = torch.ops.aten.unbind.int(permute_313);  permute_313 = None
        getitem_364: "f32[8, 12, 197, 32]" = unbind_31[0]
        getitem_365: "f32[8, 12, 197, 32]" = unbind_31[1];  unbind_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:246 in forward, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim) * self.scale
        slice_44: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_332, 1, 0, 1);  add_332 = None
        permute_314: "f32[384, 384]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        view_488: "f32[8, 384]" = torch.ops.aten.view.default(slice_44, [8, 384]);  slice_44 = None
        mm_52: "f32[8, 384]" = torch.ops.aten.mm.default(view_488, permute_314);  view_488 = permute_314 = None
        view_489: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_52, [8, 1, 384]);  mm_52 = None
        view_490: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(view_489, [8, 12, 1, 32]);  view_489 = None
        mul_307: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_490, 0.1767766952966369);  view_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:248 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_315: "f32[8, 12, 32, 197]" = torch.ops.aten.permute.default(getitem_364, [0, 1, 3, 2]);  getitem_364 = None
        expand_30: "f32[8, 12, 1, 32]" = torch.ops.aten.expand.default(mul_307, [8, 12, 1, 32]);  mul_307 = None
        view_491: "f32[96, 1, 32]" = torch.ops.aten.view.default(expand_30, [96, 1, 32]);  expand_30 = None
        expand_31: "f32[8, 12, 32, 197]" = torch.ops.aten.expand.default(permute_315, [8, 12, 32, 197]);  permute_315 = None
        clone_250: "f32[8, 12, 32, 197]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_492: "f32[96, 32, 197]" = torch.ops.aten.view.default(clone_250, [96, 32, 197]);  clone_250 = None
        bmm_14: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_491, view_492);  view_491 = view_492 = None
        view_493: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_14, [8, 12, 1, 197]);  bmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:249 in forward, code: attn = attn.softmax(dim=-1)
        amax_11: "f32[8, 12, 1, 1]" = torch.ops.aten.amax.default(view_493, [-1], True)
        sub_97: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(view_493, amax_11);  view_493 = amax_11 = None
        exp_11: "f32[8, 12, 1, 197]" = torch.ops.aten.exp.default(sub_97);  sub_97 = None
        sum_12: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[8, 12, 1, 197]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:252 in forward, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        expand_32: "f32[8, 12, 1, 197]" = torch.ops.aten.expand.default(div_11, [8, 12, 1, 197]);  div_11 = None
        view_494: "f32[96, 1, 197]" = torch.ops.aten.view.default(expand_32, [96, 1, 197]);  expand_32 = None
        expand_33: "f32[8, 12, 197, 32]" = torch.ops.aten.expand.default(getitem_365, [8, 12, 197, 32]);  getitem_365 = None
        clone_252: "f32[8, 12, 197, 32]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_495: "f32[96, 197, 32]" = torch.ops.aten.view.default(clone_252, [96, 197, 32]);  clone_252 = None
        bmm_15: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_494, view_495);  view_494 = view_495 = None
        view_496: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_15, [8, 12, 1, 32]);  bmm_15 = None
        permute_316: "f32[8, 1, 12, 32]" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        view_497: "f32[8, 1, 384]" = torch.ops.aten.view.default(permute_316, [8, 1, 384]);  permute_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:253 in forward, code: cls_embed = self.proj(cls_embed)
        view_498: "f32[8, 384]" = torch.ops.aten.view.default(view_497, [8, 384]);  view_497 = None
        permute_317: "f32[384, 384]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_118: "f32[8, 384]" = torch.ops.aten.addmm.default(arg245_1, view_498, permute_317);  arg245_1 = view_498 = permute_317 = None
        view_499: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_118, [8, 1, 384]);  addmm_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:296 in forward, code: cls_embed = cls_embed + self.drop_path1(self.attn(self.norm1(x)))
        add_333: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_42, view_499);  slice_42 = view_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:297 in forward, code: cls_embed = cls_embed + self.drop_path2(self.mlp(self.norm2(cls_embed)))
        var_mean_80 = torch.ops.aten.var_mean.correction(add_333, [2], correction = 0, keepdim = True)
        getitem_366: "f32[8, 1, 1]" = var_mean_80[0]
        getitem_367: "f32[8, 1, 1]" = var_mean_80[1];  var_mean_80 = None
        add_334: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_366, 1e-05);  getitem_366 = None
        rsqrt_80: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_98: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_333, getitem_367);  getitem_367 = None
        mul_308: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_80);  sub_98 = rsqrt_80 = None
        mul_309: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_308, arg246_1);  mul_308 = arg246_1 = None
        add_335: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(mul_309, arg247_1);  mul_309 = arg247_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_500: "f32[8, 384]" = torch.ops.aten.view.default(add_335, [8, 384]);  add_335 = None
        permute_318: "f32[384, 1152]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        addmm_119: "f32[8, 1152]" = torch.ops.aten.addmm.default(arg249_1, view_500, permute_318);  arg249_1 = view_500 = permute_318 = None
        view_501: "f32[8, 1, 1152]" = torch.ops.aten.view.default(addmm_119, [8, 1, 1152]);  addmm_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_310: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        mul_311: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_501, 0.7071067811865476);  view_501 = None
        erf_39: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_311);  mul_311 = None
        add_336: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_312: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_310, add_336);  mul_310 = add_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_502: "f32[8, 1152]" = torch.ops.aten.view.default(mul_312, [8, 1152]);  mul_312 = None
        permute_319: "f32[1152, 384]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        addmm_120: "f32[8, 384]" = torch.ops.aten.addmm.default(arg251_1, view_502, permute_319);  arg251_1 = view_502 = permute_319 = None
        view_503: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_120, [8, 1, 384]);  addmm_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:297 in forward, code: cls_embed = cls_embed + self.drop_path2(self.mlp(self.norm2(cls_embed)))
        add_337: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(add_333, view_503);  add_333 = view_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:298 in forward, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
        slice_47: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(cat_4, 1, 1, 9223372036854775807);  cat_4 = None
        cat_5: "f32[8, 197, 384]" = torch.ops.aten.cat.default([add_337, slice_47], 1);  add_337 = slice_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:800 in forward_features, code: x = self.norm(x)
        var_mean_81 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
        getitem_368: "f32[8, 197, 1]" = var_mean_81[0]
        getitem_369: "f32[8, 197, 1]" = var_mean_81[1];  var_mean_81 = None
        add_338: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_368, 1e-05);  getitem_368 = None
        rsqrt_81: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        sub_99: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_5, getitem_369);  cat_5 = getitem_369 = None
        mul_313: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_81);  sub_99 = rsqrt_81 = None
        mul_314: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_313, arg252_1);  mul_313 = arg252_1 = None
        add_339: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_314, arg253_1);  mul_314 = arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:807 in forward_head, code: out = x[:, 0]
        select_1: "f32[8, 384]" = torch.ops.aten.select.int(add_339, 1, 0)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:813 in forward_head, code: out = self.head(out)
        permute_320: "f32[384, 1000]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        addmm_121: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg255_1, select_1, permute_320);  arg255_1 = select_1 = permute_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:816 in forward_head, code: aux = self.aux_head(x[:, 1:])
        slice_50: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_339, 1, 1, 9223372036854775807);  add_339 = None
        permute_321: "f32[384, 1000]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        clone_257: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_50, memory_format = torch.contiguous_format);  slice_50 = None
        view_504: "f32[1568, 384]" = torch.ops.aten.view.default(clone_257, [1568, 384]);  clone_257 = None
        mm_53: "f32[1568, 1000]" = torch.ops.aten.mm.default(view_504, permute_321);  view_504 = permute_321 = None
        view_505: "f32[8, 196, 1000]" = torch.ops.aten.view.default(mm_53, [8, 196, 1000]);  mm_53 = None
        add_340: "f32[8, 196, 1000]" = torch.ops.aten.add.Tensor(view_505, arg257_1);  view_505 = arg257_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/volo.py:817 in forward_head, code: out = out + 0.5 * aux.max(1)[0]
        max_2 = torch.ops.aten.max.dim(add_340, 1);  add_340 = None
        getitem_370: "f32[8, 1000]" = max_2[0];  max_2 = None
        mul_315: "f32[8, 1000]" = torch.ops.aten.mul.Tensor(getitem_370, 0.5);  getitem_370 = None
        add_341: "f32[8, 1000]" = torch.ops.aten.add.Tensor(addmm_121, mul_315);  addmm_121 = mul_315 = None
        return (add_341,)
        