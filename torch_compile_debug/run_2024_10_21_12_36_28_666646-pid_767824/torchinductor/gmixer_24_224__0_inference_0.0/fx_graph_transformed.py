class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[384, 3, 16, 16]", arg2_1: "f32[384]", arg3_1: "f32[384]", arg4_1: "f32[384]", arg5_1: "f32[384, 196]", arg6_1: "f32[384]", arg7_1: "f32[196, 192]", arg8_1: "f32[196]", arg9_1: "f32[384]", arg10_1: "f32[384]", arg11_1: "f32[1536, 384]", arg12_1: "f32[1536]", arg13_1: "f32[384, 768]", arg14_1: "f32[384]", arg15_1: "f32[384]", arg16_1: "f32[384]", arg17_1: "f32[384, 196]", arg18_1: "f32[384]", arg19_1: "f32[196, 192]", arg20_1: "f32[196]", arg21_1: "f32[384]", arg22_1: "f32[384]", arg23_1: "f32[1536, 384]", arg24_1: "f32[1536]", arg25_1: "f32[384, 768]", arg26_1: "f32[384]", arg27_1: "f32[384]", arg28_1: "f32[384]", arg29_1: "f32[384, 196]", arg30_1: "f32[384]", arg31_1: "f32[196, 192]", arg32_1: "f32[196]", arg33_1: "f32[384]", arg34_1: "f32[384]", arg35_1: "f32[1536, 384]", arg36_1: "f32[1536]", arg37_1: "f32[384, 768]", arg38_1: "f32[384]", arg39_1: "f32[384]", arg40_1: "f32[384]", arg41_1: "f32[384, 196]", arg42_1: "f32[384]", arg43_1: "f32[196, 192]", arg44_1: "f32[196]", arg45_1: "f32[384]", arg46_1: "f32[384]", arg47_1: "f32[1536, 384]", arg48_1: "f32[1536]", arg49_1: "f32[384, 768]", arg50_1: "f32[384]", arg51_1: "f32[384]", arg52_1: "f32[384]", arg53_1: "f32[384, 196]", arg54_1: "f32[384]", arg55_1: "f32[196, 192]", arg56_1: "f32[196]", arg57_1: "f32[384]", arg58_1: "f32[384]", arg59_1: "f32[1536, 384]", arg60_1: "f32[1536]", arg61_1: "f32[384, 768]", arg62_1: "f32[384]", arg63_1: "f32[384]", arg64_1: "f32[384]", arg65_1: "f32[384, 196]", arg66_1: "f32[384]", arg67_1: "f32[196, 192]", arg68_1: "f32[196]", arg69_1: "f32[384]", arg70_1: "f32[384]", arg71_1: "f32[1536, 384]", arg72_1: "f32[1536]", arg73_1: "f32[384, 768]", arg74_1: "f32[384]", arg75_1: "f32[384]", arg76_1: "f32[384]", arg77_1: "f32[384, 196]", arg78_1: "f32[384]", arg79_1: "f32[196, 192]", arg80_1: "f32[196]", arg81_1: "f32[384]", arg82_1: "f32[384]", arg83_1: "f32[1536, 384]", arg84_1: "f32[1536]", arg85_1: "f32[384, 768]", arg86_1: "f32[384]", arg87_1: "f32[384]", arg88_1: "f32[384]", arg89_1: "f32[384, 196]", arg90_1: "f32[384]", arg91_1: "f32[196, 192]", arg92_1: "f32[196]", arg93_1: "f32[384]", arg94_1: "f32[384]", arg95_1: "f32[1536, 384]", arg96_1: "f32[1536]", arg97_1: "f32[384, 768]", arg98_1: "f32[384]", arg99_1: "f32[384]", arg100_1: "f32[384]", arg101_1: "f32[384, 196]", arg102_1: "f32[384]", arg103_1: "f32[196, 192]", arg104_1: "f32[196]", arg105_1: "f32[384]", arg106_1: "f32[384]", arg107_1: "f32[1536, 384]", arg108_1: "f32[1536]", arg109_1: "f32[384, 768]", arg110_1: "f32[384]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[384, 196]", arg114_1: "f32[384]", arg115_1: "f32[196, 192]", arg116_1: "f32[196]", arg117_1: "f32[384]", arg118_1: "f32[384]", arg119_1: "f32[1536, 384]", arg120_1: "f32[1536]", arg121_1: "f32[384, 768]", arg122_1: "f32[384]", arg123_1: "f32[384]", arg124_1: "f32[384]", arg125_1: "f32[384, 196]", arg126_1: "f32[384]", arg127_1: "f32[196, 192]", arg128_1: "f32[196]", arg129_1: "f32[384]", arg130_1: "f32[384]", arg131_1: "f32[1536, 384]", arg132_1: "f32[1536]", arg133_1: "f32[384, 768]", arg134_1: "f32[384]", arg135_1: "f32[384]", arg136_1: "f32[384]", arg137_1: "f32[384, 196]", arg138_1: "f32[384]", arg139_1: "f32[196, 192]", arg140_1: "f32[196]", arg141_1: "f32[384]", arg142_1: "f32[384]", arg143_1: "f32[1536, 384]", arg144_1: "f32[1536]", arg145_1: "f32[384, 768]", arg146_1: "f32[384]", arg147_1: "f32[384]", arg148_1: "f32[384]", arg149_1: "f32[384, 196]", arg150_1: "f32[384]", arg151_1: "f32[196, 192]", arg152_1: "f32[196]", arg153_1: "f32[384]", arg154_1: "f32[384]", arg155_1: "f32[1536, 384]", arg156_1: "f32[1536]", arg157_1: "f32[384, 768]", arg158_1: "f32[384]", arg159_1: "f32[384]", arg160_1: "f32[384]", arg161_1: "f32[384, 196]", arg162_1: "f32[384]", arg163_1: "f32[196, 192]", arg164_1: "f32[196]", arg165_1: "f32[384]", arg166_1: "f32[384]", arg167_1: "f32[1536, 384]", arg168_1: "f32[1536]", arg169_1: "f32[384, 768]", arg170_1: "f32[384]", arg171_1: "f32[384]", arg172_1: "f32[384]", arg173_1: "f32[384, 196]", arg174_1: "f32[384]", arg175_1: "f32[196, 192]", arg176_1: "f32[196]", arg177_1: "f32[384]", arg178_1: "f32[384]", arg179_1: "f32[1536, 384]", arg180_1: "f32[1536]", arg181_1: "f32[384, 768]", arg182_1: "f32[384]", arg183_1: "f32[384]", arg184_1: "f32[384]", arg185_1: "f32[384, 196]", arg186_1: "f32[384]", arg187_1: "f32[196, 192]", arg188_1: "f32[196]", arg189_1: "f32[384]", arg190_1: "f32[384]", arg191_1: "f32[1536, 384]", arg192_1: "f32[1536]", arg193_1: "f32[384, 768]", arg194_1: "f32[384]", arg195_1: "f32[384]", arg196_1: "f32[384]", arg197_1: "f32[384, 196]", arg198_1: "f32[384]", arg199_1: "f32[196, 192]", arg200_1: "f32[196]", arg201_1: "f32[384]", arg202_1: "f32[384]", arg203_1: "f32[1536, 384]", arg204_1: "f32[1536]", arg205_1: "f32[384, 768]", arg206_1: "f32[384]", arg207_1: "f32[384]", arg208_1: "f32[384]", arg209_1: "f32[384, 196]", arg210_1: "f32[384]", arg211_1: "f32[196, 192]", arg212_1: "f32[196]", arg213_1: "f32[384]", arg214_1: "f32[384]", arg215_1: "f32[1536, 384]", arg216_1: "f32[1536]", arg217_1: "f32[384, 768]", arg218_1: "f32[384]", arg219_1: "f32[384]", arg220_1: "f32[384]", arg221_1: "f32[384, 196]", arg222_1: "f32[384]", arg223_1: "f32[196, 192]", arg224_1: "f32[196]", arg225_1: "f32[384]", arg226_1: "f32[384]", arg227_1: "f32[1536, 384]", arg228_1: "f32[1536]", arg229_1: "f32[384, 768]", arg230_1: "f32[384]", arg231_1: "f32[384]", arg232_1: "f32[384]", arg233_1: "f32[384, 196]", arg234_1: "f32[384]", arg235_1: "f32[196, 192]", arg236_1: "f32[196]", arg237_1: "f32[384]", arg238_1: "f32[384]", arg239_1: "f32[1536, 384]", arg240_1: "f32[1536]", arg241_1: "f32[384, 768]", arg242_1: "f32[384]", arg243_1: "f32[384]", arg244_1: "f32[384]", arg245_1: "f32[384, 196]", arg246_1: "f32[384]", arg247_1: "f32[196, 192]", arg248_1: "f32[196]", arg249_1: "f32[384]", arg250_1: "f32[384]", arg251_1: "f32[1536, 384]", arg252_1: "f32[1536]", arg253_1: "f32[384, 768]", arg254_1: "f32[384]", arg255_1: "f32[384]", arg256_1: "f32[384]", arg257_1: "f32[384, 196]", arg258_1: "f32[384]", arg259_1: "f32[196, 192]", arg260_1: "f32[196]", arg261_1: "f32[384]", arg262_1: "f32[384]", arg263_1: "f32[1536, 384]", arg264_1: "f32[1536]", arg265_1: "f32[384, 768]", arg266_1: "f32[384]", arg267_1: "f32[384]", arg268_1: "f32[384]", arg269_1: "f32[384, 196]", arg270_1: "f32[384]", arg271_1: "f32[196, 192]", arg272_1: "f32[196]", arg273_1: "f32[384]", arg274_1: "f32[384]", arg275_1: "f32[1536, 384]", arg276_1: "f32[1536]", arg277_1: "f32[384, 768]", arg278_1: "f32[384]", arg279_1: "f32[384]", arg280_1: "f32[384]", arg281_1: "f32[384, 196]", arg282_1: "f32[384]", arg283_1: "f32[196, 192]", arg284_1: "f32[196]", arg285_1: "f32[384]", arg286_1: "f32[384]", arg287_1: "f32[1536, 384]", arg288_1: "f32[1536]", arg289_1: "f32[384, 768]", arg290_1: "f32[384]", arg291_1: "f32[384]", arg292_1: "f32[384]", arg293_1: "f32[1000, 384]", arg294_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_1: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_193: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(convolution_1, [8, 384, 196]);  convolution_1 = None
        permute_146: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_170: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format)
        var_mean_49 = torch.ops.aten.var_mean.correction(clone_170, [2], correction = 0, keepdim = True)
        getitem_194: "f32[8, 196, 1]" = var_mean_49[0]
        getitem_195: "f32[8, 196, 1]" = var_mean_49[1];  var_mean_49 = None
        sub_49: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_170, getitem_195);  clone_170 = getitem_195 = None
        add_170: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-06);  getitem_194 = None
        rsqrt_49: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_194: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_195: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_194, arg3_1);  mul_194 = arg3_1 = None
        add_171: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_195, arg4_1);  mul_195 = arg4_1 = None
        permute_147: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_171, [0, 2, 1]);  add_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_171: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_194: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_171, [3072, 196]);  clone_171 = None
        permute_148: "f32[196, 384]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_24: "f32[3072, 384]" = torch.ops.aten.mm.default(view_194, permute_148);  view_194 = permute_148 = None
        view_195: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_24, [8, 384, 384]);  mm_24 = None
        add_172: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_195, arg6_1);  view_195 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_48 = torch.ops.aten.split.Tensor(add_172, 192, -1);  add_172 = None
        getitem_196: "f32[8, 384, 192]" = split_48[0]
        getitem_197: "f32[8, 384, 192]" = split_48[1];  split_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_48: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_197)
        mul_196: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_197, sigmoid_48);  getitem_197 = sigmoid_48 = None
        mul_197: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_196, mul_196);  getitem_196 = mul_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_196: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_197, [3072, 192]);  mul_197 = None
        permute_149: "f32[192, 196]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        
        # No stacktrace found for following nodes
        mm_default_71: "f32[3072, 196]" = torch.ops.aten.mm.default(view_196, permute_149);  view_196 = permute_149 = None
        add_tensor_71: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_71, arg8_1);  mm_default_71 = arg8_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_197: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_71, [8, 384, 196]);  add_tensor_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_150: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
        add_173: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(permute_146, permute_150);  permute_146 = permute_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_174: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_173, memory_format = torch.contiguous_format)
        var_mean_50 = torch.ops.aten.var_mean.correction(clone_174, [2], correction = 0, keepdim = True)
        getitem_198: "f32[8, 196, 1]" = var_mean_50[0]
        getitem_199: "f32[8, 196, 1]" = var_mean_50[1];  var_mean_50 = None
        sub_50: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_174, getitem_199);  clone_174 = getitem_199 = None
        add_174: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
        rsqrt_50: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_198: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        mul_199: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_198, arg9_1);  mul_198 = arg9_1 = None
        add_175: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_199, arg10_1);  mul_199 = arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_198: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_175, [1568, 384]);  add_175 = None
        permute_151: "f32[384, 1536]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        
        # No stacktrace found for following nodes
        mm_default_70: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_198, permute_151);  view_198 = permute_151 = None
        add_tensor_70: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_70, arg12_1);  mm_default_70 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_199: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_70, [8, 196, 1536]);  add_tensor_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_49 = torch.ops.aten.split.Tensor(view_199, 768, -1);  view_199 = None
        getitem_200: "f32[8, 196, 768]" = split_49[0]
        getitem_201: "f32[8, 196, 768]" = split_49[1];  split_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_49: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_201)
        mul_200: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_201, sigmoid_49);  getitem_201 = sigmoid_49 = None
        mul_201: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_200, mul_200);  getitem_200 = mul_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_200: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_201, [1568, 768]);  mul_201 = None
        permute_152: "f32[768, 384]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        
        # No stacktrace found for following nodes
        mm_default_69: "f32[1568, 384]" = torch.ops.aten.mm.default(view_200, permute_152);  view_200 = permute_152 = None
        add_tensor_69: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_69, arg14_1);  mm_default_69 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_201: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_69, [8, 196, 384]);  add_tensor_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_176: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_173, view_201);  add_173 = view_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_177: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_176, memory_format = torch.contiguous_format)
        var_mean_51 = torch.ops.aten.var_mean.correction(clone_177, [2], correction = 0, keepdim = True)
        getitem_202: "f32[8, 196, 1]" = var_mean_51[0]
        getitem_203: "f32[8, 196, 1]" = var_mean_51[1];  var_mean_51 = None
        sub_51: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_177, getitem_203);  clone_177 = getitem_203 = None
        add_177: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-06);  getitem_202 = None
        rsqrt_51: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        mul_202: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        mul_203: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_202, arg15_1);  mul_202 = arg15_1 = None
        add_178: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_203, arg16_1);  mul_203 = arg16_1 = None
        permute_153: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_178, [0, 2, 1]);  add_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_178: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_202: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_178, [3072, 196]);  clone_178 = None
        permute_154: "f32[196, 384]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_25: "f32[3072, 384]" = torch.ops.aten.mm.default(view_202, permute_154);  view_202 = permute_154 = None
        view_203: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_25, [8, 384, 384]);  mm_25 = None
        add_179: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_203, arg18_1);  view_203 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_50 = torch.ops.aten.split.Tensor(add_179, 192, -1);  add_179 = None
        getitem_204: "f32[8, 384, 192]" = split_50[0]
        getitem_205: "f32[8, 384, 192]" = split_50[1];  split_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_50: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_205)
        mul_204: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_205, sigmoid_50);  getitem_205 = sigmoid_50 = None
        mul_205: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_204, mul_204);  getitem_204 = mul_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_204: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_205, [3072, 192]);  mul_205 = None
        permute_155: "f32[192, 196]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        
        # No stacktrace found for following nodes
        mm_default_68: "f32[3072, 196]" = torch.ops.aten.mm.default(view_204, permute_155);  view_204 = permute_155 = None
        add_tensor_68: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_68, arg20_1);  mm_default_68 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_205: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_68, [8, 384, 196]);  add_tensor_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_156: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
        add_180: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_176, permute_156);  add_176 = permute_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_181: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_180, memory_format = torch.contiguous_format)
        var_mean_52 = torch.ops.aten.var_mean.correction(clone_181, [2], correction = 0, keepdim = True)
        getitem_206: "f32[8, 196, 1]" = var_mean_52[0]
        getitem_207: "f32[8, 196, 1]" = var_mean_52[1];  var_mean_52 = None
        sub_52: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_181, getitem_207);  clone_181 = getitem_207 = None
        add_181: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-06);  getitem_206 = None
        rsqrt_52: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_206: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        mul_207: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_206, arg21_1);  mul_206 = arg21_1 = None
        add_182: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_207, arg22_1);  mul_207 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_206: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_182, [1568, 384]);  add_182 = None
        permute_157: "f32[384, 1536]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        
        # No stacktrace found for following nodes
        mm_default_67: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_206, permute_157);  view_206 = permute_157 = None
        add_tensor_67: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_67, arg24_1);  mm_default_67 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_207: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_67, [8, 196, 1536]);  add_tensor_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_51 = torch.ops.aten.split.Tensor(view_207, 768, -1);  view_207 = None
        getitem_208: "f32[8, 196, 768]" = split_51[0]
        getitem_209: "f32[8, 196, 768]" = split_51[1];  split_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_51: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_209)
        mul_208: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_209, sigmoid_51);  getitem_209 = sigmoid_51 = None
        mul_209: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_208, mul_208);  getitem_208 = mul_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_208: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_209, [1568, 768]);  mul_209 = None
        permute_158: "f32[768, 384]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        
        # No stacktrace found for following nodes
        mm_default_66: "f32[1568, 384]" = torch.ops.aten.mm.default(view_208, permute_158);  view_208 = permute_158 = None
        add_tensor_66: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_66, arg26_1);  mm_default_66 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_209: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_66, [8, 196, 384]);  add_tensor_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_183: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_180, view_209);  add_180 = view_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_184: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_183, memory_format = torch.contiguous_format)
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_184, [2], correction = 0, keepdim = True)
        getitem_210: "f32[8, 196, 1]" = var_mean_53[0]
        getitem_211: "f32[8, 196, 1]" = var_mean_53[1];  var_mean_53 = None
        sub_53: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_184, getitem_211);  clone_184 = getitem_211 = None
        add_184: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-06);  getitem_210 = None
        rsqrt_53: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_210: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
        mul_211: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_210, arg27_1);  mul_210 = arg27_1 = None
        add_185: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_211, arg28_1);  mul_211 = arg28_1 = None
        permute_159: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_185, [0, 2, 1]);  add_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_185: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_210: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_185, [3072, 196]);  clone_185 = None
        permute_160: "f32[196, 384]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_26: "f32[3072, 384]" = torch.ops.aten.mm.default(view_210, permute_160);  view_210 = permute_160 = None
        view_211: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_26, [8, 384, 384]);  mm_26 = None
        add_186: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_211, arg30_1);  view_211 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_52 = torch.ops.aten.split.Tensor(add_186, 192, -1);  add_186 = None
        getitem_212: "f32[8, 384, 192]" = split_52[0]
        getitem_213: "f32[8, 384, 192]" = split_52[1];  split_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_52: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_213)
        mul_212: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_213, sigmoid_52);  getitem_213 = sigmoid_52 = None
        mul_213: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_212, mul_212);  getitem_212 = mul_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_212: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_213, [3072, 192]);  mul_213 = None
        permute_161: "f32[192, 196]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        
        # No stacktrace found for following nodes
        mm_default_65: "f32[3072, 196]" = torch.ops.aten.mm.default(view_212, permute_161);  view_212 = permute_161 = None
        add_tensor_65: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_65, arg32_1);  mm_default_65 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_213: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_65, [8, 384, 196]);  add_tensor_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_162: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        add_187: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_183, permute_162);  add_183 = permute_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_188: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_187, memory_format = torch.contiguous_format)
        var_mean_54 = torch.ops.aten.var_mean.correction(clone_188, [2], correction = 0, keepdim = True)
        getitem_214: "f32[8, 196, 1]" = var_mean_54[0]
        getitem_215: "f32[8, 196, 1]" = var_mean_54[1];  var_mean_54 = None
        sub_54: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_188, getitem_215);  clone_188 = getitem_215 = None
        add_188: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
        rsqrt_54: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_214: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
        mul_215: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_214, arg33_1);  mul_214 = arg33_1 = None
        add_189: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_215, arg34_1);  mul_215 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_214: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_189, [1568, 384]);  add_189 = None
        permute_163: "f32[384, 1536]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        
        # No stacktrace found for following nodes
        mm_default_64: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_214, permute_163);  view_214 = permute_163 = None
        add_tensor_64: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_64, arg36_1);  mm_default_64 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_215: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_64, [8, 196, 1536]);  add_tensor_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_53 = torch.ops.aten.split.Tensor(view_215, 768, -1);  view_215 = None
        getitem_216: "f32[8, 196, 768]" = split_53[0]
        getitem_217: "f32[8, 196, 768]" = split_53[1];  split_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_53: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_217)
        mul_216: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_217, sigmoid_53);  getitem_217 = sigmoid_53 = None
        mul_217: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_216, mul_216);  getitem_216 = mul_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_216: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_217, [1568, 768]);  mul_217 = None
        permute_164: "f32[768, 384]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        
        # No stacktrace found for following nodes
        mm_default_63: "f32[1568, 384]" = torch.ops.aten.mm.default(view_216, permute_164);  view_216 = permute_164 = None
        add_tensor_63: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_63, arg38_1);  mm_default_63 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_217: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_63, [8, 196, 384]);  add_tensor_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_190: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_187, view_217);  add_187 = view_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_191: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_190, memory_format = torch.contiguous_format)
        var_mean_55 = torch.ops.aten.var_mean.correction(clone_191, [2], correction = 0, keepdim = True)
        getitem_218: "f32[8, 196, 1]" = var_mean_55[0]
        getitem_219: "f32[8, 196, 1]" = var_mean_55[1];  var_mean_55 = None
        sub_55: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_191, getitem_219);  clone_191 = getitem_219 = None
        add_191: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
        rsqrt_55: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        mul_218: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
        mul_219: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_218, arg39_1);  mul_218 = arg39_1 = None
        add_192: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_219, arg40_1);  mul_219 = arg40_1 = None
        permute_165: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_192, [0, 2, 1]);  add_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_192: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_218: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_192, [3072, 196]);  clone_192 = None
        permute_166: "f32[196, 384]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_27: "f32[3072, 384]" = torch.ops.aten.mm.default(view_218, permute_166);  view_218 = permute_166 = None
        view_219: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_27, [8, 384, 384]);  mm_27 = None
        add_193: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_219, arg42_1);  view_219 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_54 = torch.ops.aten.split.Tensor(add_193, 192, -1);  add_193 = None
        getitem_220: "f32[8, 384, 192]" = split_54[0]
        getitem_221: "f32[8, 384, 192]" = split_54[1];  split_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_54: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_221)
        mul_220: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_221, sigmoid_54);  getitem_221 = sigmoid_54 = None
        mul_221: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_220, mul_220);  getitem_220 = mul_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_220: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_221, [3072, 192]);  mul_221 = None
        permute_167: "f32[192, 196]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        
        # No stacktrace found for following nodes
        mm_default_62: "f32[3072, 196]" = torch.ops.aten.mm.default(view_220, permute_167);  view_220 = permute_167 = None
        add_tensor_62: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_62, arg44_1);  mm_default_62 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_221: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_62, [8, 384, 196]);  add_tensor_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_168: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
        add_194: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_190, permute_168);  add_190 = permute_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_195: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_194, memory_format = torch.contiguous_format)
        var_mean_56 = torch.ops.aten.var_mean.correction(clone_195, [2], correction = 0, keepdim = True)
        getitem_222: "f32[8, 196, 1]" = var_mean_56[0]
        getitem_223: "f32[8, 196, 1]" = var_mean_56[1];  var_mean_56 = None
        sub_56: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_195, getitem_223);  clone_195 = getitem_223 = None
        add_195: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_56: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_222: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
        mul_223: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_222, arg45_1);  mul_222 = arg45_1 = None
        add_196: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_223, arg46_1);  mul_223 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_222: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_196, [1568, 384]);  add_196 = None
        permute_169: "f32[384, 1536]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        
        # No stacktrace found for following nodes
        mm_default_61: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_222, permute_169);  view_222 = permute_169 = None
        add_tensor_61: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_61, arg48_1);  mm_default_61 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_223: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_61, [8, 196, 1536]);  add_tensor_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_55 = torch.ops.aten.split.Tensor(view_223, 768, -1);  view_223 = None
        getitem_224: "f32[8, 196, 768]" = split_55[0]
        getitem_225: "f32[8, 196, 768]" = split_55[1];  split_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_55: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_225)
        mul_224: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_225, sigmoid_55);  getitem_225 = sigmoid_55 = None
        mul_225: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_224, mul_224);  getitem_224 = mul_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_224: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_225, [1568, 768]);  mul_225 = None
        permute_170: "f32[768, 384]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        
        # No stacktrace found for following nodes
        mm_default_60: "f32[1568, 384]" = torch.ops.aten.mm.default(view_224, permute_170);  view_224 = permute_170 = None
        add_tensor_60: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_60, arg50_1);  mm_default_60 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_225: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_60, [8, 196, 384]);  add_tensor_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_197: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_194, view_225);  add_194 = view_225 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_198: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_197, memory_format = torch.contiguous_format)
        var_mean_57 = torch.ops.aten.var_mean.correction(clone_198, [2], correction = 0, keepdim = True)
        getitem_226: "f32[8, 196, 1]" = var_mean_57[0]
        getitem_227: "f32[8, 196, 1]" = var_mean_57[1];  var_mean_57 = None
        sub_57: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_198, getitem_227);  clone_198 = getitem_227 = None
        add_198: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
        rsqrt_57: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        mul_226: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        mul_227: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_226, arg51_1);  mul_226 = arg51_1 = None
        add_199: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_227, arg52_1);  mul_227 = arg52_1 = None
        permute_171: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_199, [0, 2, 1]);  add_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_199: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_226: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_199, [3072, 196]);  clone_199 = None
        permute_172: "f32[196, 384]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_28: "f32[3072, 384]" = torch.ops.aten.mm.default(view_226, permute_172);  view_226 = permute_172 = None
        view_227: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_28, [8, 384, 384]);  mm_28 = None
        add_200: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_227, arg54_1);  view_227 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_56 = torch.ops.aten.split.Tensor(add_200, 192, -1);  add_200 = None
        getitem_228: "f32[8, 384, 192]" = split_56[0]
        getitem_229: "f32[8, 384, 192]" = split_56[1];  split_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_56: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_229)
        mul_228: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_229, sigmoid_56);  getitem_229 = sigmoid_56 = None
        mul_229: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_228, mul_228);  getitem_228 = mul_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_228: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_229, [3072, 192]);  mul_229 = None
        permute_173: "f32[192, 196]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        
        # No stacktrace found for following nodes
        mm_default_59: "f32[3072, 196]" = torch.ops.aten.mm.default(view_228, permute_173);  view_228 = permute_173 = None
        add_tensor_59: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_59, arg56_1);  mm_default_59 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_229: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 384, 196]);  add_tensor_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_174: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        add_201: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_197, permute_174);  add_197 = permute_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_202: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_201, memory_format = torch.contiguous_format)
        var_mean_58 = torch.ops.aten.var_mean.correction(clone_202, [2], correction = 0, keepdim = True)
        getitem_230: "f32[8, 196, 1]" = var_mean_58[0]
        getitem_231: "f32[8, 196, 1]" = var_mean_58[1];  var_mean_58 = None
        sub_58: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_202, getitem_231);  clone_202 = getitem_231 = None
        add_202: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_230, 1e-06);  getitem_230 = None
        rsqrt_58: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        mul_230: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        mul_231: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_230, arg57_1);  mul_230 = arg57_1 = None
        add_203: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_231, arg58_1);  mul_231 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_230: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_203, [1568, 384]);  add_203 = None
        permute_175: "f32[384, 1536]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        
        # No stacktrace found for following nodes
        mm_default_58: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_230, permute_175);  view_230 = permute_175 = None
        add_tensor_58: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_58, arg60_1);  mm_default_58 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_231: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 196, 1536]);  add_tensor_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_57 = torch.ops.aten.split.Tensor(view_231, 768, -1);  view_231 = None
        getitem_232: "f32[8, 196, 768]" = split_57[0]
        getitem_233: "f32[8, 196, 768]" = split_57[1];  split_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_57: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_233)
        mul_232: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_233, sigmoid_57);  getitem_233 = sigmoid_57 = None
        mul_233: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_232, mul_232);  getitem_232 = mul_232 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_232: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_233, [1568, 768]);  mul_233 = None
        permute_176: "f32[768, 384]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        
        # No stacktrace found for following nodes
        mm_default_57: "f32[1568, 384]" = torch.ops.aten.mm.default(view_232, permute_176);  view_232 = permute_176 = None
        add_tensor_57: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_57, arg62_1);  mm_default_57 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_233: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_57, [8, 196, 384]);  add_tensor_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_204: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_201, view_233);  add_201 = view_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_205: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_204, memory_format = torch.contiguous_format)
        var_mean_59 = torch.ops.aten.var_mean.correction(clone_205, [2], correction = 0, keepdim = True)
        getitem_234: "f32[8, 196, 1]" = var_mean_59[0]
        getitem_235: "f32[8, 196, 1]" = var_mean_59[1];  var_mean_59 = None
        sub_59: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_205, getitem_235);  clone_205 = getitem_235 = None
        add_205: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-06);  getitem_234 = None
        rsqrt_59: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        mul_234: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        mul_235: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_234, arg63_1);  mul_234 = arg63_1 = None
        add_206: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_235, arg64_1);  mul_235 = arg64_1 = None
        permute_177: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_206, [0, 2, 1]);  add_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_206: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_234: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_206, [3072, 196]);  clone_206 = None
        permute_178: "f32[196, 384]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_29: "f32[3072, 384]" = torch.ops.aten.mm.default(view_234, permute_178);  view_234 = permute_178 = None
        view_235: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_29, [8, 384, 384]);  mm_29 = None
        add_207: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_235, arg66_1);  view_235 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_58 = torch.ops.aten.split.Tensor(add_207, 192, -1);  add_207 = None
        getitem_236: "f32[8, 384, 192]" = split_58[0]
        getitem_237: "f32[8, 384, 192]" = split_58[1];  split_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_58: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_237)
        mul_236: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_237, sigmoid_58);  getitem_237 = sigmoid_58 = None
        mul_237: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_236, mul_236);  getitem_236 = mul_236 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_236: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_237, [3072, 192]);  mul_237 = None
        permute_179: "f32[192, 196]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        
        # No stacktrace found for following nodes
        mm_default_56: "f32[3072, 196]" = torch.ops.aten.mm.default(view_236, permute_179);  view_236 = permute_179 = None
        add_tensor_56: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_56, arg68_1);  mm_default_56 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_237: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_56, [8, 384, 196]);  add_tensor_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_180: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
        add_208: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_204, permute_180);  add_204 = permute_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_209: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_208, memory_format = torch.contiguous_format)
        var_mean_60 = torch.ops.aten.var_mean.correction(clone_209, [2], correction = 0, keepdim = True)
        getitem_238: "f32[8, 196, 1]" = var_mean_60[0]
        getitem_239: "f32[8, 196, 1]" = var_mean_60[1];  var_mean_60 = None
        sub_60: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_209, getitem_239);  clone_209 = getitem_239 = None
        add_209: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-06);  getitem_238 = None
        rsqrt_60: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        mul_238: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        mul_239: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_238, arg69_1);  mul_238 = arg69_1 = None
        add_210: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_239, arg70_1);  mul_239 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_238: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_210, [1568, 384]);  add_210 = None
        permute_181: "f32[384, 1536]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        
        # No stacktrace found for following nodes
        mm_default_55: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_238, permute_181);  view_238 = permute_181 = None
        add_tensor_55: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_55, arg72_1);  mm_default_55 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_239: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_55, [8, 196, 1536]);  add_tensor_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_59 = torch.ops.aten.split.Tensor(view_239, 768, -1);  view_239 = None
        getitem_240: "f32[8, 196, 768]" = split_59[0]
        getitem_241: "f32[8, 196, 768]" = split_59[1];  split_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_59: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_241)
        mul_240: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_241, sigmoid_59);  getitem_241 = sigmoid_59 = None
        mul_241: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_240, mul_240);  getitem_240 = mul_240 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_240: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_241, [1568, 768]);  mul_241 = None
        permute_182: "f32[768, 384]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        
        # No stacktrace found for following nodes
        mm_default_54: "f32[1568, 384]" = torch.ops.aten.mm.default(view_240, permute_182);  view_240 = permute_182 = None
        add_tensor_54: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_54, arg74_1);  mm_default_54 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_241: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 196, 384]);  add_tensor_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_211: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_208, view_241);  add_208 = view_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_212: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_211, memory_format = torch.contiguous_format)
        var_mean_61 = torch.ops.aten.var_mean.correction(clone_212, [2], correction = 0, keepdim = True)
        getitem_242: "f32[8, 196, 1]" = var_mean_61[0]
        getitem_243: "f32[8, 196, 1]" = var_mean_61[1];  var_mean_61 = None
        sub_61: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_212, getitem_243);  clone_212 = getitem_243 = None
        add_212: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_61: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        mul_242: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        mul_243: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_242, arg75_1);  mul_242 = arg75_1 = None
        add_213: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_243, arg76_1);  mul_243 = arg76_1 = None
        permute_183: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_213, [0, 2, 1]);  add_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_213: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_242: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_213, [3072, 196]);  clone_213 = None
        permute_184: "f32[196, 384]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_30: "f32[3072, 384]" = torch.ops.aten.mm.default(view_242, permute_184);  view_242 = permute_184 = None
        view_243: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_30, [8, 384, 384]);  mm_30 = None
        add_214: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_243, arg78_1);  view_243 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_60 = torch.ops.aten.split.Tensor(add_214, 192, -1);  add_214 = None
        getitem_244: "f32[8, 384, 192]" = split_60[0]
        getitem_245: "f32[8, 384, 192]" = split_60[1];  split_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_60: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_245)
        mul_244: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_245, sigmoid_60);  getitem_245 = sigmoid_60 = None
        mul_245: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_244, mul_244);  getitem_244 = mul_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_244: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_245, [3072, 192]);  mul_245 = None
        permute_185: "f32[192, 196]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        
        # No stacktrace found for following nodes
        mm_default_53: "f32[3072, 196]" = torch.ops.aten.mm.default(view_244, permute_185);  view_244 = permute_185 = None
        add_tensor_53: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_53, arg80_1);  mm_default_53 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_245: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 384, 196]);  add_tensor_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_186: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
        add_215: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_211, permute_186);  add_211 = permute_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_216: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_215, memory_format = torch.contiguous_format)
        var_mean_62 = torch.ops.aten.var_mean.correction(clone_216, [2], correction = 0, keepdim = True)
        getitem_246: "f32[8, 196, 1]" = var_mean_62[0]
        getitem_247: "f32[8, 196, 1]" = var_mean_62[1];  var_mean_62 = None
        sub_62: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_216, getitem_247);  clone_216 = getitem_247 = None
        add_216: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-06);  getitem_246 = None
        rsqrt_62: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        mul_246: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        mul_247: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_246, arg81_1);  mul_246 = arg81_1 = None
        add_217: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_247, arg82_1);  mul_247 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_246: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_217, [1568, 384]);  add_217 = None
        permute_187: "f32[384, 1536]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        
        # No stacktrace found for following nodes
        mm_default_52: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_246, permute_187);  view_246 = permute_187 = None
        add_tensor_52: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_52, arg84_1);  mm_default_52 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_247: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 196, 1536]);  add_tensor_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_61 = torch.ops.aten.split.Tensor(view_247, 768, -1);  view_247 = None
        getitem_248: "f32[8, 196, 768]" = split_61[0]
        getitem_249: "f32[8, 196, 768]" = split_61[1];  split_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_61: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_249)
        mul_248: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_249, sigmoid_61);  getitem_249 = sigmoid_61 = None
        mul_249: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_248, mul_248);  getitem_248 = mul_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_248: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_249, [1568, 768]);  mul_249 = None
        permute_188: "f32[768, 384]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        
        # No stacktrace found for following nodes
        mm_default_51: "f32[1568, 384]" = torch.ops.aten.mm.default(view_248, permute_188);  view_248 = permute_188 = None
        add_tensor_51: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_51, arg86_1);  mm_default_51 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_249: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 196, 384]);  add_tensor_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_218: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_215, view_249);  add_215 = view_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_219: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_218, memory_format = torch.contiguous_format)
        var_mean_63 = torch.ops.aten.var_mean.correction(clone_219, [2], correction = 0, keepdim = True)
        getitem_250: "f32[8, 196, 1]" = var_mean_63[0]
        getitem_251: "f32[8, 196, 1]" = var_mean_63[1];  var_mean_63 = None
        sub_63: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_219, getitem_251);  clone_219 = getitem_251 = None
        add_219: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-06);  getitem_250 = None
        rsqrt_63: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_250: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        mul_251: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_250, arg87_1);  mul_250 = arg87_1 = None
        add_220: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_251, arg88_1);  mul_251 = arg88_1 = None
        permute_189: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_220, [0, 2, 1]);  add_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_220: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_250: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_220, [3072, 196]);  clone_220 = None
        permute_190: "f32[196, 384]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_31: "f32[3072, 384]" = torch.ops.aten.mm.default(view_250, permute_190);  view_250 = permute_190 = None
        view_251: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_31, [8, 384, 384]);  mm_31 = None
        add_221: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_251, arg90_1);  view_251 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_62 = torch.ops.aten.split.Tensor(add_221, 192, -1);  add_221 = None
        getitem_252: "f32[8, 384, 192]" = split_62[0]
        getitem_253: "f32[8, 384, 192]" = split_62[1];  split_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_62: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_253)
        mul_252: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_253, sigmoid_62);  getitem_253 = sigmoid_62 = None
        mul_253: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_252, mul_252);  getitem_252 = mul_252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_252: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_253, [3072, 192]);  mul_253 = None
        permute_191: "f32[192, 196]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        
        # No stacktrace found for following nodes
        mm_default_50: "f32[3072, 196]" = torch.ops.aten.mm.default(view_252, permute_191);  view_252 = permute_191 = None
        add_tensor_50: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_50, arg92_1);  mm_default_50 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_253: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_50, [8, 384, 196]);  add_tensor_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_192: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        add_222: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_218, permute_192);  add_218 = permute_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_223: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_222, memory_format = torch.contiguous_format)
        var_mean_64 = torch.ops.aten.var_mean.correction(clone_223, [2], correction = 0, keepdim = True)
        getitem_254: "f32[8, 196, 1]" = var_mean_64[0]
        getitem_255: "f32[8, 196, 1]" = var_mean_64[1];  var_mean_64 = None
        sub_64: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_223, getitem_255);  clone_223 = getitem_255 = None
        add_223: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_64: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        mul_254: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        mul_255: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_254, arg93_1);  mul_254 = arg93_1 = None
        add_224: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_255, arg94_1);  mul_255 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_254: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_224, [1568, 384]);  add_224 = None
        permute_193: "f32[384, 1536]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        
        # No stacktrace found for following nodes
        mm_default_49: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_254, permute_193);  view_254 = permute_193 = None
        add_tensor_49: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_49, arg96_1);  mm_default_49 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_255: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_49, [8, 196, 1536]);  add_tensor_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_63 = torch.ops.aten.split.Tensor(view_255, 768, -1);  view_255 = None
        getitem_256: "f32[8, 196, 768]" = split_63[0]
        getitem_257: "f32[8, 196, 768]" = split_63[1];  split_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_63: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_257)
        mul_256: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_257, sigmoid_63);  getitem_257 = sigmoid_63 = None
        mul_257: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_256, mul_256);  getitem_256 = mul_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_256: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_257, [1568, 768]);  mul_257 = None
        permute_194: "f32[768, 384]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        
        # No stacktrace found for following nodes
        mm_default_48: "f32[1568, 384]" = torch.ops.aten.mm.default(view_256, permute_194);  view_256 = permute_194 = None
        add_tensor_48: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_48, arg98_1);  mm_default_48 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_257: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_48, [8, 196, 384]);  add_tensor_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_225: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_222, view_257);  add_222 = view_257 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_226: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_225, memory_format = torch.contiguous_format)
        var_mean_65 = torch.ops.aten.var_mean.correction(clone_226, [2], correction = 0, keepdim = True)
        getitem_258: "f32[8, 196, 1]" = var_mean_65[0]
        getitem_259: "f32[8, 196, 1]" = var_mean_65[1];  var_mean_65 = None
        sub_65: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_226, getitem_259);  clone_226 = getitem_259 = None
        add_226: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-06);  getitem_258 = None
        rsqrt_65: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        mul_258: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        mul_259: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_258, arg99_1);  mul_258 = arg99_1 = None
        add_227: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_259, arg100_1);  mul_259 = arg100_1 = None
        permute_195: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_227, [0, 2, 1]);  add_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_227: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_258: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_227, [3072, 196]);  clone_227 = None
        permute_196: "f32[196, 384]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_32: "f32[3072, 384]" = torch.ops.aten.mm.default(view_258, permute_196);  view_258 = permute_196 = None
        view_259: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_32, [8, 384, 384]);  mm_32 = None
        add_228: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_259, arg102_1);  view_259 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_64 = torch.ops.aten.split.Tensor(add_228, 192, -1);  add_228 = None
        getitem_260: "f32[8, 384, 192]" = split_64[0]
        getitem_261: "f32[8, 384, 192]" = split_64[1];  split_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_64: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_261)
        mul_260: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_261, sigmoid_64);  getitem_261 = sigmoid_64 = None
        mul_261: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_260, mul_260);  getitem_260 = mul_260 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_260: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_261, [3072, 192]);  mul_261 = None
        permute_197: "f32[192, 196]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        
        # No stacktrace found for following nodes
        mm_default_47: "f32[3072, 196]" = torch.ops.aten.mm.default(view_260, permute_197);  view_260 = permute_197 = None
        add_tensor_47: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_47, arg104_1);  mm_default_47 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_261: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 384, 196]);  add_tensor_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_198: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None
        add_229: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_225, permute_198);  add_225 = permute_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_230: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_229, memory_format = torch.contiguous_format)
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_230, [2], correction = 0, keepdim = True)
        getitem_262: "f32[8, 196, 1]" = var_mean_66[0]
        getitem_263: "f32[8, 196, 1]" = var_mean_66[1];  var_mean_66 = None
        sub_66: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_230, getitem_263);  clone_230 = getitem_263 = None
        add_230: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_262, 1e-06);  getitem_262 = None
        rsqrt_66: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
        mul_262: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        mul_263: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_262, arg105_1);  mul_262 = arg105_1 = None
        add_231: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_263, arg106_1);  mul_263 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_262: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_231, [1568, 384]);  add_231 = None
        permute_199: "f32[384, 1536]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        
        # No stacktrace found for following nodes
        mm_default_46: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_262, permute_199);  view_262 = permute_199 = None
        add_tensor_46: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_46, arg108_1);  mm_default_46 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_263: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 196, 1536]);  add_tensor_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_65 = torch.ops.aten.split.Tensor(view_263, 768, -1);  view_263 = None
        getitem_264: "f32[8, 196, 768]" = split_65[0]
        getitem_265: "f32[8, 196, 768]" = split_65[1];  split_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_65: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_265)
        mul_264: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_265, sigmoid_65);  getitem_265 = sigmoid_65 = None
        mul_265: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_264, mul_264);  getitem_264 = mul_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_264: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_265, [1568, 768]);  mul_265 = None
        permute_200: "f32[768, 384]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        
        # No stacktrace found for following nodes
        mm_default_45: "f32[1568, 384]" = torch.ops.aten.mm.default(view_264, permute_200);  view_264 = permute_200 = None
        add_tensor_45: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_45, arg110_1);  mm_default_45 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_265: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 196, 384]);  add_tensor_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_232: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_229, view_265);  add_229 = view_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_233: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_232, memory_format = torch.contiguous_format)
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_233, [2], correction = 0, keepdim = True)
        getitem_266: "f32[8, 196, 1]" = var_mean_67[0]
        getitem_267: "f32[8, 196, 1]" = var_mean_67[1];  var_mean_67 = None
        sub_67: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_233, getitem_267);  clone_233 = getitem_267 = None
        add_233: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_67: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        mul_266: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        mul_267: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_266, arg111_1);  mul_266 = arg111_1 = None
        add_234: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_267, arg112_1);  mul_267 = arg112_1 = None
        permute_201: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_234, [0, 2, 1]);  add_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_234: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_266: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_234, [3072, 196]);  clone_234 = None
        permute_202: "f32[196, 384]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_33: "f32[3072, 384]" = torch.ops.aten.mm.default(view_266, permute_202);  view_266 = permute_202 = None
        view_267: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_33, [8, 384, 384]);  mm_33 = None
        add_235: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_267, arg114_1);  view_267 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_66 = torch.ops.aten.split.Tensor(add_235, 192, -1);  add_235 = None
        getitem_268: "f32[8, 384, 192]" = split_66[0]
        getitem_269: "f32[8, 384, 192]" = split_66[1];  split_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_66: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_269)
        mul_268: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_269, sigmoid_66);  getitem_269 = sigmoid_66 = None
        mul_269: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_268, mul_268);  getitem_268 = mul_268 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_268: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_269, [3072, 192]);  mul_269 = None
        permute_203: "f32[192, 196]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        
        # No stacktrace found for following nodes
        mm_default_44: "f32[3072, 196]" = torch.ops.aten.mm.default(view_268, permute_203);  view_268 = permute_203 = None
        add_tensor_44: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_44, arg116_1);  mm_default_44 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_269: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 384, 196]);  add_tensor_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_204: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
        add_236: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_232, permute_204);  add_232 = permute_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_237: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_236, memory_format = torch.contiguous_format)
        var_mean_68 = torch.ops.aten.var_mean.correction(clone_237, [2], correction = 0, keepdim = True)
        getitem_270: "f32[8, 196, 1]" = var_mean_68[0]
        getitem_271: "f32[8, 196, 1]" = var_mean_68[1];  var_mean_68 = None
        sub_68: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_237, getitem_271);  clone_237 = getitem_271 = None
        add_237: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_68: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        mul_270: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        mul_271: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_270, arg117_1);  mul_270 = arg117_1 = None
        add_238: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_271, arg118_1);  mul_271 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_270: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_238, [1568, 384]);  add_238 = None
        permute_205: "f32[384, 1536]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        
        # No stacktrace found for following nodes
        mm_default_43: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_270, permute_205);  view_270 = permute_205 = None
        add_tensor_43: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_43, arg120_1);  mm_default_43 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_271: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_43, [8, 196, 1536]);  add_tensor_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_67 = torch.ops.aten.split.Tensor(view_271, 768, -1);  view_271 = None
        getitem_272: "f32[8, 196, 768]" = split_67[0]
        getitem_273: "f32[8, 196, 768]" = split_67[1];  split_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_67: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_273)
        mul_272: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_273, sigmoid_67);  getitem_273 = sigmoid_67 = None
        mul_273: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_272, mul_272);  getitem_272 = mul_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_272: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_273, [1568, 768]);  mul_273 = None
        permute_206: "f32[768, 384]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        
        # No stacktrace found for following nodes
        mm_default_42: "f32[1568, 384]" = torch.ops.aten.mm.default(view_272, permute_206);  view_272 = permute_206 = None
        add_tensor_42: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_42, arg122_1);  mm_default_42 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_273: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_42, [8, 196, 384]);  add_tensor_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_239: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_236, view_273);  add_236 = view_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_240: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
        var_mean_69 = torch.ops.aten.var_mean.correction(clone_240, [2], correction = 0, keepdim = True)
        getitem_274: "f32[8, 196, 1]" = var_mean_69[0]
        getitem_275: "f32[8, 196, 1]" = var_mean_69[1];  var_mean_69 = None
        sub_69: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_240, getitem_275);  clone_240 = getitem_275 = None
        add_240: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_274, 1e-06);  getitem_274 = None
        rsqrt_69: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        mul_274: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        mul_275: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_274, arg123_1);  mul_274 = arg123_1 = None
        add_241: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_275, arg124_1);  mul_275 = arg124_1 = None
        permute_207: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_241, [0, 2, 1]);  add_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_241: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
        view_274: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_241, [3072, 196]);  clone_241 = None
        permute_208: "f32[196, 384]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        mm_34: "f32[3072, 384]" = torch.ops.aten.mm.default(view_274, permute_208);  view_274 = permute_208 = None
        view_275: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_34, [8, 384, 384]);  mm_34 = None
        add_242: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_275, arg126_1);  view_275 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_68 = torch.ops.aten.split.Tensor(add_242, 192, -1);  add_242 = None
        getitem_276: "f32[8, 384, 192]" = split_68[0]
        getitem_277: "f32[8, 384, 192]" = split_68[1];  split_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_68: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_277)
        mul_276: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_277, sigmoid_68);  getitem_277 = sigmoid_68 = None
        mul_277: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_276, mul_276);  getitem_276 = mul_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_276: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_277, [3072, 192]);  mul_277 = None
        permute_209: "f32[192, 196]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        
        # No stacktrace found for following nodes
        mm_default_41: "f32[3072, 196]" = torch.ops.aten.mm.default(view_276, permute_209);  view_276 = permute_209 = None
        add_tensor_41: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_41, arg128_1);  mm_default_41 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_277: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_41, [8, 384, 196]);  add_tensor_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_210: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
        add_243: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_239, permute_210);  add_239 = permute_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_244: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_243, memory_format = torch.contiguous_format)
        var_mean_70 = torch.ops.aten.var_mean.correction(clone_244, [2], correction = 0, keepdim = True)
        getitem_278: "f32[8, 196, 1]" = var_mean_70[0]
        getitem_279: "f32[8, 196, 1]" = var_mean_70[1];  var_mean_70 = None
        sub_70: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_244, getitem_279);  clone_244 = getitem_279 = None
        add_244: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
        rsqrt_70: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        mul_278: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        mul_279: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_278, arg129_1);  mul_278 = arg129_1 = None
        add_245: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_279, arg130_1);  mul_279 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_278: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_245, [1568, 384]);  add_245 = None
        permute_211: "f32[384, 1536]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        
        # No stacktrace found for following nodes
        mm_default_40: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_278, permute_211);  view_278 = permute_211 = None
        add_tensor_40: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_40, arg132_1);  mm_default_40 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_279: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 196, 1536]);  add_tensor_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_69 = torch.ops.aten.split.Tensor(view_279, 768, -1);  view_279 = None
        getitem_280: "f32[8, 196, 768]" = split_69[0]
        getitem_281: "f32[8, 196, 768]" = split_69[1];  split_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_69: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_281)
        mul_280: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_281, sigmoid_69);  getitem_281 = sigmoid_69 = None
        mul_281: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_280, mul_280);  getitem_280 = mul_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_280: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_281, [1568, 768]);  mul_281 = None
        permute_212: "f32[768, 384]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        
        # No stacktrace found for following nodes
        mm_default_39: "f32[1568, 384]" = torch.ops.aten.mm.default(view_280, permute_212);  view_280 = permute_212 = None
        add_tensor_39: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_39, arg134_1);  mm_default_39 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_281: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 196, 384]);  add_tensor_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_246: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_243, view_281);  add_243 = view_281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_247: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_246, memory_format = torch.contiguous_format)
        var_mean_71 = torch.ops.aten.var_mean.correction(clone_247, [2], correction = 0, keepdim = True)
        getitem_282: "f32[8, 196, 1]" = var_mean_71[0]
        getitem_283: "f32[8, 196, 1]" = var_mean_71[1];  var_mean_71 = None
        sub_71: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_247, getitem_283);  clone_247 = getitem_283 = None
        add_247: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-06);  getitem_282 = None
        rsqrt_71: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_282: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        mul_283: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_282, arg135_1);  mul_282 = arg135_1 = None
        add_248: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_283, arg136_1);  mul_283 = arg136_1 = None
        permute_213: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_248, [0, 2, 1]);  add_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_248: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_282: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_248, [3072, 196]);  clone_248 = None
        permute_214: "f32[196, 384]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_35: "f32[3072, 384]" = torch.ops.aten.mm.default(view_282, permute_214);  view_282 = permute_214 = None
        view_283: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_35, [8, 384, 384]);  mm_35 = None
        add_249: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_283, arg138_1);  view_283 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_70 = torch.ops.aten.split.Tensor(add_249, 192, -1);  add_249 = None
        getitem_284: "f32[8, 384, 192]" = split_70[0]
        getitem_285: "f32[8, 384, 192]" = split_70[1];  split_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_70: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_285)
        mul_284: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_285, sigmoid_70);  getitem_285 = sigmoid_70 = None
        mul_285: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_284, mul_284);  getitem_284 = mul_284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_284: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_285, [3072, 192]);  mul_285 = None
        permute_215: "f32[192, 196]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        
        # No stacktrace found for following nodes
        mm_default_38: "f32[3072, 196]" = torch.ops.aten.mm.default(view_284, permute_215);  view_284 = permute_215 = None
        add_tensor_38: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_38, arg140_1);  mm_default_38 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_285: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 384, 196]);  add_tensor_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_216: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
        add_250: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_246, permute_216);  add_246 = permute_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_251: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_250, memory_format = torch.contiguous_format)
        var_mean_72 = torch.ops.aten.var_mean.correction(clone_251, [2], correction = 0, keepdim = True)
        getitem_286: "f32[8, 196, 1]" = var_mean_72[0]
        getitem_287: "f32[8, 196, 1]" = var_mean_72[1];  var_mean_72 = None
        sub_72: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_251, getitem_287);  clone_251 = getitem_287 = None
        add_251: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_72: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        mul_286: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        mul_287: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_286, arg141_1);  mul_286 = arg141_1 = None
        add_252: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_287, arg142_1);  mul_287 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_286: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_252, [1568, 384]);  add_252 = None
        permute_217: "f32[384, 1536]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_286, permute_217);  view_286 = permute_217 = None
        add_tensor_37: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_37, arg144_1);  mm_default_37 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_287: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 196, 1536]);  add_tensor_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_71 = torch.ops.aten.split.Tensor(view_287, 768, -1);  view_287 = None
        getitem_288: "f32[8, 196, 768]" = split_71[0]
        getitem_289: "f32[8, 196, 768]" = split_71[1];  split_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_71: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_289)
        mul_288: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_289, sigmoid_71);  getitem_289 = sigmoid_71 = None
        mul_289: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_288, mul_288);  getitem_288 = mul_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_288: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_289, [1568, 768]);  mul_289 = None
        permute_218: "f32[768, 384]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[1568, 384]" = torch.ops.aten.mm.default(view_288, permute_218);  view_288 = permute_218 = None
        add_tensor_36: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_36, arg146_1);  mm_default_36 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_289: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 196, 384]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_253: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_250, view_289);  add_250 = view_289 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_254: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_253, memory_format = torch.contiguous_format)
        var_mean_73 = torch.ops.aten.var_mean.correction(clone_254, [2], correction = 0, keepdim = True)
        getitem_290: "f32[8, 196, 1]" = var_mean_73[0]
        getitem_291: "f32[8, 196, 1]" = var_mean_73[1];  var_mean_73 = None
        sub_73: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_254, getitem_291);  clone_254 = getitem_291 = None
        add_254: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-06);  getitem_290 = None
        rsqrt_73: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        mul_290: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        mul_291: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_290, arg147_1);  mul_290 = arg147_1 = None
        add_255: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_291, arg148_1);  mul_291 = arg148_1 = None
        permute_219: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_255, [0, 2, 1]);  add_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_255: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_290: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_255, [3072, 196]);  clone_255 = None
        permute_220: "f32[196, 384]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_36: "f32[3072, 384]" = torch.ops.aten.mm.default(view_290, permute_220);  view_290 = permute_220 = None
        view_291: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_36, [8, 384, 384]);  mm_36 = None
        add_256: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_291, arg150_1);  view_291 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_72 = torch.ops.aten.split.Tensor(add_256, 192, -1);  add_256 = None
        getitem_292: "f32[8, 384, 192]" = split_72[0]
        getitem_293: "f32[8, 384, 192]" = split_72[1];  split_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_72: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_293)
        mul_292: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_293, sigmoid_72);  getitem_293 = sigmoid_72 = None
        mul_293: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_292, mul_292);  getitem_292 = mul_292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_292: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_293, [3072, 192]);  mul_293 = None
        permute_221: "f32[192, 196]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[3072, 196]" = torch.ops.aten.mm.default(view_292, permute_221);  view_292 = permute_221 = None
        add_tensor_35: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_35, arg152_1);  mm_default_35 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_293: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 384, 196]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_222: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
        add_257: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_253, permute_222);  add_253 = permute_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_258: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_257, memory_format = torch.contiguous_format)
        var_mean_74 = torch.ops.aten.var_mean.correction(clone_258, [2], correction = 0, keepdim = True)
        getitem_294: "f32[8, 196, 1]" = var_mean_74[0]
        getitem_295: "f32[8, 196, 1]" = var_mean_74[1];  var_mean_74 = None
        sub_74: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_258, getitem_295);  clone_258 = getitem_295 = None
        add_258: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_294, 1e-06);  getitem_294 = None
        rsqrt_74: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        mul_294: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        mul_295: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_294, arg153_1);  mul_294 = arg153_1 = None
        add_259: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_295, arg154_1);  mul_295 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_294: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_259, [1568, 384]);  add_259 = None
        permute_223: "f32[384, 1536]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_294, permute_223);  view_294 = permute_223 = None
        add_tensor_34: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_34, arg156_1);  mm_default_34 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_295: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 196, 1536]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_73 = torch.ops.aten.split.Tensor(view_295, 768, -1);  view_295 = None
        getitem_296: "f32[8, 196, 768]" = split_73[0]
        getitem_297: "f32[8, 196, 768]" = split_73[1];  split_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_73: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_297)
        mul_296: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_297, sigmoid_73);  getitem_297 = sigmoid_73 = None
        mul_297: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_296, mul_296);  getitem_296 = mul_296 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_296: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_297, [1568, 768]);  mul_297 = None
        permute_224: "f32[768, 384]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[1568, 384]" = torch.ops.aten.mm.default(view_296, permute_224);  view_296 = permute_224 = None
        add_tensor_33: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_33, arg158_1);  mm_default_33 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_297: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 196, 384]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_260: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_257, view_297);  add_257 = view_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_261: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
        var_mean_75 = torch.ops.aten.var_mean.correction(clone_261, [2], correction = 0, keepdim = True)
        getitem_298: "f32[8, 196, 1]" = var_mean_75[0]
        getitem_299: "f32[8, 196, 1]" = var_mean_75[1];  var_mean_75 = None
        sub_75: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_261, getitem_299);  clone_261 = getitem_299 = None
        add_261: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_298, 1e-06);  getitem_298 = None
        rsqrt_75: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        mul_298: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        mul_299: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_298, arg159_1);  mul_298 = arg159_1 = None
        add_262: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_299, arg160_1);  mul_299 = arg160_1 = None
        permute_225: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_262, [0, 2, 1]);  add_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_262: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_298: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_262, [3072, 196]);  clone_262 = None
        permute_226: "f32[196, 384]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        mm_37: "f32[3072, 384]" = torch.ops.aten.mm.default(view_298, permute_226);  view_298 = permute_226 = None
        view_299: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_37, [8, 384, 384]);  mm_37 = None
        add_263: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_299, arg162_1);  view_299 = arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_74 = torch.ops.aten.split.Tensor(add_263, 192, -1);  add_263 = None
        getitem_300: "f32[8, 384, 192]" = split_74[0]
        getitem_301: "f32[8, 384, 192]" = split_74[1];  split_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_74: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_301)
        mul_300: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_301, sigmoid_74);  getitem_301 = sigmoid_74 = None
        mul_301: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_300, mul_300);  getitem_300 = mul_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_300: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_301, [3072, 192]);  mul_301 = None
        permute_227: "f32[192, 196]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[3072, 196]" = torch.ops.aten.mm.default(view_300, permute_227);  view_300 = permute_227 = None
        add_tensor_32: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_32, arg164_1);  mm_default_32 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_301: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 384, 196]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_228: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
        add_264: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_260, permute_228);  add_260 = permute_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_265: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_264, memory_format = torch.contiguous_format)
        var_mean_76 = torch.ops.aten.var_mean.correction(clone_265, [2], correction = 0, keepdim = True)
        getitem_302: "f32[8, 196, 1]" = var_mean_76[0]
        getitem_303: "f32[8, 196, 1]" = var_mean_76[1];  var_mean_76 = None
        sub_76: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_265, getitem_303);  clone_265 = getitem_303 = None
        add_265: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-06);  getitem_302 = None
        rsqrt_76: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
        mul_302: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        mul_303: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_302, arg165_1);  mul_302 = arg165_1 = None
        add_266: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_303, arg166_1);  mul_303 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_302: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_266, [1568, 384]);  add_266 = None
        permute_229: "f32[384, 1536]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_302, permute_229);  view_302 = permute_229 = None
        add_tensor_31: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_31, arg168_1);  mm_default_31 = arg168_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_303: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 196, 1536]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_75 = torch.ops.aten.split.Tensor(view_303, 768, -1);  view_303 = None
        getitem_304: "f32[8, 196, 768]" = split_75[0]
        getitem_305: "f32[8, 196, 768]" = split_75[1];  split_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_75: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_305)
        mul_304: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_305, sigmoid_75);  getitem_305 = sigmoid_75 = None
        mul_305: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_304, mul_304);  getitem_304 = mul_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_304: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_305, [1568, 768]);  mul_305 = None
        permute_230: "f32[768, 384]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[1568, 384]" = torch.ops.aten.mm.default(view_304, permute_230);  view_304 = permute_230 = None
        add_tensor_30: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_30, arg170_1);  mm_default_30 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_305: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 196, 384]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_267: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_264, view_305);  add_264 = view_305 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_268: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_267, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_268, [2], correction = 0, keepdim = True)
        getitem_306: "f32[8, 196, 1]" = var_mean_77[0]
        getitem_307: "f32[8, 196, 1]" = var_mean_77[1];  var_mean_77 = None
        sub_77: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_268, getitem_307);  clone_268 = getitem_307 = None
        add_268: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_306, 1e-06);  getitem_306 = None
        rsqrt_77: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        mul_306: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        mul_307: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_306, arg171_1);  mul_306 = arg171_1 = None
        add_269: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_307, arg172_1);  mul_307 = arg172_1 = None
        permute_231: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_269, [0, 2, 1]);  add_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_269: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_306: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_269, [3072, 196]);  clone_269 = None
        permute_232: "f32[196, 384]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_38: "f32[3072, 384]" = torch.ops.aten.mm.default(view_306, permute_232);  view_306 = permute_232 = None
        view_307: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_38, [8, 384, 384]);  mm_38 = None
        add_270: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_307, arg174_1);  view_307 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_76 = torch.ops.aten.split.Tensor(add_270, 192, -1);  add_270 = None
        getitem_308: "f32[8, 384, 192]" = split_76[0]
        getitem_309: "f32[8, 384, 192]" = split_76[1];  split_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_76: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_309)
        mul_308: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_309, sigmoid_76);  getitem_309 = sigmoid_76 = None
        mul_309: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_308, mul_308);  getitem_308 = mul_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_308: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_309, [3072, 192]);  mul_309 = None
        permute_233: "f32[192, 196]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[3072, 196]" = torch.ops.aten.mm.default(view_308, permute_233);  view_308 = permute_233 = None
        add_tensor_29: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_29, arg176_1);  mm_default_29 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_309: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 384, 196]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_234: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
        add_271: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_267, permute_234);  add_267 = permute_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_272: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_271, memory_format = torch.contiguous_format)
        var_mean_78 = torch.ops.aten.var_mean.correction(clone_272, [2], correction = 0, keepdim = True)
        getitem_310: "f32[8, 196, 1]" = var_mean_78[0]
        getitem_311: "f32[8, 196, 1]" = var_mean_78[1];  var_mean_78 = None
        sub_78: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_272, getitem_311);  clone_272 = getitem_311 = None
        add_272: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-06);  getitem_310 = None
        rsqrt_78: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        mul_310: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        mul_311: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_310, arg177_1);  mul_310 = arg177_1 = None
        add_273: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_311, arg178_1);  mul_311 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_310: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_273, [1568, 384]);  add_273 = None
        permute_235: "f32[384, 1536]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_310, permute_235);  view_310 = permute_235 = None
        add_tensor_28: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_28, arg180_1);  mm_default_28 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_311: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 196, 1536]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_77 = torch.ops.aten.split.Tensor(view_311, 768, -1);  view_311 = None
        getitem_312: "f32[8, 196, 768]" = split_77[0]
        getitem_313: "f32[8, 196, 768]" = split_77[1];  split_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_77: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_313)
        mul_312: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_313, sigmoid_77);  getitem_313 = sigmoid_77 = None
        mul_313: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_312, mul_312);  getitem_312 = mul_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_312: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_313, [1568, 768]);  mul_313 = None
        permute_236: "f32[768, 384]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[1568, 384]" = torch.ops.aten.mm.default(view_312, permute_236);  view_312 = permute_236 = None
        add_tensor_27: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_27, arg182_1);  mm_default_27 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_313: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 196, 384]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_274: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_271, view_313);  add_271 = view_313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_275: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_274, memory_format = torch.contiguous_format)
        var_mean_79 = torch.ops.aten.var_mean.correction(clone_275, [2], correction = 0, keepdim = True)
        getitem_314: "f32[8, 196, 1]" = var_mean_79[0]
        getitem_315: "f32[8, 196, 1]" = var_mean_79[1];  var_mean_79 = None
        sub_79: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_275, getitem_315);  clone_275 = getitem_315 = None
        add_275: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_314, 1e-06);  getitem_314 = None
        rsqrt_79: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        mul_314: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        mul_315: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_314, arg183_1);  mul_314 = arg183_1 = None
        add_276: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_315, arg184_1);  mul_315 = arg184_1 = None
        permute_237: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_276, [0, 2, 1]);  add_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_276: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_314: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_276, [3072, 196]);  clone_276 = None
        permute_238: "f32[196, 384]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_39: "f32[3072, 384]" = torch.ops.aten.mm.default(view_314, permute_238);  view_314 = permute_238 = None
        view_315: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_39, [8, 384, 384]);  mm_39 = None
        add_277: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_315, arg186_1);  view_315 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_78 = torch.ops.aten.split.Tensor(add_277, 192, -1);  add_277 = None
        getitem_316: "f32[8, 384, 192]" = split_78[0]
        getitem_317: "f32[8, 384, 192]" = split_78[1];  split_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_78: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_317)
        mul_316: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_317, sigmoid_78);  getitem_317 = sigmoid_78 = None
        mul_317: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_316, mul_316);  getitem_316 = mul_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_316: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_317, [3072, 192]);  mul_317 = None
        permute_239: "f32[192, 196]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[3072, 196]" = torch.ops.aten.mm.default(view_316, permute_239);  view_316 = permute_239 = None
        add_tensor_26: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_26, arg188_1);  mm_default_26 = arg188_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_317: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 384, 196]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_240: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        add_278: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_274, permute_240);  add_274 = permute_240 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_279: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_278, memory_format = torch.contiguous_format)
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_279, [2], correction = 0, keepdim = True)
        getitem_318: "f32[8, 196, 1]" = var_mean_80[0]
        getitem_319: "f32[8, 196, 1]" = var_mean_80[1];  var_mean_80 = None
        sub_80: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_279, getitem_319);  clone_279 = getitem_319 = None
        add_279: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_318, 1e-06);  getitem_318 = None
        rsqrt_80: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
        mul_318: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        mul_319: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_318, arg189_1);  mul_318 = arg189_1 = None
        add_280: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_319, arg190_1);  mul_319 = arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_318: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_280, [1568, 384]);  add_280 = None
        permute_241: "f32[384, 1536]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_318, permute_241);  view_318 = permute_241 = None
        add_tensor_25: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_25, arg192_1);  mm_default_25 = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_319: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 196, 1536]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_79 = torch.ops.aten.split.Tensor(view_319, 768, -1);  view_319 = None
        getitem_320: "f32[8, 196, 768]" = split_79[0]
        getitem_321: "f32[8, 196, 768]" = split_79[1];  split_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_79: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_321)
        mul_320: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_321, sigmoid_79);  getitem_321 = sigmoid_79 = None
        mul_321: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_320, mul_320);  getitem_320 = mul_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_320: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_321, [1568, 768]);  mul_321 = None
        permute_242: "f32[768, 384]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[1568, 384]" = torch.ops.aten.mm.default(view_320, permute_242);  view_320 = permute_242 = None
        add_tensor_24: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_24, arg194_1);  mm_default_24 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_321: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 196, 384]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_281: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_278, view_321);  add_278 = view_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_282: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_281, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_282, [2], correction = 0, keepdim = True)
        getitem_322: "f32[8, 196, 1]" = var_mean_81[0]
        getitem_323: "f32[8, 196, 1]" = var_mean_81[1];  var_mean_81 = None
        sub_81: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_282, getitem_323);  clone_282 = getitem_323 = None
        add_282: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_322, 1e-06);  getitem_322 = None
        rsqrt_81: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        mul_322: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        mul_323: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_322, arg195_1);  mul_322 = arg195_1 = None
        add_283: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_323, arg196_1);  mul_323 = arg196_1 = None
        permute_243: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_283, [0, 2, 1]);  add_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_283: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_322: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_283, [3072, 196]);  clone_283 = None
        permute_244: "f32[196, 384]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        mm_40: "f32[3072, 384]" = torch.ops.aten.mm.default(view_322, permute_244);  view_322 = permute_244 = None
        view_323: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_40, [8, 384, 384]);  mm_40 = None
        add_284: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_323, arg198_1);  view_323 = arg198_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_80 = torch.ops.aten.split.Tensor(add_284, 192, -1);  add_284 = None
        getitem_324: "f32[8, 384, 192]" = split_80[0]
        getitem_325: "f32[8, 384, 192]" = split_80[1];  split_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_80: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_325)
        mul_324: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_325, sigmoid_80);  getitem_325 = sigmoid_80 = None
        mul_325: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_324, mul_324);  getitem_324 = mul_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_324: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_325, [3072, 192]);  mul_325 = None
        permute_245: "f32[192, 196]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[3072, 196]" = torch.ops.aten.mm.default(view_324, permute_245);  view_324 = permute_245 = None
        add_tensor_23: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_23, arg200_1);  mm_default_23 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_325: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 384, 196]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_246: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
        add_285: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_281, permute_246);  add_281 = permute_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_286: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_285, memory_format = torch.contiguous_format)
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_286, [2], correction = 0, keepdim = True)
        getitem_326: "f32[8, 196, 1]" = var_mean_82[0]
        getitem_327: "f32[8, 196, 1]" = var_mean_82[1];  var_mean_82 = None
        sub_82: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_286, getitem_327);  clone_286 = getitem_327 = None
        add_286: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_326, 1e-06);  getitem_326 = None
        rsqrt_82: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        mul_326: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = rsqrt_82 = None
        mul_327: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_326, arg201_1);  mul_326 = arg201_1 = None
        add_287: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_327, arg202_1);  mul_327 = arg202_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_326: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_287, [1568, 384]);  add_287 = None
        permute_247: "f32[384, 1536]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_326, permute_247);  view_326 = permute_247 = None
        add_tensor_22: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_22, arg204_1);  mm_default_22 = arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_327: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 196, 1536]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_81 = torch.ops.aten.split.Tensor(view_327, 768, -1);  view_327 = None
        getitem_328: "f32[8, 196, 768]" = split_81[0]
        getitem_329: "f32[8, 196, 768]" = split_81[1];  split_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_81: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_329)
        mul_328: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_329, sigmoid_81);  getitem_329 = sigmoid_81 = None
        mul_329: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_328, mul_328);  getitem_328 = mul_328 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_328: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_329, [1568, 768]);  mul_329 = None
        permute_248: "f32[768, 384]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[1568, 384]" = torch.ops.aten.mm.default(view_328, permute_248);  view_328 = permute_248 = None
        add_tensor_21: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_21, arg206_1);  mm_default_21 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_329: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 196, 384]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_288: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_285, view_329);  add_285 = view_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_289: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
        var_mean_83 = torch.ops.aten.var_mean.correction(clone_289, [2], correction = 0, keepdim = True)
        getitem_330: "f32[8, 196, 1]" = var_mean_83[0]
        getitem_331: "f32[8, 196, 1]" = var_mean_83[1];  var_mean_83 = None
        sub_83: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_289, getitem_331);  clone_289 = getitem_331 = None
        add_289: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_330, 1e-06);  getitem_330 = None
        rsqrt_83: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        mul_330: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = rsqrt_83 = None
        mul_331: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_330, arg207_1);  mul_330 = arg207_1 = None
        add_290: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_331, arg208_1);  mul_331 = arg208_1 = None
        permute_249: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_290, [0, 2, 1]);  add_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_290: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_330: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_290, [3072, 196]);  clone_290 = None
        permute_250: "f32[196, 384]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        mm_41: "f32[3072, 384]" = torch.ops.aten.mm.default(view_330, permute_250);  view_330 = permute_250 = None
        view_331: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_41, [8, 384, 384]);  mm_41 = None
        add_291: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_331, arg210_1);  view_331 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_82 = torch.ops.aten.split.Tensor(add_291, 192, -1);  add_291 = None
        getitem_332: "f32[8, 384, 192]" = split_82[0]
        getitem_333: "f32[8, 384, 192]" = split_82[1];  split_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_82: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_333)
        mul_332: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_333, sigmoid_82);  getitem_333 = sigmoid_82 = None
        mul_333: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_332, mul_332);  getitem_332 = mul_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_332: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_333, [3072, 192]);  mul_333 = None
        permute_251: "f32[192, 196]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[3072, 196]" = torch.ops.aten.mm.default(view_332, permute_251);  view_332 = permute_251 = None
        add_tensor_20: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_20, arg212_1);  mm_default_20 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_333: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 384, 196]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_252: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        add_292: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_288, permute_252);  add_288 = permute_252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_293: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_292, memory_format = torch.contiguous_format)
        var_mean_84 = torch.ops.aten.var_mean.correction(clone_293, [2], correction = 0, keepdim = True)
        getitem_334: "f32[8, 196, 1]" = var_mean_84[0]
        getitem_335: "f32[8, 196, 1]" = var_mean_84[1];  var_mean_84 = None
        sub_84: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_293, getitem_335);  clone_293 = getitem_335 = None
        add_293: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_334, 1e-06);  getitem_334 = None
        rsqrt_84: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        mul_334: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = rsqrt_84 = None
        mul_335: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_334, arg213_1);  mul_334 = arg213_1 = None
        add_294: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_335, arg214_1);  mul_335 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_334: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_294, [1568, 384]);  add_294 = None
        permute_253: "f32[384, 1536]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_334, permute_253);  view_334 = permute_253 = None
        add_tensor_19: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_19, arg216_1);  mm_default_19 = arg216_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_335: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 196, 1536]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_83 = torch.ops.aten.split.Tensor(view_335, 768, -1);  view_335 = None
        getitem_336: "f32[8, 196, 768]" = split_83[0]
        getitem_337: "f32[8, 196, 768]" = split_83[1];  split_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_83: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_337)
        mul_336: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_337, sigmoid_83);  getitem_337 = sigmoid_83 = None
        mul_337: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_336, mul_336);  getitem_336 = mul_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_336: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_337, [1568, 768]);  mul_337 = None
        permute_254: "f32[768, 384]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[1568, 384]" = torch.ops.aten.mm.default(view_336, permute_254);  view_336 = permute_254 = None
        add_tensor_18: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_18, arg218_1);  mm_default_18 = arg218_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_337: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 196, 384]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_295: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_292, view_337);  add_292 = view_337 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_296: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_295, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_296, [2], correction = 0, keepdim = True)
        getitem_338: "f32[8, 196, 1]" = var_mean_85[0]
        getitem_339: "f32[8, 196, 1]" = var_mean_85[1];  var_mean_85 = None
        sub_85: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_296, getitem_339);  clone_296 = getitem_339 = None
        add_296: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_85: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        mul_338: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = rsqrt_85 = None
        mul_339: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_338, arg219_1);  mul_338 = arg219_1 = None
        add_297: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_339, arg220_1);  mul_339 = arg220_1 = None
        permute_255: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_297, [0, 2, 1]);  add_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_297: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
        view_338: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_297, [3072, 196]);  clone_297 = None
        permute_256: "f32[196, 384]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        mm_42: "f32[3072, 384]" = torch.ops.aten.mm.default(view_338, permute_256);  view_338 = permute_256 = None
        view_339: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_42, [8, 384, 384]);  mm_42 = None
        add_298: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_339, arg222_1);  view_339 = arg222_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_84 = torch.ops.aten.split.Tensor(add_298, 192, -1);  add_298 = None
        getitem_340: "f32[8, 384, 192]" = split_84[0]
        getitem_341: "f32[8, 384, 192]" = split_84[1];  split_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_84: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_341)
        mul_340: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_341, sigmoid_84);  getitem_341 = sigmoid_84 = None
        mul_341: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_340, mul_340);  getitem_340 = mul_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_340: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_341, [3072, 192]);  mul_341 = None
        permute_257: "f32[192, 196]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[3072, 196]" = torch.ops.aten.mm.default(view_340, permute_257);  view_340 = permute_257 = None
        add_tensor_17: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_17, arg224_1);  mm_default_17 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_341: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 384, 196]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_258: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
        add_299: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_295, permute_258);  add_295 = permute_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_300: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_299, memory_format = torch.contiguous_format)
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_300, [2], correction = 0, keepdim = True)
        getitem_342: "f32[8, 196, 1]" = var_mean_86[0]
        getitem_343: "f32[8, 196, 1]" = var_mean_86[1];  var_mean_86 = None
        sub_86: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_300, getitem_343);  clone_300 = getitem_343 = None
        add_300: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_342, 1e-06);  getitem_342 = None
        rsqrt_86: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
        mul_342: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        mul_343: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_342, arg225_1);  mul_342 = arg225_1 = None
        add_301: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_343, arg226_1);  mul_343 = arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_342: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_301, [1568, 384]);  add_301 = None
        permute_259: "f32[384, 1536]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_342, permute_259);  view_342 = permute_259 = None
        add_tensor_16: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_16, arg228_1);  mm_default_16 = arg228_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_343: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 196, 1536]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_85 = torch.ops.aten.split.Tensor(view_343, 768, -1);  view_343 = None
        getitem_344: "f32[8, 196, 768]" = split_85[0]
        getitem_345: "f32[8, 196, 768]" = split_85[1];  split_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_85: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_345)
        mul_344: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_345, sigmoid_85);  getitem_345 = sigmoid_85 = None
        mul_345: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_344, mul_344);  getitem_344 = mul_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_344: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_345, [1568, 768]);  mul_345 = None
        permute_260: "f32[768, 384]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[1568, 384]" = torch.ops.aten.mm.default(view_344, permute_260);  view_344 = permute_260 = None
        add_tensor_15: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_15, arg230_1);  mm_default_15 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_345: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 196, 384]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_302: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_299, view_345);  add_299 = view_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_303: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_302, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_303, [2], correction = 0, keepdim = True)
        getitem_346: "f32[8, 196, 1]" = var_mean_87[0]
        getitem_347: "f32[8, 196, 1]" = var_mean_87[1];  var_mean_87 = None
        sub_87: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_303, getitem_347);  clone_303 = getitem_347 = None
        add_303: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_346, 1e-06);  getitem_346 = None
        rsqrt_87: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        mul_346: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        mul_347: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_346, arg231_1);  mul_346 = arg231_1 = None
        add_304: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_347, arg232_1);  mul_347 = arg232_1 = None
        permute_261: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_304, [0, 2, 1]);  add_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_304: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_346: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_304, [3072, 196]);  clone_304 = None
        permute_262: "f32[196, 384]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        mm_43: "f32[3072, 384]" = torch.ops.aten.mm.default(view_346, permute_262);  view_346 = permute_262 = None
        view_347: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_43, [8, 384, 384]);  mm_43 = None
        add_305: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_347, arg234_1);  view_347 = arg234_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_86 = torch.ops.aten.split.Tensor(add_305, 192, -1);  add_305 = None
        getitem_348: "f32[8, 384, 192]" = split_86[0]
        getitem_349: "f32[8, 384, 192]" = split_86[1];  split_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_86: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_349)
        mul_348: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_349, sigmoid_86);  getitem_349 = sigmoid_86 = None
        mul_349: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_348, mul_348);  getitem_348 = mul_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_348: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_349, [3072, 192]);  mul_349 = None
        permute_263: "f32[192, 196]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[3072, 196]" = torch.ops.aten.mm.default(view_348, permute_263);  view_348 = permute_263 = None
        add_tensor_14: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_14, arg236_1);  mm_default_14 = arg236_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_349: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 384, 196]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_264: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
        add_306: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_302, permute_264);  add_302 = permute_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_307: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_306, memory_format = torch.contiguous_format)
        var_mean_88 = torch.ops.aten.var_mean.correction(clone_307, [2], correction = 0, keepdim = True)
        getitem_350: "f32[8, 196, 1]" = var_mean_88[0]
        getitem_351: "f32[8, 196, 1]" = var_mean_88[1];  var_mean_88 = None
        sub_88: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_307, getitem_351);  clone_307 = getitem_351 = None
        add_307: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_350, 1e-06);  getitem_350 = None
        rsqrt_88: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
        mul_350: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        mul_351: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_350, arg237_1);  mul_350 = arg237_1 = None
        add_308: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_351, arg238_1);  mul_351 = arg238_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_350: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_308, [1568, 384]);  add_308 = None
        permute_265: "f32[384, 1536]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_350, permute_265);  view_350 = permute_265 = None
        add_tensor_13: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_13, arg240_1);  mm_default_13 = arg240_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_351: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 196, 1536]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_87 = torch.ops.aten.split.Tensor(view_351, 768, -1);  view_351 = None
        getitem_352: "f32[8, 196, 768]" = split_87[0]
        getitem_353: "f32[8, 196, 768]" = split_87[1];  split_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_87: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_353)
        mul_352: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_353, sigmoid_87);  getitem_353 = sigmoid_87 = None
        mul_353: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_352, mul_352);  getitem_352 = mul_352 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_352: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_353, [1568, 768]);  mul_353 = None
        permute_266: "f32[768, 384]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[1568, 384]" = torch.ops.aten.mm.default(view_352, permute_266);  view_352 = permute_266 = None
        add_tensor_12: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_12, arg242_1);  mm_default_12 = arg242_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_353: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 196, 384]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_309: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_306, view_353);  add_306 = view_353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_310: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format)
        var_mean_89 = torch.ops.aten.var_mean.correction(clone_310, [2], correction = 0, keepdim = True)
        getitem_354: "f32[8, 196, 1]" = var_mean_89[0]
        getitem_355: "f32[8, 196, 1]" = var_mean_89[1];  var_mean_89 = None
        sub_89: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_310, getitem_355);  clone_310 = getitem_355 = None
        add_310: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_354, 1e-06);  getitem_354 = None
        rsqrt_89: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        mul_354: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        mul_355: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_354, arg243_1);  mul_354 = arg243_1 = None
        add_311: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_355, arg244_1);  mul_355 = arg244_1 = None
        permute_267: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_311, [0, 2, 1]);  add_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_311: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_354: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_311, [3072, 196]);  clone_311 = None
        permute_268: "f32[196, 384]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        mm_44: "f32[3072, 384]" = torch.ops.aten.mm.default(view_354, permute_268);  view_354 = permute_268 = None
        view_355: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_44, [8, 384, 384]);  mm_44 = None
        add_312: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_355, arg246_1);  view_355 = arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_88 = torch.ops.aten.split.Tensor(add_312, 192, -1);  add_312 = None
        getitem_356: "f32[8, 384, 192]" = split_88[0]
        getitem_357: "f32[8, 384, 192]" = split_88[1];  split_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_88: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_357)
        mul_356: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_357, sigmoid_88);  getitem_357 = sigmoid_88 = None
        mul_357: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_356, mul_356);  getitem_356 = mul_356 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_356: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_357, [3072, 192]);  mul_357 = None
        permute_269: "f32[192, 196]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[3072, 196]" = torch.ops.aten.mm.default(view_356, permute_269);  view_356 = permute_269 = None
        add_tensor_11: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_11, arg248_1);  mm_default_11 = arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_357: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 384, 196]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_270: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
        add_313: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_309, permute_270);  add_309 = permute_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_314: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_313, memory_format = torch.contiguous_format)
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_314, [2], correction = 0, keepdim = True)
        getitem_358: "f32[8, 196, 1]" = var_mean_90[0]
        getitem_359: "f32[8, 196, 1]" = var_mean_90[1];  var_mean_90 = None
        sub_90: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_314, getitem_359);  clone_314 = getitem_359 = None
        add_314: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_358, 1e-06);  getitem_358 = None
        rsqrt_90: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
        mul_358: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        mul_359: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_358, arg249_1);  mul_358 = arg249_1 = None
        add_315: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_359, arg250_1);  mul_359 = arg250_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_358: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_315, [1568, 384]);  add_315 = None
        permute_271: "f32[384, 1536]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_358, permute_271);  view_358 = permute_271 = None
        add_tensor_10: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_10, arg252_1);  mm_default_10 = arg252_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_359: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 1536]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_89 = torch.ops.aten.split.Tensor(view_359, 768, -1);  view_359 = None
        getitem_360: "f32[8, 196, 768]" = split_89[0]
        getitem_361: "f32[8, 196, 768]" = split_89[1];  split_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_89: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_361)
        mul_360: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_361, sigmoid_89);  getitem_361 = sigmoid_89 = None
        mul_361: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_360, mul_360);  getitem_360 = mul_360 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_360: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_361, [1568, 768]);  mul_361 = None
        permute_272: "f32[768, 384]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[1568, 384]" = torch.ops.aten.mm.default(view_360, permute_272);  view_360 = permute_272 = None
        add_tensor_9: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_9, arg254_1);  mm_default_9 = arg254_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_361: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 196, 384]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_316: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_313, view_361);  add_313 = view_361 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_317: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_316, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_317, [2], correction = 0, keepdim = True)
        getitem_362: "f32[8, 196, 1]" = var_mean_91[0]
        getitem_363: "f32[8, 196, 1]" = var_mean_91[1];  var_mean_91 = None
        sub_91: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_317, getitem_363);  clone_317 = getitem_363 = None
        add_317: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_362, 1e-06);  getitem_362 = None
        rsqrt_91: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        mul_362: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        mul_363: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_362, arg255_1);  mul_362 = arg255_1 = None
        add_318: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_363, arg256_1);  mul_363 = arg256_1 = None
        permute_273: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_318, [0, 2, 1]);  add_318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_318: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
        view_362: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_318, [3072, 196]);  clone_318 = None
        permute_274: "f32[196, 384]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        mm_45: "f32[3072, 384]" = torch.ops.aten.mm.default(view_362, permute_274);  view_362 = permute_274 = None
        view_363: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_45, [8, 384, 384]);  mm_45 = None
        add_319: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_363, arg258_1);  view_363 = arg258_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_90 = torch.ops.aten.split.Tensor(add_319, 192, -1);  add_319 = None
        getitem_364: "f32[8, 384, 192]" = split_90[0]
        getitem_365: "f32[8, 384, 192]" = split_90[1];  split_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_90: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_365)
        mul_364: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_365, sigmoid_90);  getitem_365 = sigmoid_90 = None
        mul_365: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_364, mul_364);  getitem_364 = mul_364 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_364: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_365, [3072, 192]);  mul_365 = None
        permute_275: "f32[192, 196]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[3072, 196]" = torch.ops.aten.mm.default(view_364, permute_275);  view_364 = permute_275 = None
        add_tensor_8: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_8, arg260_1);  mm_default_8 = arg260_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_365: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 384, 196]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_276: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
        add_320: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_316, permute_276);  add_316 = permute_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_321: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_320, memory_format = torch.contiguous_format)
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_321, [2], correction = 0, keepdim = True)
        getitem_366: "f32[8, 196, 1]" = var_mean_92[0]
        getitem_367: "f32[8, 196, 1]" = var_mean_92[1];  var_mean_92 = None
        sub_92: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_321, getitem_367);  clone_321 = getitem_367 = None
        add_321: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_366, 1e-06);  getitem_366 = None
        rsqrt_92: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
        mul_366: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        mul_367: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_366, arg261_1);  mul_366 = arg261_1 = None
        add_322: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_367, arg262_1);  mul_367 = arg262_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_366: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_322, [1568, 384]);  add_322 = None
        permute_277: "f32[384, 1536]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_366, permute_277);  view_366 = permute_277 = None
        add_tensor_7: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_7, arg264_1);  mm_default_7 = arg264_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_367: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 196, 1536]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_91 = torch.ops.aten.split.Tensor(view_367, 768, -1);  view_367 = None
        getitem_368: "f32[8, 196, 768]" = split_91[0]
        getitem_369: "f32[8, 196, 768]" = split_91[1];  split_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_91: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_369)
        mul_368: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_369, sigmoid_91);  getitem_369 = sigmoid_91 = None
        mul_369: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_368, mul_368);  getitem_368 = mul_368 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_368: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_369, [1568, 768]);  mul_369 = None
        permute_278: "f32[768, 384]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[1568, 384]" = torch.ops.aten.mm.default(view_368, permute_278);  view_368 = permute_278 = None
        add_tensor_6: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_6, arg266_1);  mm_default_6 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_369: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 196, 384]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_323: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_320, view_369);  add_320 = view_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_324: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_323, memory_format = torch.contiguous_format)
        var_mean_93 = torch.ops.aten.var_mean.correction(clone_324, [2], correction = 0, keepdim = True)
        getitem_370: "f32[8, 196, 1]" = var_mean_93[0]
        getitem_371: "f32[8, 196, 1]" = var_mean_93[1];  var_mean_93 = None
        sub_93: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_324, getitem_371);  clone_324 = getitem_371 = None
        add_324: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_370, 1e-06);  getitem_370 = None
        rsqrt_93: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        mul_370: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        mul_371: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_370, arg267_1);  mul_370 = arg267_1 = None
        add_325: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_371, arg268_1);  mul_371 = arg268_1 = None
        permute_279: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_325, [0, 2, 1]);  add_325 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_325: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
        view_370: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_325, [3072, 196]);  clone_325 = None
        permute_280: "f32[196, 384]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        mm_46: "f32[3072, 384]" = torch.ops.aten.mm.default(view_370, permute_280);  view_370 = permute_280 = None
        view_371: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_46, [8, 384, 384]);  mm_46 = None
        add_326: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_371, arg270_1);  view_371 = arg270_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_92 = torch.ops.aten.split.Tensor(add_326, 192, -1);  add_326 = None
        getitem_372: "f32[8, 384, 192]" = split_92[0]
        getitem_373: "f32[8, 384, 192]" = split_92[1];  split_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_92: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_373)
        mul_372: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_373, sigmoid_92);  getitem_373 = sigmoid_92 = None
        mul_373: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_372, mul_372);  getitem_372 = mul_372 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_372: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_373, [3072, 192]);  mul_373 = None
        permute_281: "f32[192, 196]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[3072, 196]" = torch.ops.aten.mm.default(view_372, permute_281);  view_372 = permute_281 = None
        add_tensor_5: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_5, arg272_1);  mm_default_5 = arg272_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_373: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 384, 196]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_282: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
        add_327: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_323, permute_282);  add_323 = permute_282 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_328: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_327, memory_format = torch.contiguous_format)
        var_mean_94 = torch.ops.aten.var_mean.correction(clone_328, [2], correction = 0, keepdim = True)
        getitem_374: "f32[8, 196, 1]" = var_mean_94[0]
        getitem_375: "f32[8, 196, 1]" = var_mean_94[1];  var_mean_94 = None
        sub_94: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_328, getitem_375);  clone_328 = getitem_375 = None
        add_328: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_374, 1e-06);  getitem_374 = None
        rsqrt_94: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_328);  add_328 = None
        mul_374: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        mul_375: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_374, arg273_1);  mul_374 = arg273_1 = None
        add_329: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_375, arg274_1);  mul_375 = arg274_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_374: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_329, [1568, 384]);  add_329 = None
        permute_283: "f32[384, 1536]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_374, permute_283);  view_374 = permute_283 = None
        add_tensor_4: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_4, arg276_1);  mm_default_4 = arg276_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_375: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 196, 1536]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_93 = torch.ops.aten.split.Tensor(view_375, 768, -1);  view_375 = None
        getitem_376: "f32[8, 196, 768]" = split_93[0]
        getitem_377: "f32[8, 196, 768]" = split_93[1];  split_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_93: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_377)
        mul_376: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_377, sigmoid_93);  getitem_377 = sigmoid_93 = None
        mul_377: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_376, mul_376);  getitem_376 = mul_376 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_376: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_377, [1568, 768]);  mul_377 = None
        permute_284: "f32[768, 384]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[1568, 384]" = torch.ops.aten.mm.default(view_376, permute_284);  view_376 = permute_284 = None
        add_tensor_3: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_3, arg278_1);  mm_default_3 = arg278_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_377: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 196, 384]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_330: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_327, view_377);  add_327 = view_377 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        clone_331: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_331, [2], correction = 0, keepdim = True)
        getitem_378: "f32[8, 196, 1]" = var_mean_95[0]
        getitem_379: "f32[8, 196, 1]" = var_mean_95[1];  var_mean_95 = None
        sub_95: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_331, getitem_379);  clone_331 = getitem_379 = None
        add_331: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_378, 1e-06);  getitem_378 = None
        rsqrt_95: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        mul_378: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        mul_379: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_378, arg279_1);  mul_378 = arg279_1 = None
        add_332: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_379, arg280_1);  mul_379 = arg280_1 = None
        permute_285: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_332, [0, 2, 1]);  add_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        clone_332: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
        view_378: "f32[3072, 196]" = torch.ops.aten.reshape.default(clone_332, [3072, 196]);  clone_332 = None
        permute_286: "f32[196, 384]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        mm_47: "f32[3072, 384]" = torch.ops.aten.mm.default(view_378, permute_286);  view_378 = permute_286 = None
        view_379: "f32[8, 384, 384]" = torch.ops.aten.reshape.default(mm_47, [8, 384, 384]);  mm_47 = None
        add_333: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_379, arg282_1);  view_379 = arg282_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_94 = torch.ops.aten.split.Tensor(add_333, 192, -1);  add_333 = None
        getitem_380: "f32[8, 384, 192]" = split_94[0]
        getitem_381: "f32[8, 384, 192]" = split_94[1];  split_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_94: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_381)
        mul_380: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_381, sigmoid_94);  getitem_381 = sigmoid_94 = None
        mul_381: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_380, mul_380);  getitem_380 = mul_380 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_380: "f32[3072, 192]" = torch.ops.aten.reshape.default(mul_381, [3072, 192]);  mul_381 = None
        permute_287: "f32[192, 196]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[3072, 196]" = torch.ops.aten.mm.default(view_380, permute_287);  view_380 = permute_287 = None
        add_tensor_2: "f32[3072, 196]" = torch.ops.aten.add.Tensor(mm_default_2, arg284_1);  mm_default_2 = arg284_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_381: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 384, 196]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:82 in forward, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        permute_288: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
        add_334: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_330, permute_288);  add_330 = permute_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        clone_335: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_334, memory_format = torch.contiguous_format)
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_335, [2], correction = 0, keepdim = True)
        getitem_382: "f32[8, 196, 1]" = var_mean_96[0]
        getitem_383: "f32[8, 196, 1]" = var_mean_96[1];  var_mean_96 = None
        sub_96: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_335, getitem_383);  clone_335 = getitem_383 = None
        add_335: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_382, 1e-06);  getitem_382 = None
        rsqrt_96: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
        mul_382: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        mul_383: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_382, arg285_1);  mul_382 = arg285_1 = None
        add_336: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_383, arg286_1);  mul_383 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_382: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_336, [1568, 384]);  add_336 = None
        permute_289: "f32[384, 1536]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_382, permute_289);  view_382 = permute_289 = None
        add_tensor_1: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_1, arg288_1);  mm_default_1 = arg288_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:91 in forward, code: x = self.fc1(x)
        view_383: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 196, 1536]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:92 in forward, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
        split_95 = torch.ops.aten.split.Tensor(view_383, 768, -1);  view_383 = None
        getitem_384: "f32[8, 196, 768]" = split_95[0]
        getitem_385: "f32[8, 196, 768]" = split_95[1];  split_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:93 in forward, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        sigmoid_95: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_385)
        mul_384: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_385, sigmoid_95);  getitem_385 = sigmoid_95 = None
        mul_385: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_384, mul_384);  getitem_384 = mul_384 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_384: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_385, [1568, 768]);  mul_385 = None
        permute_290: "f32[768, 384]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[1568, 384]" = torch.ops.aten.mm.default(view_384, permute_290);  view_384 = permute_290 = None
        add_tensor: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default, arg290_1);  mm_default = arg290_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:96 in forward, code: x = self.fc2(x)
        view_385: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor, [8, 196, 384]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:83 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        add_337: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_334, view_385);  add_334 = view_385 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:341 in forward_features, code: x = self.norm(x)
        clone_338: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_337, memory_format = torch.contiguous_format);  add_337 = None
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_338, [2], correction = 0, keepdim = True)
        getitem_386: "f32[8, 196, 1]" = var_mean_97[0]
        getitem_387: "f32[8, 196, 1]" = var_mean_97[1];  var_mean_97 = None
        sub_97: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_338, getitem_387);  clone_338 = getitem_387 = None
        add_338: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_386, 1e-06);  getitem_386 = None
        rsqrt_97: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        mul_386: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        mul_387: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_386, arg291_1);  mul_386 = arg291_1 = None
        add_339: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_387, arg292_1);  mul_387 = arg292_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:346 in forward_head, code: x = x.mean(dim=1)
        mean_1: "f32[8, 384]" = torch.ops.aten.mean.dim(add_339, [1]);  add_339 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:348 in forward_head, code: return x if pre_logits else self.head(x)
        permute_291: "f32[384, 1000]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_145: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg294_1, mean_1, permute_291);  arg294_1 = mean_1 = permute_291 = None
        return (addmm_145,)
        