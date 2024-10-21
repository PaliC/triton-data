class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[256, 3, 16, 16]", arg2_1: "f32[256]", arg3_1: "f32[256]", arg4_1: "f32[256]", arg5_1: "f32[1536, 256]", arg6_1: "f32[1536]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[196, 196]", arg10_1: "f32[196]", arg11_1: "f32[256, 768]", arg12_1: "f32[256]", arg13_1: "f32[256]", arg14_1: "f32[256]", arg15_1: "f32[1536, 256]", arg16_1: "f32[1536]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[196, 196]", arg20_1: "f32[196]", arg21_1: "f32[256, 768]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[256]", arg25_1: "f32[1536, 256]", arg26_1: "f32[1536]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[196, 196]", arg30_1: "f32[196]", arg31_1: "f32[256, 768]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[1536, 256]", arg36_1: "f32[1536]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[196, 196]", arg40_1: "f32[196]", arg41_1: "f32[256, 768]", arg42_1: "f32[256]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[1536, 256]", arg46_1: "f32[1536]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[196, 196]", arg50_1: "f32[196]", arg51_1: "f32[256, 768]", arg52_1: "f32[256]", arg53_1: "f32[256]", arg54_1: "f32[256]", arg55_1: "f32[1536, 256]", arg56_1: "f32[1536]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[196, 196]", arg60_1: "f32[196]", arg61_1: "f32[256, 768]", arg62_1: "f32[256]", arg63_1: "f32[256]", arg64_1: "f32[256]", arg65_1: "f32[1536, 256]", arg66_1: "f32[1536]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[196, 196]", arg70_1: "f32[196]", arg71_1: "f32[256, 768]", arg72_1: "f32[256]", arg73_1: "f32[256]", arg74_1: "f32[256]", arg75_1: "f32[1536, 256]", arg76_1: "f32[1536]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[196, 196]", arg80_1: "f32[196]", arg81_1: "f32[256, 768]", arg82_1: "f32[256]", arg83_1: "f32[256]", arg84_1: "f32[256]", arg85_1: "f32[1536, 256]", arg86_1: "f32[1536]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[196, 196]", arg90_1: "f32[196]", arg91_1: "f32[256, 768]", arg92_1: "f32[256]", arg93_1: "f32[256]", arg94_1: "f32[256]", arg95_1: "f32[1536, 256]", arg96_1: "f32[1536]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[196, 196]", arg100_1: "f32[196]", arg101_1: "f32[256, 768]", arg102_1: "f32[256]", arg103_1: "f32[256]", arg104_1: "f32[256]", arg105_1: "f32[1536, 256]", arg106_1: "f32[1536]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[196, 196]", arg110_1: "f32[196]", arg111_1: "f32[256, 768]", arg112_1: "f32[256]", arg113_1: "f32[256]", arg114_1: "f32[256]", arg115_1: "f32[1536, 256]", arg116_1: "f32[1536]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[196, 196]", arg120_1: "f32[196]", arg121_1: "f32[256, 768]", arg122_1: "f32[256]", arg123_1: "f32[256]", arg124_1: "f32[256]", arg125_1: "f32[1536, 256]", arg126_1: "f32[1536]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[196, 196]", arg130_1: "f32[196]", arg131_1: "f32[256, 768]", arg132_1: "f32[256]", arg133_1: "f32[256]", arg134_1: "f32[256]", arg135_1: "f32[1536, 256]", arg136_1: "f32[1536]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[196, 196]", arg140_1: "f32[196]", arg141_1: "f32[256, 768]", arg142_1: "f32[256]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[1536, 256]", arg146_1: "f32[1536]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[196, 196]", arg150_1: "f32[196]", arg151_1: "f32[256, 768]", arg152_1: "f32[256]", arg153_1: "f32[256]", arg154_1: "f32[256]", arg155_1: "f32[1536, 256]", arg156_1: "f32[1536]", arg157_1: "f32[768]", arg158_1: "f32[768]", arg159_1: "f32[196, 196]", arg160_1: "f32[196]", arg161_1: "f32[256, 768]", arg162_1: "f32[256]", arg163_1: "f32[256]", arg164_1: "f32[256]", arg165_1: "f32[1536, 256]", arg166_1: "f32[1536]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[196, 196]", arg170_1: "f32[196]", arg171_1: "f32[256, 768]", arg172_1: "f32[256]", arg173_1: "f32[256]", arg174_1: "f32[256]", arg175_1: "f32[1536, 256]", arg176_1: "f32[1536]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[196, 196]", arg180_1: "f32[196]", arg181_1: "f32[256, 768]", arg182_1: "f32[256]", arg183_1: "f32[256]", arg184_1: "f32[256]", arg185_1: "f32[1536, 256]", arg186_1: "f32[1536]", arg187_1: "f32[768]", arg188_1: "f32[768]", arg189_1: "f32[196, 196]", arg190_1: "f32[196]", arg191_1: "f32[256, 768]", arg192_1: "f32[256]", arg193_1: "f32[256]", arg194_1: "f32[256]", arg195_1: "f32[1536, 256]", arg196_1: "f32[1536]", arg197_1: "f32[768]", arg198_1: "f32[768]", arg199_1: "f32[196, 196]", arg200_1: "f32[196]", arg201_1: "f32[256, 768]", arg202_1: "f32[256]", arg203_1: "f32[256]", arg204_1: "f32[256]", arg205_1: "f32[1536, 256]", arg206_1: "f32[1536]", arg207_1: "f32[768]", arg208_1: "f32[768]", arg209_1: "f32[196, 196]", arg210_1: "f32[196]", arg211_1: "f32[256, 768]", arg212_1: "f32[256]", arg213_1: "f32[256]", arg214_1: "f32[256]", arg215_1: "f32[1536, 256]", arg216_1: "f32[1536]", arg217_1: "f32[768]", arg218_1: "f32[768]", arg219_1: "f32[196, 196]", arg220_1: "f32[196]", arg221_1: "f32[256, 768]", arg222_1: "f32[256]", arg223_1: "f32[256]", arg224_1: "f32[256]", arg225_1: "f32[1536, 256]", arg226_1: "f32[1536]", arg227_1: "f32[768]", arg228_1: "f32[768]", arg229_1: "f32[196, 196]", arg230_1: "f32[196]", arg231_1: "f32[256, 768]", arg232_1: "f32[256]", arg233_1: "f32[256]", arg234_1: "f32[256]", arg235_1: "f32[1536, 256]", arg236_1: "f32[1536]", arg237_1: "f32[768]", arg238_1: "f32[768]", arg239_1: "f32[196, 196]", arg240_1: "f32[196]", arg241_1: "f32[256, 768]", arg242_1: "f32[256]", arg243_1: "f32[256]", arg244_1: "f32[256]", arg245_1: "f32[1536, 256]", arg246_1: "f32[1536]", arg247_1: "f32[768]", arg248_1: "f32[768]", arg249_1: "f32[196, 196]", arg250_1: "f32[196]", arg251_1: "f32[256, 768]", arg252_1: "f32[256]", arg253_1: "f32[256]", arg254_1: "f32[256]", arg255_1: "f32[1536, 256]", arg256_1: "f32[1536]", arg257_1: "f32[768]", arg258_1: "f32[768]", arg259_1: "f32[196, 196]", arg260_1: "f32[196]", arg261_1: "f32[256, 768]", arg262_1: "f32[256]", arg263_1: "f32[256]", arg264_1: "f32[256]", arg265_1: "f32[1536, 256]", arg266_1: "f32[1536]", arg267_1: "f32[768]", arg268_1: "f32[768]", arg269_1: "f32[196, 196]", arg270_1: "f32[196]", arg271_1: "f32[256, 768]", arg272_1: "f32[256]", arg273_1: "f32[256]", arg274_1: "f32[256]", arg275_1: "f32[1536, 256]", arg276_1: "f32[1536]", arg277_1: "f32[768]", arg278_1: "f32[768]", arg279_1: "f32[196, 196]", arg280_1: "f32[196]", arg281_1: "f32[256, 768]", arg282_1: "f32[256]", arg283_1: "f32[256]", arg284_1: "f32[256]", arg285_1: "f32[1536, 256]", arg286_1: "f32[1536]", arg287_1: "f32[768]", arg288_1: "f32[768]", arg289_1: "f32[196, 196]", arg290_1: "f32[196]", arg291_1: "f32[256, 768]", arg292_1: "f32[256]", arg293_1: "f32[256]", arg294_1: "f32[256]", arg295_1: "f32[1536, 256]", arg296_1: "f32[1536]", arg297_1: "f32[768]", arg298_1: "f32[768]", arg299_1: "f32[196, 196]", arg300_1: "f32[196]", arg301_1: "f32[256, 768]", arg302_1: "f32[256]", arg303_1: "f32[256]", arg304_1: "f32[256]", arg305_1: "f32[1000, 256]", arg306_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_1: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:133 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view_181: "f32[8, 256, 196]" = torch.ops.aten.reshape.default(convolution_1, [8, 256, 196]);  convolution_1 = None
        permute_152: "f32[8, 196, 256]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        clone_152: "f32[8, 196, 256]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format)
        var_mean_61 = torch.ops.aten.var_mean.correction(clone_152, [2], correction = 0, keepdim = True)
        getitem_182: "f32[8, 196, 1]" = var_mean_61[0]
        getitem_183: "f32[8, 196, 1]" = var_mean_61[1];  var_mean_61 = None
        sub_61: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_152, getitem_183);  clone_152 = getitem_183 = None
        add_212: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-06);  getitem_182 = None
        rsqrt_61: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        mul_242: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        mul_243: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_242, arg3_1);  mul_242 = arg3_1 = None
        add_213: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_243, arg4_1);  mul_243 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_182: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_213, [1568, 256]);  add_213 = None
        permute_153: "f32[256, 1536]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        
        # No stacktrace found for following nodes
        mm_default_59: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_182, permute_153);  view_182 = permute_153 = None
        add_tensor_59: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_59, arg6_1);  mm_default_59 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_183: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 196, 1536]);  add_tensor_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_244: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_183, 0.5)
        mul_245: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_183, 0.7071067811865476);  view_183 = None
        erf_30: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_245);  mul_245 = None
        add_214: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_246: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_244, add_214);  mul_244 = add_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_30 = torch.ops.aten.split.Tensor(mul_246, 768, -1);  mul_246 = None
        getitem_184: "f32[8, 196, 768]" = split_30[0]
        getitem_185: "f32[8, 196, 768]" = split_30[1];  split_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_154: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_185, memory_format = torch.contiguous_format);  getitem_185 = None
        var_mean_62 = torch.ops.aten.var_mean.correction(clone_154, [2], correction = 0, keepdim = True)
        getitem_186: "f32[8, 196, 1]" = var_mean_62[0]
        getitem_187: "f32[8, 196, 1]" = var_mean_62[1];  var_mean_62 = None
        sub_62: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_154, getitem_187);  clone_154 = getitem_187 = None
        add_215: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_62: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        mul_247: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        mul_248: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_247, arg7_1);  mul_247 = arg7_1 = None
        add_216: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_248, arg8_1);  mul_248 = arg8_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_154: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_216, [0, 2, 1]);  add_216 = None
        clone_155: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_184: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_155, [6144, 196]);  clone_155 = None
        permute_155: "f32[196, 196]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_30: "f32[6144, 196]" = torch.ops.aten.mm.default(view_184, permute_155);  view_184 = permute_155 = None
        view_185: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_30, [8, 768, 196]);  mm_30 = None
        add_217: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_185, arg10_1);  view_185 = arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_156: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_217, [0, 2, 1]);  add_217 = None
        mul_249: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_184, permute_156);  getitem_184 = permute_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_186: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_249, [1568, 768]);  mul_249 = None
        permute_157: "f32[768, 256]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        
        # No stacktrace found for following nodes
        mm_default_58: "f32[1568, 256]" = torch.ops.aten.mm.default(view_186, permute_157);  view_186 = permute_157 = None
        add_tensor_58: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_58, arg12_1);  mm_default_58 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_187: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 196, 256]);  add_tensor_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_218: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(permute_152, view_187);  permute_152 = view_187 = None
        clone_157: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_218, memory_format = torch.contiguous_format)
        var_mean_63 = torch.ops.aten.var_mean.correction(clone_157, [2], correction = 0, keepdim = True)
        getitem_188: "f32[8, 196, 1]" = var_mean_63[0]
        getitem_189: "f32[8, 196, 1]" = var_mean_63[1];  var_mean_63 = None
        sub_63: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_157, getitem_189);  clone_157 = getitem_189 = None
        add_219: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-06);  getitem_188 = None
        rsqrt_63: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_250: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        mul_251: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_250, arg13_1);  mul_250 = arg13_1 = None
        add_220: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_251, arg14_1);  mul_251 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_188: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_220, [1568, 256]);  add_220 = None
        permute_158: "f32[256, 1536]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        
        # No stacktrace found for following nodes
        mm_default_57: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_188, permute_158);  view_188 = permute_158 = None
        add_tensor_57: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_57, arg16_1);  mm_default_57 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_189: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_57, [8, 196, 1536]);  add_tensor_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_252: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_253: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_31: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_253);  mul_253 = None
        add_221: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_254: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_252, add_221);  mul_252 = add_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_31 = torch.ops.aten.split.Tensor(mul_254, 768, -1);  mul_254 = None
        getitem_190: "f32[8, 196, 768]" = split_31[0]
        getitem_191: "f32[8, 196, 768]" = split_31[1];  split_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_159: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_191, memory_format = torch.contiguous_format);  getitem_191 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(clone_159, [2], correction = 0, keepdim = True)
        getitem_192: "f32[8, 196, 1]" = var_mean_64[0]
        getitem_193: "f32[8, 196, 1]" = var_mean_64[1];  var_mean_64 = None
        sub_64: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_159, getitem_193);  clone_159 = getitem_193 = None
        add_222: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_64: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        mul_255: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        mul_256: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_255, arg17_1);  mul_255 = arg17_1 = None
        add_223: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_256, arg18_1);  mul_256 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_159: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_223, [0, 2, 1]);  add_223 = None
        clone_160: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_190: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_160, [6144, 196]);  clone_160 = None
        permute_160: "f32[196, 196]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_31: "f32[6144, 196]" = torch.ops.aten.mm.default(view_190, permute_160);  view_190 = permute_160 = None
        view_191: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_31, [8, 768, 196]);  mm_31 = None
        add_224: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_191, arg20_1);  view_191 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_161: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_224, [0, 2, 1]);  add_224 = None
        mul_257: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_190, permute_161);  getitem_190 = permute_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_192: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_257, [1568, 768]);  mul_257 = None
        permute_162: "f32[768, 256]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        
        # No stacktrace found for following nodes
        mm_default_56: "f32[1568, 256]" = torch.ops.aten.mm.default(view_192, permute_162);  view_192 = permute_162 = None
        add_tensor_56: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_56, arg22_1);  mm_default_56 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_193: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_56, [8, 196, 256]);  add_tensor_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_225: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_218, view_193);  add_218 = view_193 = None
        clone_162: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_225, memory_format = torch.contiguous_format)
        var_mean_65 = torch.ops.aten.var_mean.correction(clone_162, [2], correction = 0, keepdim = True)
        getitem_194: "f32[8, 196, 1]" = var_mean_65[0]
        getitem_195: "f32[8, 196, 1]" = var_mean_65[1];  var_mean_65 = None
        sub_65: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_162, getitem_195);  clone_162 = getitem_195 = None
        add_226: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-06);  getitem_194 = None
        rsqrt_65: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        mul_258: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        mul_259: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_258, arg23_1);  mul_258 = arg23_1 = None
        add_227: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_259, arg24_1);  mul_259 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_194: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_227, [1568, 256]);  add_227 = None
        permute_163: "f32[256, 1536]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        
        # No stacktrace found for following nodes
        mm_default_55: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_194, permute_163);  view_194 = permute_163 = None
        add_tensor_55: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_55, arg26_1);  mm_default_55 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_195: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_55, [8, 196, 1536]);  add_tensor_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_260: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_261: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_32: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_261);  mul_261 = None
        add_228: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_262: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_260, add_228);  mul_260 = add_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_32 = torch.ops.aten.split.Tensor(mul_262, 768, -1);  mul_262 = None
        getitem_196: "f32[8, 196, 768]" = split_32[0]
        getitem_197: "f32[8, 196, 768]" = split_32[1];  split_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_164: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_197, memory_format = torch.contiguous_format);  getitem_197 = None
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_164, [2], correction = 0, keepdim = True)
        getitem_198: "f32[8, 196, 1]" = var_mean_66[0]
        getitem_199: "f32[8, 196, 1]" = var_mean_66[1];  var_mean_66 = None
        sub_66: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_164, getitem_199);  clone_164 = getitem_199 = None
        add_229: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_66: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_263: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        mul_264: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_263, arg27_1);  mul_263 = arg27_1 = None
        add_230: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_264, arg28_1);  mul_264 = arg28_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_164: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_230, [0, 2, 1]);  add_230 = None
        clone_165: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_196: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_165, [6144, 196]);  clone_165 = None
        permute_165: "f32[196, 196]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_32: "f32[6144, 196]" = torch.ops.aten.mm.default(view_196, permute_165);  view_196 = permute_165 = None
        view_197: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_32, [8, 768, 196]);  mm_32 = None
        add_231: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_197, arg30_1);  view_197 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_166: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_231, [0, 2, 1]);  add_231 = None
        mul_265: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_196, permute_166);  getitem_196 = permute_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_198: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_265, [1568, 768]);  mul_265 = None
        permute_167: "f32[768, 256]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        
        # No stacktrace found for following nodes
        mm_default_54: "f32[1568, 256]" = torch.ops.aten.mm.default(view_198, permute_167);  view_198 = permute_167 = None
        add_tensor_54: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_54, arg32_1);  mm_default_54 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_199: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 196, 256]);  add_tensor_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_232: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_225, view_199);  add_225 = view_199 = None
        clone_167: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_232, memory_format = torch.contiguous_format)
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_167, [2], correction = 0, keepdim = True)
        getitem_200: "f32[8, 196, 1]" = var_mean_67[0]
        getitem_201: "f32[8, 196, 1]" = var_mean_67[1];  var_mean_67 = None
        sub_67: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_167, getitem_201);  clone_167 = getitem_201 = None
        add_233: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
        rsqrt_67: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        mul_266: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        mul_267: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_266, arg33_1);  mul_266 = arg33_1 = None
        add_234: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_267, arg34_1);  mul_267 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_200: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_234, [1568, 256]);  add_234 = None
        permute_168: "f32[256, 1536]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        
        # No stacktrace found for following nodes
        mm_default_53: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_200, permute_168);  view_200 = permute_168 = None
        add_tensor_53: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_53, arg36_1);  mm_default_53 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_201: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 196, 1536]);  add_tensor_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_268: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_201, 0.5)
        mul_269: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476);  view_201 = None
        erf_33: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_269);  mul_269 = None
        add_235: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_270: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_268, add_235);  mul_268 = add_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_33 = torch.ops.aten.split.Tensor(mul_270, 768, -1);  mul_270 = None
        getitem_202: "f32[8, 196, 768]" = split_33[0]
        getitem_203: "f32[8, 196, 768]" = split_33[1];  split_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_169: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_203, memory_format = torch.contiguous_format);  getitem_203 = None
        var_mean_68 = torch.ops.aten.var_mean.correction(clone_169, [2], correction = 0, keepdim = True)
        getitem_204: "f32[8, 196, 1]" = var_mean_68[0]
        getitem_205: "f32[8, 196, 1]" = var_mean_68[1];  var_mean_68 = None
        sub_68: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_169, getitem_205);  clone_169 = getitem_205 = None
        add_236: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_68: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_271: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        mul_272: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_271, arg37_1);  mul_271 = arg37_1 = None
        add_237: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_272, arg38_1);  mul_272 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_169: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_237, [0, 2, 1]);  add_237 = None
        clone_170: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_202: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_170, [6144, 196]);  clone_170 = None
        permute_170: "f32[196, 196]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_33: "f32[6144, 196]" = torch.ops.aten.mm.default(view_202, permute_170);  view_202 = permute_170 = None
        view_203: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_33, [8, 768, 196]);  mm_33 = None
        add_238: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_203, arg40_1);  view_203 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_171: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_238, [0, 2, 1]);  add_238 = None
        mul_273: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_202, permute_171);  getitem_202 = permute_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_204: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_273, [1568, 768]);  mul_273 = None
        permute_172: "f32[768, 256]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        
        # No stacktrace found for following nodes
        mm_default_52: "f32[1568, 256]" = torch.ops.aten.mm.default(view_204, permute_172);  view_204 = permute_172 = None
        add_tensor_52: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_52, arg42_1);  mm_default_52 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_205: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 196, 256]);  add_tensor_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_239: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_232, view_205);  add_232 = view_205 = None
        clone_172: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
        var_mean_69 = torch.ops.aten.var_mean.correction(clone_172, [2], correction = 0, keepdim = True)
        getitem_206: "f32[8, 196, 1]" = var_mean_69[0]
        getitem_207: "f32[8, 196, 1]" = var_mean_69[1];  var_mean_69 = None
        sub_69: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_172, getitem_207);  clone_172 = getitem_207 = None
        add_240: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-06);  getitem_206 = None
        rsqrt_69: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        mul_274: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        mul_275: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_274, arg43_1);  mul_274 = arg43_1 = None
        add_241: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_275, arg44_1);  mul_275 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_206: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_241, [1568, 256]);  add_241 = None
        permute_173: "f32[256, 1536]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        
        # No stacktrace found for following nodes
        mm_default_51: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_206, permute_173);  view_206 = permute_173 = None
        add_tensor_51: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_51, arg46_1);  mm_default_51 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_207: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 196, 1536]);  add_tensor_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_276: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_207, 0.5)
        mul_277: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_207, 0.7071067811865476);  view_207 = None
        erf_34: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_277);  mul_277 = None
        add_242: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_278: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_276, add_242);  mul_276 = add_242 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_34 = torch.ops.aten.split.Tensor(mul_278, 768, -1);  mul_278 = None
        getitem_208: "f32[8, 196, 768]" = split_34[0]
        getitem_209: "f32[8, 196, 768]" = split_34[1];  split_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_174: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_209, memory_format = torch.contiguous_format);  getitem_209 = None
        var_mean_70 = torch.ops.aten.var_mean.correction(clone_174, [2], correction = 0, keepdim = True)
        getitem_210: "f32[8, 196, 1]" = var_mean_70[0]
        getitem_211: "f32[8, 196, 1]" = var_mean_70[1];  var_mean_70 = None
        sub_70: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_174, getitem_211);  clone_174 = getitem_211 = None
        add_243: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_70: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_279: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        mul_280: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_279, arg47_1);  mul_279 = arg47_1 = None
        add_244: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_280, arg48_1);  mul_280 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_174: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_244, [0, 2, 1]);  add_244 = None
        clone_175: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
        view_208: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_175, [6144, 196]);  clone_175 = None
        permute_175: "f32[196, 196]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_34: "f32[6144, 196]" = torch.ops.aten.mm.default(view_208, permute_175);  view_208 = permute_175 = None
        view_209: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_34, [8, 768, 196]);  mm_34 = None
        add_245: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_209, arg50_1);  view_209 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_176: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_245, [0, 2, 1]);  add_245 = None
        mul_281: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_208, permute_176);  getitem_208 = permute_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_210: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_281, [1568, 768]);  mul_281 = None
        permute_177: "f32[768, 256]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        
        # No stacktrace found for following nodes
        mm_default_50: "f32[1568, 256]" = torch.ops.aten.mm.default(view_210, permute_177);  view_210 = permute_177 = None
        add_tensor_50: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_50, arg52_1);  mm_default_50 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_211: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_50, [8, 196, 256]);  add_tensor_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_246: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_239, view_211);  add_239 = view_211 = None
        clone_177: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_246, memory_format = torch.contiguous_format)
        var_mean_71 = torch.ops.aten.var_mean.correction(clone_177, [2], correction = 0, keepdim = True)
        getitem_212: "f32[8, 196, 1]" = var_mean_71[0]
        getitem_213: "f32[8, 196, 1]" = var_mean_71[1];  var_mean_71 = None
        sub_71: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_177, getitem_213);  clone_177 = getitem_213 = None
        add_247: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-06);  getitem_212 = None
        rsqrt_71: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_282: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        mul_283: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_282, arg53_1);  mul_282 = arg53_1 = None
        add_248: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_283, arg54_1);  mul_283 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_212: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_248, [1568, 256]);  add_248 = None
        permute_178: "f32[256, 1536]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        
        # No stacktrace found for following nodes
        mm_default_49: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_212, permute_178);  view_212 = permute_178 = None
        add_tensor_49: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_49, arg56_1);  mm_default_49 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_213: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_49, [8, 196, 1536]);  add_tensor_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_284: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
        mul_285: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
        erf_35: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_285);  mul_285 = None
        add_249: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_286: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_284, add_249);  mul_284 = add_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_35 = torch.ops.aten.split.Tensor(mul_286, 768, -1);  mul_286 = None
        getitem_214: "f32[8, 196, 768]" = split_35[0]
        getitem_215: "f32[8, 196, 768]" = split_35[1];  split_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_179: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_215, memory_format = torch.contiguous_format);  getitem_215 = None
        var_mean_72 = torch.ops.aten.var_mean.correction(clone_179, [2], correction = 0, keepdim = True)
        getitem_216: "f32[8, 196, 1]" = var_mean_72[0]
        getitem_217: "f32[8, 196, 1]" = var_mean_72[1];  var_mean_72 = None
        sub_72: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_179, getitem_217);  clone_179 = getitem_217 = None
        add_250: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_72: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        mul_287: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        mul_288: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_287, arg57_1);  mul_287 = arg57_1 = None
        add_251: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_288, arg58_1);  mul_288 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_179: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_251, [0, 2, 1]);  add_251 = None
        clone_180: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_214: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_180, [6144, 196]);  clone_180 = None
        permute_180: "f32[196, 196]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_35: "f32[6144, 196]" = torch.ops.aten.mm.default(view_214, permute_180);  view_214 = permute_180 = None
        view_215: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_35, [8, 768, 196]);  mm_35 = None
        add_252: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_215, arg60_1);  view_215 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_181: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_252, [0, 2, 1]);  add_252 = None
        mul_289: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_214, permute_181);  getitem_214 = permute_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_216: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_289, [1568, 768]);  mul_289 = None
        permute_182: "f32[768, 256]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        
        # No stacktrace found for following nodes
        mm_default_48: "f32[1568, 256]" = torch.ops.aten.mm.default(view_216, permute_182);  view_216 = permute_182 = None
        add_tensor_48: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_48, arg62_1);  mm_default_48 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_217: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_48, [8, 196, 256]);  add_tensor_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_253: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_246, view_217);  add_246 = view_217 = None
        clone_182: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_253, memory_format = torch.contiguous_format)
        var_mean_73 = torch.ops.aten.var_mean.correction(clone_182, [2], correction = 0, keepdim = True)
        getitem_218: "f32[8, 196, 1]" = var_mean_73[0]
        getitem_219: "f32[8, 196, 1]" = var_mean_73[1];  var_mean_73 = None
        sub_73: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_182, getitem_219);  clone_182 = getitem_219 = None
        add_254: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
        rsqrt_73: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        mul_290: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        mul_291: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_290, arg63_1);  mul_290 = arg63_1 = None
        add_255: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_291, arg64_1);  mul_291 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_218: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_255, [1568, 256]);  add_255 = None
        permute_183: "f32[256, 1536]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        
        # No stacktrace found for following nodes
        mm_default_47: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_218, permute_183);  view_218 = permute_183 = None
        add_tensor_47: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_47, arg66_1);  mm_default_47 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_219: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 196, 1536]);  add_tensor_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_292: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_293: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_36: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_293);  mul_293 = None
        add_256: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_294: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_292, add_256);  mul_292 = add_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_36 = torch.ops.aten.split.Tensor(mul_294, 768, -1);  mul_294 = None
        getitem_220: "f32[8, 196, 768]" = split_36[0]
        getitem_221: "f32[8, 196, 768]" = split_36[1];  split_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_184: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_221, memory_format = torch.contiguous_format);  getitem_221 = None
        var_mean_74 = torch.ops.aten.var_mean.correction(clone_184, [2], correction = 0, keepdim = True)
        getitem_222: "f32[8, 196, 1]" = var_mean_74[0]
        getitem_223: "f32[8, 196, 1]" = var_mean_74[1];  var_mean_74 = None
        sub_74: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_184, getitem_223);  clone_184 = getitem_223 = None
        add_257: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_74: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        mul_295: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        mul_296: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_295, arg67_1);  mul_295 = arg67_1 = None
        add_258: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_296, arg68_1);  mul_296 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_184: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_258, [0, 2, 1]);  add_258 = None
        clone_185: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_220: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_185, [6144, 196]);  clone_185 = None
        permute_185: "f32[196, 196]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_36: "f32[6144, 196]" = torch.ops.aten.mm.default(view_220, permute_185);  view_220 = permute_185 = None
        view_221: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_36, [8, 768, 196]);  mm_36 = None
        add_259: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_221, arg70_1);  view_221 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_186: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_259, [0, 2, 1]);  add_259 = None
        mul_297: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_220, permute_186);  getitem_220 = permute_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_222: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_297, [1568, 768]);  mul_297 = None
        permute_187: "f32[768, 256]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        
        # No stacktrace found for following nodes
        mm_default_46: "f32[1568, 256]" = torch.ops.aten.mm.default(view_222, permute_187);  view_222 = permute_187 = None
        add_tensor_46: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_46, arg72_1);  mm_default_46 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_223: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 196, 256]);  add_tensor_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_260: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_253, view_223);  add_253 = view_223 = None
        clone_187: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
        var_mean_75 = torch.ops.aten.var_mean.correction(clone_187, [2], correction = 0, keepdim = True)
        getitem_224: "f32[8, 196, 1]" = var_mean_75[0]
        getitem_225: "f32[8, 196, 1]" = var_mean_75[1];  var_mean_75 = None
        sub_75: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_187, getitem_225);  clone_187 = getitem_225 = None
        add_261: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-06);  getitem_224 = None
        rsqrt_75: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        mul_298: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        mul_299: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_298, arg73_1);  mul_298 = arg73_1 = None
        add_262: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_299, arg74_1);  mul_299 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_224: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_262, [1568, 256]);  add_262 = None
        permute_188: "f32[256, 1536]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        
        # No stacktrace found for following nodes
        mm_default_45: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_224, permute_188);  view_224 = permute_188 = None
        add_tensor_45: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_45, arg76_1);  mm_default_45 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_225: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 196, 1536]);  add_tensor_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_300: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_225, 0.5)
        mul_301: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_225, 0.7071067811865476);  view_225 = None
        erf_37: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_301);  mul_301 = None
        add_263: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_302: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_300, add_263);  mul_300 = add_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_37 = torch.ops.aten.split.Tensor(mul_302, 768, -1);  mul_302 = None
        getitem_226: "f32[8, 196, 768]" = split_37[0]
        getitem_227: "f32[8, 196, 768]" = split_37[1];  split_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_189: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_227, memory_format = torch.contiguous_format);  getitem_227 = None
        var_mean_76 = torch.ops.aten.var_mean.correction(clone_189, [2], correction = 0, keepdim = True)
        getitem_228: "f32[8, 196, 1]" = var_mean_76[0]
        getitem_229: "f32[8, 196, 1]" = var_mean_76[1];  var_mean_76 = None
        sub_76: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_189, getitem_229);  clone_189 = getitem_229 = None
        add_264: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-05);  getitem_228 = None
        rsqrt_76: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        mul_303: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        mul_304: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_303, arg77_1);  mul_303 = arg77_1 = None
        add_265: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_304, arg78_1);  mul_304 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_189: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_265, [0, 2, 1]);  add_265 = None
        clone_190: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_226: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_190, [6144, 196]);  clone_190 = None
        permute_190: "f32[196, 196]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_37: "f32[6144, 196]" = torch.ops.aten.mm.default(view_226, permute_190);  view_226 = permute_190 = None
        view_227: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_37, [8, 768, 196]);  mm_37 = None
        add_266: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_227, arg80_1);  view_227 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_191: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_266, [0, 2, 1]);  add_266 = None
        mul_305: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_226, permute_191);  getitem_226 = permute_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_228: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_305, [1568, 768]);  mul_305 = None
        permute_192: "f32[768, 256]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        
        # No stacktrace found for following nodes
        mm_default_44: "f32[1568, 256]" = torch.ops.aten.mm.default(view_228, permute_192);  view_228 = permute_192 = None
        add_tensor_44: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_44, arg82_1);  mm_default_44 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_229: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 196, 256]);  add_tensor_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_267: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_260, view_229);  add_260 = view_229 = None
        clone_192: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_267, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_192, [2], correction = 0, keepdim = True)
        getitem_230: "f32[8, 196, 1]" = var_mean_77[0]
        getitem_231: "f32[8, 196, 1]" = var_mean_77[1];  var_mean_77 = None
        sub_77: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_192, getitem_231);  clone_192 = getitem_231 = None
        add_268: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_230, 1e-06);  getitem_230 = None
        rsqrt_77: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        mul_306: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        mul_307: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_306, arg83_1);  mul_306 = arg83_1 = None
        add_269: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_307, arg84_1);  mul_307 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_230: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_269, [1568, 256]);  add_269 = None
        permute_193: "f32[256, 1536]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        
        # No stacktrace found for following nodes
        mm_default_43: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_230, permute_193);  view_230 = permute_193 = None
        add_tensor_43: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_43, arg86_1);  mm_default_43 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_231: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_43, [8, 196, 1536]);  add_tensor_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_308: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_231, 0.5)
        mul_309: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_231, 0.7071067811865476);  view_231 = None
        erf_38: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_309);  mul_309 = None
        add_270: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_310: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_308, add_270);  mul_308 = add_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_38 = torch.ops.aten.split.Tensor(mul_310, 768, -1);  mul_310 = None
        getitem_232: "f32[8, 196, 768]" = split_38[0]
        getitem_233: "f32[8, 196, 768]" = split_38[1];  split_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_194: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_233, memory_format = torch.contiguous_format);  getitem_233 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(clone_194, [2], correction = 0, keepdim = True)
        getitem_234: "f32[8, 196, 1]" = var_mean_78[0]
        getitem_235: "f32[8, 196, 1]" = var_mean_78[1];  var_mean_78 = None
        sub_78: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_194, getitem_235);  clone_194 = getitem_235 = None
        add_271: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-05);  getitem_234 = None
        rsqrt_78: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        mul_311: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        mul_312: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_311, arg87_1);  mul_311 = arg87_1 = None
        add_272: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_312, arg88_1);  mul_312 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_194: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_272, [0, 2, 1]);  add_272 = None
        clone_195: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_232: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_195, [6144, 196]);  clone_195 = None
        permute_195: "f32[196, 196]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_38: "f32[6144, 196]" = torch.ops.aten.mm.default(view_232, permute_195);  view_232 = permute_195 = None
        view_233: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_38, [8, 768, 196]);  mm_38 = None
        add_273: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_233, arg90_1);  view_233 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_196: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_273, [0, 2, 1]);  add_273 = None
        mul_313: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_232, permute_196);  getitem_232 = permute_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_234: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_313, [1568, 768]);  mul_313 = None
        permute_197: "f32[768, 256]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        
        # No stacktrace found for following nodes
        mm_default_42: "f32[1568, 256]" = torch.ops.aten.mm.default(view_234, permute_197);  view_234 = permute_197 = None
        add_tensor_42: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_42, arg92_1);  mm_default_42 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_235: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_42, [8, 196, 256]);  add_tensor_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_274: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_267, view_235);  add_267 = view_235 = None
        clone_197: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_274, memory_format = torch.contiguous_format)
        var_mean_79 = torch.ops.aten.var_mean.correction(clone_197, [2], correction = 0, keepdim = True)
        getitem_236: "f32[8, 196, 1]" = var_mean_79[0]
        getitem_237: "f32[8, 196, 1]" = var_mean_79[1];  var_mean_79 = None
        sub_79: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_197, getitem_237);  clone_197 = getitem_237 = None
        add_275: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_236, 1e-06);  getitem_236 = None
        rsqrt_79: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        mul_314: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        mul_315: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_314, arg93_1);  mul_314 = arg93_1 = None
        add_276: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_315, arg94_1);  mul_315 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_236: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_276, [1568, 256]);  add_276 = None
        permute_198: "f32[256, 1536]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        
        # No stacktrace found for following nodes
        mm_default_41: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_236, permute_198);  view_236 = permute_198 = None
        add_tensor_41: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_41, arg96_1);  mm_default_41 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_237: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_41, [8, 196, 1536]);  add_tensor_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_316: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_237, 0.5)
        mul_317: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_237, 0.7071067811865476);  view_237 = None
        erf_39: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_317);  mul_317 = None
        add_277: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_318: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_316, add_277);  mul_316 = add_277 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_39 = torch.ops.aten.split.Tensor(mul_318, 768, -1);  mul_318 = None
        getitem_238: "f32[8, 196, 768]" = split_39[0]
        getitem_239: "f32[8, 196, 768]" = split_39[1];  split_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_199: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_239, memory_format = torch.contiguous_format);  getitem_239 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_199, [2], correction = 0, keepdim = True)
        getitem_240: "f32[8, 196, 1]" = var_mean_80[0]
        getitem_241: "f32[8, 196, 1]" = var_mean_80[1];  var_mean_80 = None
        sub_80: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_199, getitem_241);  clone_199 = getitem_241 = None
        add_278: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_80: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        mul_319: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        mul_320: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_319, arg97_1);  mul_319 = arg97_1 = None
        add_279: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_320, arg98_1);  mul_320 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_199: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_279, [0, 2, 1]);  add_279 = None
        clone_200: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_238: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_200, [6144, 196]);  clone_200 = None
        permute_200: "f32[196, 196]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_39: "f32[6144, 196]" = torch.ops.aten.mm.default(view_238, permute_200);  view_238 = permute_200 = None
        view_239: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_39, [8, 768, 196]);  mm_39 = None
        add_280: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_239, arg100_1);  view_239 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_201: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_280, [0, 2, 1]);  add_280 = None
        mul_321: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_238, permute_201);  getitem_238 = permute_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_240: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_321, [1568, 768]);  mul_321 = None
        permute_202: "f32[768, 256]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        
        # No stacktrace found for following nodes
        mm_default_40: "f32[1568, 256]" = torch.ops.aten.mm.default(view_240, permute_202);  view_240 = permute_202 = None
        add_tensor_40: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_40, arg102_1);  mm_default_40 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_241: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 196, 256]);  add_tensor_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_281: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_274, view_241);  add_274 = view_241 = None
        clone_202: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_281, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_202, [2], correction = 0, keepdim = True)
        getitem_242: "f32[8, 196, 1]" = var_mean_81[0]
        getitem_243: "f32[8, 196, 1]" = var_mean_81[1];  var_mean_81 = None
        sub_81: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_202, getitem_243);  clone_202 = getitem_243 = None
        add_282: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_81: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        mul_322: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        mul_323: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_322, arg103_1);  mul_322 = arg103_1 = None
        add_283: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_323, arg104_1);  mul_323 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_242: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_283, [1568, 256]);  add_283 = None
        permute_203: "f32[256, 1536]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        
        # No stacktrace found for following nodes
        mm_default_39: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_242, permute_203);  view_242 = permute_203 = None
        add_tensor_39: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_39, arg106_1);  mm_default_39 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_243: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 196, 1536]);  add_tensor_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_324: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_243, 0.5)
        mul_325: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_243, 0.7071067811865476);  view_243 = None
        erf_40: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_325);  mul_325 = None
        add_284: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_326: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_324, add_284);  mul_324 = add_284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_40 = torch.ops.aten.split.Tensor(mul_326, 768, -1);  mul_326 = None
        getitem_244: "f32[8, 196, 768]" = split_40[0]
        getitem_245: "f32[8, 196, 768]" = split_40[1];  split_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_204: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_245, memory_format = torch.contiguous_format);  getitem_245 = None
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_204, [2], correction = 0, keepdim = True)
        getitem_246: "f32[8, 196, 1]" = var_mean_82[0]
        getitem_247: "f32[8, 196, 1]" = var_mean_82[1];  var_mean_82 = None
        sub_82: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_204, getitem_247);  clone_204 = getitem_247 = None
        add_285: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_82: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        mul_327: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = rsqrt_82 = None
        mul_328: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_327, arg107_1);  mul_327 = arg107_1 = None
        add_286: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_328, arg108_1);  mul_328 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_204: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_286, [0, 2, 1]);  add_286 = None
        clone_205: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_244: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_205, [6144, 196]);  clone_205 = None
        permute_205: "f32[196, 196]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_40: "f32[6144, 196]" = torch.ops.aten.mm.default(view_244, permute_205);  view_244 = permute_205 = None
        view_245: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_40, [8, 768, 196]);  mm_40 = None
        add_287: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_245, arg110_1);  view_245 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_206: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_287, [0, 2, 1]);  add_287 = None
        mul_329: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_244, permute_206);  getitem_244 = permute_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_246: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_329, [1568, 768]);  mul_329 = None
        permute_207: "f32[768, 256]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        
        # No stacktrace found for following nodes
        mm_default_38: "f32[1568, 256]" = torch.ops.aten.mm.default(view_246, permute_207);  view_246 = permute_207 = None
        add_tensor_38: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_38, arg112_1);  mm_default_38 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_247: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 196, 256]);  add_tensor_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_288: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_281, view_247);  add_281 = view_247 = None
        clone_207: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
        var_mean_83 = torch.ops.aten.var_mean.correction(clone_207, [2], correction = 0, keepdim = True)
        getitem_248: "f32[8, 196, 1]" = var_mean_83[0]
        getitem_249: "f32[8, 196, 1]" = var_mean_83[1];  var_mean_83 = None
        sub_83: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_207, getitem_249);  clone_207 = getitem_249 = None
        add_289: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-06);  getitem_248 = None
        rsqrt_83: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        mul_330: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = rsqrt_83 = None
        mul_331: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_330, arg113_1);  mul_330 = arg113_1 = None
        add_290: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_331, arg114_1);  mul_331 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_248: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_290, [1568, 256]);  add_290 = None
        permute_208: "f32[256, 1536]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_248, permute_208);  view_248 = permute_208 = None
        add_tensor_37: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_37, arg116_1);  mm_default_37 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_249: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 196, 1536]);  add_tensor_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_332: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
        mul_333: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476);  view_249 = None
        erf_41: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_333);  mul_333 = None
        add_291: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_334: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_332, add_291);  mul_332 = add_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_41 = torch.ops.aten.split.Tensor(mul_334, 768, -1);  mul_334 = None
        getitem_250: "f32[8, 196, 768]" = split_41[0]
        getitem_251: "f32[8, 196, 768]" = split_41[1];  split_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_209: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_251, memory_format = torch.contiguous_format);  getitem_251 = None
        var_mean_84 = torch.ops.aten.var_mean.correction(clone_209, [2], correction = 0, keepdim = True)
        getitem_252: "f32[8, 196, 1]" = var_mean_84[0]
        getitem_253: "f32[8, 196, 1]" = var_mean_84[1];  var_mean_84 = None
        sub_84: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_209, getitem_253);  clone_209 = getitem_253 = None
        add_292: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_84: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        mul_335: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = rsqrt_84 = None
        mul_336: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_335, arg117_1);  mul_335 = arg117_1 = None
        add_293: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_336, arg118_1);  mul_336 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_209: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_293, [0, 2, 1]);  add_293 = None
        clone_210: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        view_250: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_210, [6144, 196]);  clone_210 = None
        permute_210: "f32[196, 196]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_41: "f32[6144, 196]" = torch.ops.aten.mm.default(view_250, permute_210);  view_250 = permute_210 = None
        view_251: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_41, [8, 768, 196]);  mm_41 = None
        add_294: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_251, arg120_1);  view_251 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_211: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_294, [0, 2, 1]);  add_294 = None
        mul_337: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_250, permute_211);  getitem_250 = permute_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_252: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_337, [1568, 768]);  mul_337 = None
        permute_212: "f32[768, 256]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[1568, 256]" = torch.ops.aten.mm.default(view_252, permute_212);  view_252 = permute_212 = None
        add_tensor_36: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_36, arg122_1);  mm_default_36 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_253: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 196, 256]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_295: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_288, view_253);  add_288 = view_253 = None
        clone_212: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_295, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_212, [2], correction = 0, keepdim = True)
        getitem_254: "f32[8, 196, 1]" = var_mean_85[0]
        getitem_255: "f32[8, 196, 1]" = var_mean_85[1];  var_mean_85 = None
        sub_85: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_212, getitem_255);  clone_212 = getitem_255 = None
        add_296: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_85: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        mul_338: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = rsqrt_85 = None
        mul_339: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_338, arg123_1);  mul_338 = arg123_1 = None
        add_297: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_339, arg124_1);  mul_339 = arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_254: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_297, [1568, 256]);  add_297 = None
        permute_213: "f32[256, 1536]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_254, permute_213);  view_254 = permute_213 = None
        add_tensor_35: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_35, arg126_1);  mm_default_35 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_255: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 196, 1536]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_340: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_255, 0.5)
        mul_341: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_255, 0.7071067811865476);  view_255 = None
        erf_42: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_341);  mul_341 = None
        add_298: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_342: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_340, add_298);  mul_340 = add_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_42 = torch.ops.aten.split.Tensor(mul_342, 768, -1);  mul_342 = None
        getitem_256: "f32[8, 196, 768]" = split_42[0]
        getitem_257: "f32[8, 196, 768]" = split_42[1];  split_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_214: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_257, memory_format = torch.contiguous_format);  getitem_257 = None
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_214, [2], correction = 0, keepdim = True)
        getitem_258: "f32[8, 196, 1]" = var_mean_86[0]
        getitem_259: "f32[8, 196, 1]" = var_mean_86[1];  var_mean_86 = None
        sub_86: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_214, getitem_259);  clone_214 = getitem_259 = None
        add_299: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-05);  getitem_258 = None
        rsqrt_86: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        mul_343: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        mul_344: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_343, arg127_1);  mul_343 = arg127_1 = None
        add_300: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_344, arg128_1);  mul_344 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_214: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_300, [0, 2, 1]);  add_300 = None
        clone_215: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_256: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_215, [6144, 196]);  clone_215 = None
        permute_215: "f32[196, 196]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_42: "f32[6144, 196]" = torch.ops.aten.mm.default(view_256, permute_215);  view_256 = permute_215 = None
        view_257: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_42, [8, 768, 196]);  mm_42 = None
        add_301: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_257, arg130_1);  view_257 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_216: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_301, [0, 2, 1]);  add_301 = None
        mul_345: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_256, permute_216);  getitem_256 = permute_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_258: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_345, [1568, 768]);  mul_345 = None
        permute_217: "f32[768, 256]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[1568, 256]" = torch.ops.aten.mm.default(view_258, permute_217);  view_258 = permute_217 = None
        add_tensor_34: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_34, arg132_1);  mm_default_34 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_259: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 196, 256]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_302: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_295, view_259);  add_295 = view_259 = None
        clone_217: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_302, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_217, [2], correction = 0, keepdim = True)
        getitem_260: "f32[8, 196, 1]" = var_mean_87[0]
        getitem_261: "f32[8, 196, 1]" = var_mean_87[1];  var_mean_87 = None
        sub_87: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_217, getitem_261);  clone_217 = getitem_261 = None
        add_303: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-06);  getitem_260 = None
        rsqrt_87: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        mul_346: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        mul_347: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_346, arg133_1);  mul_346 = arg133_1 = None
        add_304: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_347, arg134_1);  mul_347 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_260: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_304, [1568, 256]);  add_304 = None
        permute_218: "f32[256, 1536]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_260, permute_218);  view_260 = permute_218 = None
        add_tensor_33: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_33, arg136_1);  mm_default_33 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_261: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 196, 1536]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_348: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_349: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_43: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_349);  mul_349 = None
        add_305: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_350: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_348, add_305);  mul_348 = add_305 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_43 = torch.ops.aten.split.Tensor(mul_350, 768, -1);  mul_350 = None
        getitem_262: "f32[8, 196, 768]" = split_43[0]
        getitem_263: "f32[8, 196, 768]" = split_43[1];  split_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_219: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_263, memory_format = torch.contiguous_format);  getitem_263 = None
        var_mean_88 = torch.ops.aten.var_mean.correction(clone_219, [2], correction = 0, keepdim = True)
        getitem_264: "f32[8, 196, 1]" = var_mean_88[0]
        getitem_265: "f32[8, 196, 1]" = var_mean_88[1];  var_mean_88 = None
        sub_88: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_219, getitem_265);  clone_219 = getitem_265 = None
        add_306: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-05);  getitem_264 = None
        rsqrt_88: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        mul_351: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        mul_352: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_351, arg137_1);  mul_351 = arg137_1 = None
        add_307: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_352, arg138_1);  mul_352 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_219: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_307, [0, 2, 1]);  add_307 = None
        clone_220: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_262: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_220, [6144, 196]);  clone_220 = None
        permute_220: "f32[196, 196]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        mm_43: "f32[6144, 196]" = torch.ops.aten.mm.default(view_262, permute_220);  view_262 = permute_220 = None
        view_263: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_43, [8, 768, 196]);  mm_43 = None
        add_308: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_263, arg140_1);  view_263 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_221: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_308, [0, 2, 1]);  add_308 = None
        mul_353: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_262, permute_221);  getitem_262 = permute_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_264: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_353, [1568, 768]);  mul_353 = None
        permute_222: "f32[768, 256]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[1568, 256]" = torch.ops.aten.mm.default(view_264, permute_222);  view_264 = permute_222 = None
        add_tensor_32: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_32, arg142_1);  mm_default_32 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_265: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 196, 256]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_309: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_302, view_265);  add_302 = view_265 = None
        clone_222: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format)
        var_mean_89 = torch.ops.aten.var_mean.correction(clone_222, [2], correction = 0, keepdim = True)
        getitem_266: "f32[8, 196, 1]" = var_mean_89[0]
        getitem_267: "f32[8, 196, 1]" = var_mean_89[1];  var_mean_89 = None
        sub_89: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_222, getitem_267);  clone_222 = getitem_267 = None
        add_310: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-06);  getitem_266 = None
        rsqrt_89: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        mul_354: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        mul_355: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_354, arg143_1);  mul_354 = arg143_1 = None
        add_311: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_355, arg144_1);  mul_355 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_266: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_311, [1568, 256]);  add_311 = None
        permute_223: "f32[256, 1536]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_266, permute_223);  view_266 = permute_223 = None
        add_tensor_31: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_31, arg146_1);  mm_default_31 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_267: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 196, 1536]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_356: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
        mul_357: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476);  view_267 = None
        erf_44: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_357);  mul_357 = None
        add_312: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_358: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_356, add_312);  mul_356 = add_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_44 = torch.ops.aten.split.Tensor(mul_358, 768, -1);  mul_358 = None
        getitem_268: "f32[8, 196, 768]" = split_44[0]
        getitem_269: "f32[8, 196, 768]" = split_44[1];  split_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_224: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_269, memory_format = torch.contiguous_format);  getitem_269 = None
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_224, [2], correction = 0, keepdim = True)
        getitem_270: "f32[8, 196, 1]" = var_mean_90[0]
        getitem_271: "f32[8, 196, 1]" = var_mean_90[1];  var_mean_90 = None
        sub_90: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_224, getitem_271);  clone_224 = getitem_271 = None
        add_313: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-05);  getitem_270 = None
        rsqrt_90: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        mul_359: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        mul_360: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_359, arg147_1);  mul_359 = arg147_1 = None
        add_314: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_360, arg148_1);  mul_360 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_224: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_314, [0, 2, 1]);  add_314 = None
        clone_225: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_268: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_225, [6144, 196]);  clone_225 = None
        permute_225: "f32[196, 196]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_44: "f32[6144, 196]" = torch.ops.aten.mm.default(view_268, permute_225);  view_268 = permute_225 = None
        view_269: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_44, [8, 768, 196]);  mm_44 = None
        add_315: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_269, arg150_1);  view_269 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_226: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_315, [0, 2, 1]);  add_315 = None
        mul_361: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_268, permute_226);  getitem_268 = permute_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_270: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_361, [1568, 768]);  mul_361 = None
        permute_227: "f32[768, 256]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[1568, 256]" = torch.ops.aten.mm.default(view_270, permute_227);  view_270 = permute_227 = None
        add_tensor_30: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_30, arg152_1);  mm_default_30 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_271: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 196, 256]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_316: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_309, view_271);  add_309 = view_271 = None
        clone_227: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_316, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_227, [2], correction = 0, keepdim = True)
        getitem_272: "f32[8, 196, 1]" = var_mean_91[0]
        getitem_273: "f32[8, 196, 1]" = var_mean_91[1];  var_mean_91 = None
        sub_91: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_227, getitem_273);  clone_227 = getitem_273 = None
        add_317: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-06);  getitem_272 = None
        rsqrt_91: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        mul_362: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        mul_363: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_362, arg153_1);  mul_362 = arg153_1 = None
        add_318: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_363, arg154_1);  mul_363 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_272: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_318, [1568, 256]);  add_318 = None
        permute_228: "f32[256, 1536]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_272, permute_228);  view_272 = permute_228 = None
        add_tensor_29: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_29, arg156_1);  mm_default_29 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_273: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 196, 1536]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_364: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_273, 0.5)
        mul_365: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_273, 0.7071067811865476);  view_273 = None
        erf_45: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_365);  mul_365 = None
        add_319: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_366: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_364, add_319);  mul_364 = add_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_45 = torch.ops.aten.split.Tensor(mul_366, 768, -1);  mul_366 = None
        getitem_274: "f32[8, 196, 768]" = split_45[0]
        getitem_275: "f32[8, 196, 768]" = split_45[1];  split_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_229: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_275, memory_format = torch.contiguous_format);  getitem_275 = None
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_229, [2], correction = 0, keepdim = True)
        getitem_276: "f32[8, 196, 1]" = var_mean_92[0]
        getitem_277: "f32[8, 196, 1]" = var_mean_92[1];  var_mean_92 = None
        sub_92: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_229, getitem_277);  clone_229 = getitem_277 = None
        add_320: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_276, 1e-05);  getitem_276 = None
        rsqrt_92: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        mul_367: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        mul_368: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_367, arg157_1);  mul_367 = arg157_1 = None
        add_321: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_368, arg158_1);  mul_368 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_229: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_321, [0, 2, 1]);  add_321 = None
        clone_230: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        view_274: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_230, [6144, 196]);  clone_230 = None
        permute_230: "f32[196, 196]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_45: "f32[6144, 196]" = torch.ops.aten.mm.default(view_274, permute_230);  view_274 = permute_230 = None
        view_275: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_45, [8, 768, 196]);  mm_45 = None
        add_322: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_275, arg160_1);  view_275 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_231: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_322, [0, 2, 1]);  add_322 = None
        mul_369: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_274, permute_231);  getitem_274 = permute_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_276: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_369, [1568, 768]);  mul_369 = None
        permute_232: "f32[768, 256]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[1568, 256]" = torch.ops.aten.mm.default(view_276, permute_232);  view_276 = permute_232 = None
        add_tensor_28: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_28, arg162_1);  mm_default_28 = arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_277: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 196, 256]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_323: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_316, view_277);  add_316 = view_277 = None
        clone_232: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_323, memory_format = torch.contiguous_format)
        var_mean_93 = torch.ops.aten.var_mean.correction(clone_232, [2], correction = 0, keepdim = True)
        getitem_278: "f32[8, 196, 1]" = var_mean_93[0]
        getitem_279: "f32[8, 196, 1]" = var_mean_93[1];  var_mean_93 = None
        sub_93: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_232, getitem_279);  clone_232 = getitem_279 = None
        add_324: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
        rsqrt_93: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        mul_370: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        mul_371: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_370, arg163_1);  mul_370 = arg163_1 = None
        add_325: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_371, arg164_1);  mul_371 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_278: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_325, [1568, 256]);  add_325 = None
        permute_233: "f32[256, 1536]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_278, permute_233);  view_278 = permute_233 = None
        add_tensor_27: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_27, arg166_1);  mm_default_27 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_279: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 196, 1536]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_372: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
        mul_373: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476);  view_279 = None
        erf_46: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_373);  mul_373 = None
        add_326: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_374: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_372, add_326);  mul_372 = add_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_46 = torch.ops.aten.split.Tensor(mul_374, 768, -1);  mul_374 = None
        getitem_280: "f32[8, 196, 768]" = split_46[0]
        getitem_281: "f32[8, 196, 768]" = split_46[1];  split_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_234: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_281, memory_format = torch.contiguous_format);  getitem_281 = None
        var_mean_94 = torch.ops.aten.var_mean.correction(clone_234, [2], correction = 0, keepdim = True)
        getitem_282: "f32[8, 196, 1]" = var_mean_94[0]
        getitem_283: "f32[8, 196, 1]" = var_mean_94[1];  var_mean_94 = None
        sub_94: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_234, getitem_283);  clone_234 = getitem_283 = None
        add_327: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-05);  getitem_282 = None
        rsqrt_94: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        mul_375: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        mul_376: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_375, arg167_1);  mul_375 = arg167_1 = None
        add_328: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_376, arg168_1);  mul_376 = arg168_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_234: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_328, [0, 2, 1]);  add_328 = None
        clone_235: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_280: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_235, [6144, 196]);  clone_235 = None
        permute_235: "f32[196, 196]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_46: "f32[6144, 196]" = torch.ops.aten.mm.default(view_280, permute_235);  view_280 = permute_235 = None
        view_281: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_46, [8, 768, 196]);  mm_46 = None
        add_329: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_281, arg170_1);  view_281 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_236: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_329, [0, 2, 1]);  add_329 = None
        mul_377: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_280, permute_236);  getitem_280 = permute_236 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_282: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_377, [1568, 768]);  mul_377 = None
        permute_237: "f32[768, 256]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[1568, 256]" = torch.ops.aten.mm.default(view_282, permute_237);  view_282 = permute_237 = None
        add_tensor_26: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_26, arg172_1);  mm_default_26 = arg172_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_283: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 196, 256]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_330: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_323, view_283);  add_323 = view_283 = None
        clone_237: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_237, [2], correction = 0, keepdim = True)
        getitem_284: "f32[8, 196, 1]" = var_mean_95[0]
        getitem_285: "f32[8, 196, 1]" = var_mean_95[1];  var_mean_95 = None
        sub_95: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_237, getitem_285);  clone_237 = getitem_285 = None
        add_331: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
        rsqrt_95: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        mul_378: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        mul_379: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_378, arg173_1);  mul_378 = arg173_1 = None
        add_332: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_379, arg174_1);  mul_379 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_284: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_332, [1568, 256]);  add_332 = None
        permute_238: "f32[256, 1536]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_284, permute_238);  view_284 = permute_238 = None
        add_tensor_25: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_25, arg176_1);  mm_default_25 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_285: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 196, 1536]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_380: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
        mul_381: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476);  view_285 = None
        erf_47: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_381);  mul_381 = None
        add_333: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_382: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_380, add_333);  mul_380 = add_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_47 = torch.ops.aten.split.Tensor(mul_382, 768, -1);  mul_382 = None
        getitem_286: "f32[8, 196, 768]" = split_47[0]
        getitem_287: "f32[8, 196, 768]" = split_47[1];  split_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_239: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_287, memory_format = torch.contiguous_format);  getitem_287 = None
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_239, [2], correction = 0, keepdim = True)
        getitem_288: "f32[8, 196, 1]" = var_mean_96[0]
        getitem_289: "f32[8, 196, 1]" = var_mean_96[1];  var_mean_96 = None
        sub_96: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_239, getitem_289);  clone_239 = getitem_289 = None
        add_334: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_96: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        mul_383: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        mul_384: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_383, arg177_1);  mul_383 = arg177_1 = None
        add_335: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_384, arg178_1);  mul_384 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_239: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_335, [0, 2, 1]);  add_335 = None
        clone_240: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_286: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_240, [6144, 196]);  clone_240 = None
        permute_240: "f32[196, 196]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        mm_47: "f32[6144, 196]" = torch.ops.aten.mm.default(view_286, permute_240);  view_286 = permute_240 = None
        view_287: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_47, [8, 768, 196]);  mm_47 = None
        add_336: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_287, arg180_1);  view_287 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_241: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_336, [0, 2, 1]);  add_336 = None
        mul_385: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_286, permute_241);  getitem_286 = permute_241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_288: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_385, [1568, 768]);  mul_385 = None
        permute_242: "f32[768, 256]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[1568, 256]" = torch.ops.aten.mm.default(view_288, permute_242);  view_288 = permute_242 = None
        add_tensor_24: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_24, arg182_1);  mm_default_24 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_289: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 196, 256]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_337: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_330, view_289);  add_330 = view_289 = None
        clone_242: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_337, memory_format = torch.contiguous_format)
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_242, [2], correction = 0, keepdim = True)
        getitem_290: "f32[8, 196, 1]" = var_mean_97[0]
        getitem_291: "f32[8, 196, 1]" = var_mean_97[1];  var_mean_97 = None
        sub_97: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_242, getitem_291);  clone_242 = getitem_291 = None
        add_338: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-06);  getitem_290 = None
        rsqrt_97: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        mul_386: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        mul_387: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_386, arg183_1);  mul_386 = arg183_1 = None
        add_339: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_387, arg184_1);  mul_387 = arg184_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_290: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_339, [1568, 256]);  add_339 = None
        permute_243: "f32[256, 1536]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_290, permute_243);  view_290 = permute_243 = None
        add_tensor_23: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_23, arg186_1);  mm_default_23 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_291: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 196, 1536]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_388: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_291, 0.5)
        mul_389: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_291, 0.7071067811865476);  view_291 = None
        erf_48: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_389);  mul_389 = None
        add_340: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_390: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_388, add_340);  mul_388 = add_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_48 = torch.ops.aten.split.Tensor(mul_390, 768, -1);  mul_390 = None
        getitem_292: "f32[8, 196, 768]" = split_48[0]
        getitem_293: "f32[8, 196, 768]" = split_48[1];  split_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_244: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_293, memory_format = torch.contiguous_format);  getitem_293 = None
        var_mean_98 = torch.ops.aten.var_mean.correction(clone_244, [2], correction = 0, keepdim = True)
        getitem_294: "f32[8, 196, 1]" = var_mean_98[0]
        getitem_295: "f32[8, 196, 1]" = var_mean_98[1];  var_mean_98 = None
        sub_98: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_244, getitem_295);  clone_244 = getitem_295 = None
        add_341: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_294, 1e-05);  getitem_294 = None
        rsqrt_98: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        mul_391: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = rsqrt_98 = None
        mul_392: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_391, arg187_1);  mul_391 = arg187_1 = None
        add_342: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_392, arg188_1);  mul_392 = arg188_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_244: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_342, [0, 2, 1]);  add_342 = None
        clone_245: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_292: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_245, [6144, 196]);  clone_245 = None
        permute_245: "f32[196, 196]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_48: "f32[6144, 196]" = torch.ops.aten.mm.default(view_292, permute_245);  view_292 = permute_245 = None
        view_293: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_48, [8, 768, 196]);  mm_48 = None
        add_343: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_293, arg190_1);  view_293 = arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_246: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_343, [0, 2, 1]);  add_343 = None
        mul_393: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_292, permute_246);  getitem_292 = permute_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_294: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_393, [1568, 768]);  mul_393 = None
        permute_247: "f32[768, 256]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[1568, 256]" = torch.ops.aten.mm.default(view_294, permute_247);  view_294 = permute_247 = None
        add_tensor_22: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_22, arg192_1);  mm_default_22 = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_295: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 196, 256]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_344: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_337, view_295);  add_337 = view_295 = None
        clone_247: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_344, memory_format = torch.contiguous_format)
        var_mean_99 = torch.ops.aten.var_mean.correction(clone_247, [2], correction = 0, keepdim = True)
        getitem_296: "f32[8, 196, 1]" = var_mean_99[0]
        getitem_297: "f32[8, 196, 1]" = var_mean_99[1];  var_mean_99 = None
        sub_99: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_247, getitem_297);  clone_247 = getitem_297 = None
        add_345: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-06);  getitem_296 = None
        rsqrt_99: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
        mul_394: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = rsqrt_99 = None
        mul_395: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_394, arg193_1);  mul_394 = arg193_1 = None
        add_346: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_395, arg194_1);  mul_395 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_296: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_346, [1568, 256]);  add_346 = None
        permute_248: "f32[256, 1536]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_296, permute_248);  view_296 = permute_248 = None
        add_tensor_21: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_21, arg196_1);  mm_default_21 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_297: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 196, 1536]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_396: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
        mul_397: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_297, 0.7071067811865476);  view_297 = None
        erf_49: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_397);  mul_397 = None
        add_347: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_398: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_396, add_347);  mul_396 = add_347 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_49 = torch.ops.aten.split.Tensor(mul_398, 768, -1);  mul_398 = None
        getitem_298: "f32[8, 196, 768]" = split_49[0]
        getitem_299: "f32[8, 196, 768]" = split_49[1];  split_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_249: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_299, memory_format = torch.contiguous_format);  getitem_299 = None
        var_mean_100 = torch.ops.aten.var_mean.correction(clone_249, [2], correction = 0, keepdim = True)
        getitem_300: "f32[8, 196, 1]" = var_mean_100[0]
        getitem_301: "f32[8, 196, 1]" = var_mean_100[1];  var_mean_100 = None
        sub_100: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_249, getitem_301);  clone_249 = getitem_301 = None
        add_348: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_300, 1e-05);  getitem_300 = None
        rsqrt_100: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        mul_399: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = rsqrt_100 = None
        mul_400: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_399, arg197_1);  mul_399 = arg197_1 = None
        add_349: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_400, arg198_1);  mul_400 = arg198_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_249: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_349, [0, 2, 1]);  add_349 = None
        clone_250: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_298: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_250, [6144, 196]);  clone_250 = None
        permute_250: "f32[196, 196]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        mm_49: "f32[6144, 196]" = torch.ops.aten.mm.default(view_298, permute_250);  view_298 = permute_250 = None
        view_299: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_49, [8, 768, 196]);  mm_49 = None
        add_350: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_299, arg200_1);  view_299 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_251: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_350, [0, 2, 1]);  add_350 = None
        mul_401: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_298, permute_251);  getitem_298 = permute_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_300: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_401, [1568, 768]);  mul_401 = None
        permute_252: "f32[768, 256]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[1568, 256]" = torch.ops.aten.mm.default(view_300, permute_252);  view_300 = permute_252 = None
        add_tensor_20: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_20, arg202_1);  mm_default_20 = arg202_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_301: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 196, 256]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_351: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_344, view_301);  add_344 = view_301 = None
        clone_252: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_351, memory_format = torch.contiguous_format)
        var_mean_101 = torch.ops.aten.var_mean.correction(clone_252, [2], correction = 0, keepdim = True)
        getitem_302: "f32[8, 196, 1]" = var_mean_101[0]
        getitem_303: "f32[8, 196, 1]" = var_mean_101[1];  var_mean_101 = None
        sub_101: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_252, getitem_303);  clone_252 = getitem_303 = None
        add_352: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-06);  getitem_302 = None
        rsqrt_101: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_352);  add_352 = None
        mul_402: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = rsqrt_101 = None
        mul_403: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_402, arg203_1);  mul_402 = arg203_1 = None
        add_353: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_403, arg204_1);  mul_403 = arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_302: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_353, [1568, 256]);  add_353 = None
        permute_253: "f32[256, 1536]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_302, permute_253);  view_302 = permute_253 = None
        add_tensor_19: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_19, arg206_1);  mm_default_19 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_303: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 196, 1536]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_404: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_303, 0.5)
        mul_405: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_303, 0.7071067811865476);  view_303 = None
        erf_50: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_405);  mul_405 = None
        add_354: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_406: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_404, add_354);  mul_404 = add_354 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_50 = torch.ops.aten.split.Tensor(mul_406, 768, -1);  mul_406 = None
        getitem_304: "f32[8, 196, 768]" = split_50[0]
        getitem_305: "f32[8, 196, 768]" = split_50[1];  split_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_254: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_305, memory_format = torch.contiguous_format);  getitem_305 = None
        var_mean_102 = torch.ops.aten.var_mean.correction(clone_254, [2], correction = 0, keepdim = True)
        getitem_306: "f32[8, 196, 1]" = var_mean_102[0]
        getitem_307: "f32[8, 196, 1]" = var_mean_102[1];  var_mean_102 = None
        sub_102: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_254, getitem_307);  clone_254 = getitem_307 = None
        add_355: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_306, 1e-05);  getitem_306 = None
        rsqrt_102: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        mul_407: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = rsqrt_102 = None
        mul_408: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_407, arg207_1);  mul_407 = arg207_1 = None
        add_356: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_408, arg208_1);  mul_408 = arg208_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_254: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_356, [0, 2, 1]);  add_356 = None
        clone_255: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_304: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_255, [6144, 196]);  clone_255 = None
        permute_255: "f32[196, 196]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        mm_50: "f32[6144, 196]" = torch.ops.aten.mm.default(view_304, permute_255);  view_304 = permute_255 = None
        view_305: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_50, [8, 768, 196]);  mm_50 = None
        add_357: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_305, arg210_1);  view_305 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_256: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_357, [0, 2, 1]);  add_357 = None
        mul_409: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_304, permute_256);  getitem_304 = permute_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_306: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_409, [1568, 768]);  mul_409 = None
        permute_257: "f32[768, 256]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[1568, 256]" = torch.ops.aten.mm.default(view_306, permute_257);  view_306 = permute_257 = None
        add_tensor_18: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_18, arg212_1);  mm_default_18 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_307: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 196, 256]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_358: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_351, view_307);  add_351 = view_307 = None
        clone_257: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_358, memory_format = torch.contiguous_format)
        var_mean_103 = torch.ops.aten.var_mean.correction(clone_257, [2], correction = 0, keepdim = True)
        getitem_308: "f32[8, 196, 1]" = var_mean_103[0]
        getitem_309: "f32[8, 196, 1]" = var_mean_103[1];  var_mean_103 = None
        sub_103: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_257, getitem_309);  clone_257 = getitem_309 = None
        add_359: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-06);  getitem_308 = None
        rsqrt_103: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
        mul_410: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = rsqrt_103 = None
        mul_411: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_410, arg213_1);  mul_410 = arg213_1 = None
        add_360: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_411, arg214_1);  mul_411 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_308: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_360, [1568, 256]);  add_360 = None
        permute_258: "f32[256, 1536]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_308, permute_258);  view_308 = permute_258 = None
        add_tensor_17: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_17, arg216_1);  mm_default_17 = arg216_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_309: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 196, 1536]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_412: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_309, 0.5)
        mul_413: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_309, 0.7071067811865476);  view_309 = None
        erf_51: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_413);  mul_413 = None
        add_361: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_414: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_412, add_361);  mul_412 = add_361 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_51 = torch.ops.aten.split.Tensor(mul_414, 768, -1);  mul_414 = None
        getitem_310: "f32[8, 196, 768]" = split_51[0]
        getitem_311: "f32[8, 196, 768]" = split_51[1];  split_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_259: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_311, memory_format = torch.contiguous_format);  getitem_311 = None
        var_mean_104 = torch.ops.aten.var_mean.correction(clone_259, [2], correction = 0, keepdim = True)
        getitem_312: "f32[8, 196, 1]" = var_mean_104[0]
        getitem_313: "f32[8, 196, 1]" = var_mean_104[1];  var_mean_104 = None
        sub_104: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_259, getitem_313);  clone_259 = getitem_313 = None
        add_362: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_312, 1e-05);  getitem_312 = None
        rsqrt_104: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_362);  add_362 = None
        mul_415: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = rsqrt_104 = None
        mul_416: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_415, arg217_1);  mul_415 = arg217_1 = None
        add_363: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_416, arg218_1);  mul_416 = arg218_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_259: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_363, [0, 2, 1]);  add_363 = None
        clone_260: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_310: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_260, [6144, 196]);  clone_260 = None
        permute_260: "f32[196, 196]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        mm_51: "f32[6144, 196]" = torch.ops.aten.mm.default(view_310, permute_260);  view_310 = permute_260 = None
        view_311: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_51, [8, 768, 196]);  mm_51 = None
        add_364: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_311, arg220_1);  view_311 = arg220_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_261: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_364, [0, 2, 1]);  add_364 = None
        mul_417: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_310, permute_261);  getitem_310 = permute_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_312: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_417, [1568, 768]);  mul_417 = None
        permute_262: "f32[768, 256]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[1568, 256]" = torch.ops.aten.mm.default(view_312, permute_262);  view_312 = permute_262 = None
        add_tensor_16: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_16, arg222_1);  mm_default_16 = arg222_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_313: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 196, 256]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_365: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_358, view_313);  add_358 = view_313 = None
        clone_262: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_365, memory_format = torch.contiguous_format)
        var_mean_105 = torch.ops.aten.var_mean.correction(clone_262, [2], correction = 0, keepdim = True)
        getitem_314: "f32[8, 196, 1]" = var_mean_105[0]
        getitem_315: "f32[8, 196, 1]" = var_mean_105[1];  var_mean_105 = None
        sub_105: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_262, getitem_315);  clone_262 = getitem_315 = None
        add_366: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_314, 1e-06);  getitem_314 = None
        rsqrt_105: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_366);  add_366 = None
        mul_418: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = rsqrt_105 = None
        mul_419: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_418, arg223_1);  mul_418 = arg223_1 = None
        add_367: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_419, arg224_1);  mul_419 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_314: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_367, [1568, 256]);  add_367 = None
        permute_263: "f32[256, 1536]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_314, permute_263);  view_314 = permute_263 = None
        add_tensor_15: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_15, arg226_1);  mm_default_15 = arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_315: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 196, 1536]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_420: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_315, 0.5)
        mul_421: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_315, 0.7071067811865476);  view_315 = None
        erf_52: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_421);  mul_421 = None
        add_368: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_422: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_420, add_368);  mul_420 = add_368 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_52 = torch.ops.aten.split.Tensor(mul_422, 768, -1);  mul_422 = None
        getitem_316: "f32[8, 196, 768]" = split_52[0]
        getitem_317: "f32[8, 196, 768]" = split_52[1];  split_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_264: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_317, memory_format = torch.contiguous_format);  getitem_317 = None
        var_mean_106 = torch.ops.aten.var_mean.correction(clone_264, [2], correction = 0, keepdim = True)
        getitem_318: "f32[8, 196, 1]" = var_mean_106[0]
        getitem_319: "f32[8, 196, 1]" = var_mean_106[1];  var_mean_106 = None
        sub_106: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_264, getitem_319);  clone_264 = getitem_319 = None
        add_369: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_318, 1e-05);  getitem_318 = None
        rsqrt_106: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
        mul_423: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = rsqrt_106 = None
        mul_424: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_423, arg227_1);  mul_423 = arg227_1 = None
        add_370: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_424, arg228_1);  mul_424 = arg228_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_264: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_370, [0, 2, 1]);  add_370 = None
        clone_265: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
        view_316: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_265, [6144, 196]);  clone_265 = None
        permute_265: "f32[196, 196]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        mm_52: "f32[6144, 196]" = torch.ops.aten.mm.default(view_316, permute_265);  view_316 = permute_265 = None
        view_317: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_52, [8, 768, 196]);  mm_52 = None
        add_371: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_317, arg230_1);  view_317 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_266: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_371, [0, 2, 1]);  add_371 = None
        mul_425: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_316, permute_266);  getitem_316 = permute_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_318: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_425, [1568, 768]);  mul_425 = None
        permute_267: "f32[768, 256]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[1568, 256]" = torch.ops.aten.mm.default(view_318, permute_267);  view_318 = permute_267 = None
        add_tensor_14: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_14, arg232_1);  mm_default_14 = arg232_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_319: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 196, 256]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_372: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_365, view_319);  add_365 = view_319 = None
        clone_267: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_372, memory_format = torch.contiguous_format)
        var_mean_107 = torch.ops.aten.var_mean.correction(clone_267, [2], correction = 0, keepdim = True)
        getitem_320: "f32[8, 196, 1]" = var_mean_107[0]
        getitem_321: "f32[8, 196, 1]" = var_mean_107[1];  var_mean_107 = None
        sub_107: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_267, getitem_321);  clone_267 = getitem_321 = None
        add_373: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_320, 1e-06);  getitem_320 = None
        rsqrt_107: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
        mul_426: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = rsqrt_107 = None
        mul_427: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_426, arg233_1);  mul_426 = arg233_1 = None
        add_374: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_427, arg234_1);  mul_427 = arg234_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_320: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_374, [1568, 256]);  add_374 = None
        permute_268: "f32[256, 1536]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_320, permute_268);  view_320 = permute_268 = None
        add_tensor_13: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_13, arg236_1);  mm_default_13 = arg236_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_321: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 196, 1536]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_428: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_321, 0.5)
        mul_429: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_321, 0.7071067811865476);  view_321 = None
        erf_53: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_429);  mul_429 = None
        add_375: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_430: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_428, add_375);  mul_428 = add_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_53 = torch.ops.aten.split.Tensor(mul_430, 768, -1);  mul_430 = None
        getitem_322: "f32[8, 196, 768]" = split_53[0]
        getitem_323: "f32[8, 196, 768]" = split_53[1];  split_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_269: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_323, memory_format = torch.contiguous_format);  getitem_323 = None
        var_mean_108 = torch.ops.aten.var_mean.correction(clone_269, [2], correction = 0, keepdim = True)
        getitem_324: "f32[8, 196, 1]" = var_mean_108[0]
        getitem_325: "f32[8, 196, 1]" = var_mean_108[1];  var_mean_108 = None
        sub_108: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_269, getitem_325);  clone_269 = getitem_325 = None
        add_376: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_324, 1e-05);  getitem_324 = None
        rsqrt_108: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        mul_431: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = rsqrt_108 = None
        mul_432: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_431, arg237_1);  mul_431 = arg237_1 = None
        add_377: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_432, arg238_1);  mul_432 = arg238_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_269: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_377, [0, 2, 1]);  add_377 = None
        clone_270: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_322: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_270, [6144, 196]);  clone_270 = None
        permute_270: "f32[196, 196]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_53: "f32[6144, 196]" = torch.ops.aten.mm.default(view_322, permute_270);  view_322 = permute_270 = None
        view_323: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_53, [8, 768, 196]);  mm_53 = None
        add_378: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_323, arg240_1);  view_323 = arg240_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_271: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_378, [0, 2, 1]);  add_378 = None
        mul_433: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_322, permute_271);  getitem_322 = permute_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_324: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_433, [1568, 768]);  mul_433 = None
        permute_272: "f32[768, 256]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[1568, 256]" = torch.ops.aten.mm.default(view_324, permute_272);  view_324 = permute_272 = None
        add_tensor_12: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_12, arg242_1);  mm_default_12 = arg242_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_325: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 196, 256]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_379: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_372, view_325);  add_372 = view_325 = None
        clone_272: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_379, memory_format = torch.contiguous_format)
        var_mean_109 = torch.ops.aten.var_mean.correction(clone_272, [2], correction = 0, keepdim = True)
        getitem_326: "f32[8, 196, 1]" = var_mean_109[0]
        getitem_327: "f32[8, 196, 1]" = var_mean_109[1];  var_mean_109 = None
        sub_109: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_272, getitem_327);  clone_272 = getitem_327 = None
        add_380: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_326, 1e-06);  getitem_326 = None
        rsqrt_109: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_380);  add_380 = None
        mul_434: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = rsqrt_109 = None
        mul_435: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_434, arg243_1);  mul_434 = arg243_1 = None
        add_381: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_435, arg244_1);  mul_435 = arg244_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_326: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_381, [1568, 256]);  add_381 = None
        permute_273: "f32[256, 1536]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_326, permute_273);  view_326 = permute_273 = None
        add_tensor_11: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_11, arg246_1);  mm_default_11 = arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_327: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 196, 1536]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_436: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_437: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
        erf_54: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_437);  mul_437 = None
        add_382: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_438: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_436, add_382);  mul_436 = add_382 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_54 = torch.ops.aten.split.Tensor(mul_438, 768, -1);  mul_438 = None
        getitem_328: "f32[8, 196, 768]" = split_54[0]
        getitem_329: "f32[8, 196, 768]" = split_54[1];  split_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_274: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_329, memory_format = torch.contiguous_format);  getitem_329 = None
        var_mean_110 = torch.ops.aten.var_mean.correction(clone_274, [2], correction = 0, keepdim = True)
        getitem_330: "f32[8, 196, 1]" = var_mean_110[0]
        getitem_331: "f32[8, 196, 1]" = var_mean_110[1];  var_mean_110 = None
        sub_110: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_274, getitem_331);  clone_274 = getitem_331 = None
        add_383: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_330, 1e-05);  getitem_330 = None
        rsqrt_110: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
        mul_439: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = rsqrt_110 = None
        mul_440: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_439, arg247_1);  mul_439 = arg247_1 = None
        add_384: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_440, arg248_1);  mul_440 = arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_274: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_384, [0, 2, 1]);  add_384 = None
        clone_275: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
        view_328: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_275, [6144, 196]);  clone_275 = None
        permute_275: "f32[196, 196]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        mm_54: "f32[6144, 196]" = torch.ops.aten.mm.default(view_328, permute_275);  view_328 = permute_275 = None
        view_329: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_54, [8, 768, 196]);  mm_54 = None
        add_385: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_329, arg250_1);  view_329 = arg250_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_276: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_385, [0, 2, 1]);  add_385 = None
        mul_441: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_328, permute_276);  getitem_328 = permute_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_330: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_441, [1568, 768]);  mul_441 = None
        permute_277: "f32[768, 256]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[1568, 256]" = torch.ops.aten.mm.default(view_330, permute_277);  view_330 = permute_277 = None
        add_tensor_10: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_10, arg252_1);  mm_default_10 = arg252_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_331: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 256]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_386: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_379, view_331);  add_379 = view_331 = None
        clone_277: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_386, memory_format = torch.contiguous_format)
        var_mean_111 = torch.ops.aten.var_mean.correction(clone_277, [2], correction = 0, keepdim = True)
        getitem_332: "f32[8, 196, 1]" = var_mean_111[0]
        getitem_333: "f32[8, 196, 1]" = var_mean_111[1];  var_mean_111 = None
        sub_111: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_277, getitem_333);  clone_277 = getitem_333 = None
        add_387: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_332, 1e-06);  getitem_332 = None
        rsqrt_111: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_387);  add_387 = None
        mul_442: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = rsqrt_111 = None
        mul_443: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_442, arg253_1);  mul_442 = arg253_1 = None
        add_388: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_443, arg254_1);  mul_443 = arg254_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_332: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_388, [1568, 256]);  add_388 = None
        permute_278: "f32[256, 1536]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_332, permute_278);  view_332 = permute_278 = None
        add_tensor_9: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_9, arg256_1);  mm_default_9 = arg256_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_333: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 196, 1536]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_444: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_333, 0.5)
        mul_445: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_333, 0.7071067811865476);  view_333 = None
        erf_55: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_445);  mul_445 = None
        add_389: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_446: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_444, add_389);  mul_444 = add_389 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_55 = torch.ops.aten.split.Tensor(mul_446, 768, -1);  mul_446 = None
        getitem_334: "f32[8, 196, 768]" = split_55[0]
        getitem_335: "f32[8, 196, 768]" = split_55[1];  split_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_279: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_335, memory_format = torch.contiguous_format);  getitem_335 = None
        var_mean_112 = torch.ops.aten.var_mean.correction(clone_279, [2], correction = 0, keepdim = True)
        getitem_336: "f32[8, 196, 1]" = var_mean_112[0]
        getitem_337: "f32[8, 196, 1]" = var_mean_112[1];  var_mean_112 = None
        sub_112: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_279, getitem_337);  clone_279 = getitem_337 = None
        add_390: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_112: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_390);  add_390 = None
        mul_447: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = rsqrt_112 = None
        mul_448: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_447, arg257_1);  mul_447 = arg257_1 = None
        add_391: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_448, arg258_1);  mul_448 = arg258_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_279: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_391, [0, 2, 1]);  add_391 = None
        clone_280: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
        view_334: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_280, [6144, 196]);  clone_280 = None
        permute_280: "f32[196, 196]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        mm_55: "f32[6144, 196]" = torch.ops.aten.mm.default(view_334, permute_280);  view_334 = permute_280 = None
        view_335: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_55, [8, 768, 196]);  mm_55 = None
        add_392: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_335, arg260_1);  view_335 = arg260_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_281: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_392, [0, 2, 1]);  add_392 = None
        mul_449: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_334, permute_281);  getitem_334 = permute_281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_336: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_449, [1568, 768]);  mul_449 = None
        permute_282: "f32[768, 256]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[1568, 256]" = torch.ops.aten.mm.default(view_336, permute_282);  view_336 = permute_282 = None
        add_tensor_8: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_8, arg262_1);  mm_default_8 = arg262_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_337: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 196, 256]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_393: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_386, view_337);  add_386 = view_337 = None
        clone_282: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_393, memory_format = torch.contiguous_format)
        var_mean_113 = torch.ops.aten.var_mean.correction(clone_282, [2], correction = 0, keepdim = True)
        getitem_338: "f32[8, 196, 1]" = var_mean_113[0]
        getitem_339: "f32[8, 196, 1]" = var_mean_113[1];  var_mean_113 = None
        sub_113: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_282, getitem_339);  clone_282 = getitem_339 = None
        add_394: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_113: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
        mul_450: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = rsqrt_113 = None
        mul_451: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_450, arg263_1);  mul_450 = arg263_1 = None
        add_395: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_451, arg264_1);  mul_451 = arg264_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_338: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_395, [1568, 256]);  add_395 = None
        permute_283: "f32[256, 1536]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_338, permute_283);  view_338 = permute_283 = None
        add_tensor_7: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_7, arg266_1);  mm_default_7 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_339: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 196, 1536]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_452: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_339, 0.5)
        mul_453: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_339, 0.7071067811865476);  view_339 = None
        erf_56: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_453);  mul_453 = None
        add_396: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_454: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_452, add_396);  mul_452 = add_396 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_56 = torch.ops.aten.split.Tensor(mul_454, 768, -1);  mul_454 = None
        getitem_340: "f32[8, 196, 768]" = split_56[0]
        getitem_341: "f32[8, 196, 768]" = split_56[1];  split_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_284: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_341, memory_format = torch.contiguous_format);  getitem_341 = None
        var_mean_114 = torch.ops.aten.var_mean.correction(clone_284, [2], correction = 0, keepdim = True)
        getitem_342: "f32[8, 196, 1]" = var_mean_114[0]
        getitem_343: "f32[8, 196, 1]" = var_mean_114[1];  var_mean_114 = None
        sub_114: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_284, getitem_343);  clone_284 = getitem_343 = None
        add_397: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_342, 1e-05);  getitem_342 = None
        rsqrt_114: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
        mul_455: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = rsqrt_114 = None
        mul_456: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_455, arg267_1);  mul_455 = arg267_1 = None
        add_398: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_456, arg268_1);  mul_456 = arg268_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_284: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_398, [0, 2, 1]);  add_398 = None
        clone_285: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
        view_340: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_285, [6144, 196]);  clone_285 = None
        permute_285: "f32[196, 196]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        mm_56: "f32[6144, 196]" = torch.ops.aten.mm.default(view_340, permute_285);  view_340 = permute_285 = None
        view_341: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_56, [8, 768, 196]);  mm_56 = None
        add_399: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_341, arg270_1);  view_341 = arg270_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_286: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_399, [0, 2, 1]);  add_399 = None
        mul_457: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_340, permute_286);  getitem_340 = permute_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_342: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_457, [1568, 768]);  mul_457 = None
        permute_287: "f32[768, 256]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[1568, 256]" = torch.ops.aten.mm.default(view_342, permute_287);  view_342 = permute_287 = None
        add_tensor_6: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_6, arg272_1);  mm_default_6 = arg272_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_343: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 196, 256]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_400: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_393, view_343);  add_393 = view_343 = None
        clone_287: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_400, memory_format = torch.contiguous_format)
        var_mean_115 = torch.ops.aten.var_mean.correction(clone_287, [2], correction = 0, keepdim = True)
        getitem_344: "f32[8, 196, 1]" = var_mean_115[0]
        getitem_345: "f32[8, 196, 1]" = var_mean_115[1];  var_mean_115 = None
        sub_115: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_287, getitem_345);  clone_287 = getitem_345 = None
        add_401: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_344, 1e-06);  getitem_344 = None
        rsqrt_115: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
        mul_458: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = rsqrt_115 = None
        mul_459: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_458, arg273_1);  mul_458 = arg273_1 = None
        add_402: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_459, arg274_1);  mul_459 = arg274_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_344: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_402, [1568, 256]);  add_402 = None
        permute_288: "f32[256, 1536]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_344, permute_288);  view_344 = permute_288 = None
        add_tensor_5: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_5, arg276_1);  mm_default_5 = arg276_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_345: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 196, 1536]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_460: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_345, 0.5)
        mul_461: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_345, 0.7071067811865476);  view_345 = None
        erf_57: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_461);  mul_461 = None
        add_403: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_462: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_460, add_403);  mul_460 = add_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_57 = torch.ops.aten.split.Tensor(mul_462, 768, -1);  mul_462 = None
        getitem_346: "f32[8, 196, 768]" = split_57[0]
        getitem_347: "f32[8, 196, 768]" = split_57[1];  split_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_289: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_347, memory_format = torch.contiguous_format);  getitem_347 = None
        var_mean_116 = torch.ops.aten.var_mean.correction(clone_289, [2], correction = 0, keepdim = True)
        getitem_348: "f32[8, 196, 1]" = var_mean_116[0]
        getitem_349: "f32[8, 196, 1]" = var_mean_116[1];  var_mean_116 = None
        sub_116: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_289, getitem_349);  clone_289 = getitem_349 = None
        add_404: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_348, 1e-05);  getitem_348 = None
        rsqrt_116: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
        mul_463: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = rsqrt_116 = None
        mul_464: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_463, arg277_1);  mul_463 = arg277_1 = None
        add_405: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_464, arg278_1);  mul_464 = arg278_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_289: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_405, [0, 2, 1]);  add_405 = None
        clone_290: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
        view_346: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_290, [6144, 196]);  clone_290 = None
        permute_290: "f32[196, 196]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        mm_57: "f32[6144, 196]" = torch.ops.aten.mm.default(view_346, permute_290);  view_346 = permute_290 = None
        view_347: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_57, [8, 768, 196]);  mm_57 = None
        add_406: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_347, arg280_1);  view_347 = arg280_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_291: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_406, [0, 2, 1]);  add_406 = None
        mul_465: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_346, permute_291);  getitem_346 = permute_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_348: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_465, [1568, 768]);  mul_465 = None
        permute_292: "f32[768, 256]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[1568, 256]" = torch.ops.aten.mm.default(view_348, permute_292);  view_348 = permute_292 = None
        add_tensor_4: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_4, arg282_1);  mm_default_4 = arg282_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_349: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 196, 256]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_407: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_400, view_349);  add_400 = view_349 = None
        clone_292: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_407, memory_format = torch.contiguous_format)
        var_mean_117 = torch.ops.aten.var_mean.correction(clone_292, [2], correction = 0, keepdim = True)
        getitem_350: "f32[8, 196, 1]" = var_mean_117[0]
        getitem_351: "f32[8, 196, 1]" = var_mean_117[1];  var_mean_117 = None
        sub_117: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_292, getitem_351);  clone_292 = getitem_351 = None
        add_408: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_350, 1e-06);  getitem_350 = None
        rsqrt_117: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_408);  add_408 = None
        mul_466: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = rsqrt_117 = None
        mul_467: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_466, arg283_1);  mul_466 = arg283_1 = None
        add_409: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_467, arg284_1);  mul_467 = arg284_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_350: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_409, [1568, 256]);  add_409 = None
        permute_293: "f32[256, 1536]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_350, permute_293);  view_350 = permute_293 = None
        add_tensor_3: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_3, arg286_1);  mm_default_3 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_351: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 196, 1536]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_468: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_351, 0.5)
        mul_469: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_351, 0.7071067811865476);  view_351 = None
        erf_58: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_469);  mul_469 = None
        add_410: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_470: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_468, add_410);  mul_468 = add_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_58 = torch.ops.aten.split.Tensor(mul_470, 768, -1);  mul_470 = None
        getitem_352: "f32[8, 196, 768]" = split_58[0]
        getitem_353: "f32[8, 196, 768]" = split_58[1];  split_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_294: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_353, memory_format = torch.contiguous_format);  getitem_353 = None
        var_mean_118 = torch.ops.aten.var_mean.correction(clone_294, [2], correction = 0, keepdim = True)
        getitem_354: "f32[8, 196, 1]" = var_mean_118[0]
        getitem_355: "f32[8, 196, 1]" = var_mean_118[1];  var_mean_118 = None
        sub_118: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_294, getitem_355);  clone_294 = getitem_355 = None
        add_411: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_354, 1e-05);  getitem_354 = None
        rsqrt_118: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
        mul_471: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = rsqrt_118 = None
        mul_472: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_471, arg287_1);  mul_471 = arg287_1 = None
        add_412: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_472, arg288_1);  mul_472 = arg288_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_294: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_412, [0, 2, 1]);  add_412 = None
        clone_295: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        view_352: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_295, [6144, 196]);  clone_295 = None
        permute_295: "f32[196, 196]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        mm_58: "f32[6144, 196]" = torch.ops.aten.mm.default(view_352, permute_295);  view_352 = permute_295 = None
        view_353: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_58, [8, 768, 196]);  mm_58 = None
        add_413: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_353, arg290_1);  view_353 = arg290_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_296: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_413, [0, 2, 1]);  add_413 = None
        mul_473: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_352, permute_296);  getitem_352 = permute_296 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_354: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_473, [1568, 768]);  mul_473 = None
        permute_297: "f32[768, 256]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[1568, 256]" = torch.ops.aten.mm.default(view_354, permute_297);  view_354 = permute_297 = None
        add_tensor_2: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default_2, arg292_1);  mm_default_2 = arg292_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_355: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 196, 256]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_414: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_407, view_355);  add_407 = view_355 = None
        clone_297: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_414, memory_format = torch.contiguous_format)
        var_mean_119 = torch.ops.aten.var_mean.correction(clone_297, [2], correction = 0, keepdim = True)
        getitem_356: "f32[8, 196, 1]" = var_mean_119[0]
        getitem_357: "f32[8, 196, 1]" = var_mean_119[1];  var_mean_119 = None
        sub_119: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_297, getitem_357);  clone_297 = getitem_357 = None
        add_415: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_356, 1e-06);  getitem_356 = None
        rsqrt_119: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
        mul_474: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = rsqrt_119 = None
        mul_475: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_474, arg293_1);  mul_474 = arg293_1 = None
        add_416: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_475, arg294_1);  mul_475 = arg294_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_356: "f32[1568, 256]" = torch.ops.aten.reshape.default(add_416, [1568, 256]);  add_416 = None
        permute_298: "f32[256, 1536]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_356, permute_298);  view_356 = permute_298 = None
        add_tensor_1: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_1, arg296_1);  mm_default_1 = arg296_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:183 in forward, code: x = self.fc1(x)
        view_357: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 196, 1536]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:184 in forward, code: x = self.act(x)
        mul_476: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_357, 0.5)
        mul_477: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
        erf_59: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_477);  mul_477 = None
        add_417: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_478: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_476, add_417);  mul_476 = add_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:147 in forward, code: u, v = x.chunk(2, dim=-1)
        split_59 = torch.ops.aten.split.Tensor(mul_478, 768, -1);  mul_478 = None
        getitem_358: "f32[8, 196, 768]" = split_59[0]
        getitem_359: "f32[8, 196, 768]" = split_59[1];  split_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:148 in forward, code: v = self.norm(v)
        clone_299: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_359, memory_format = torch.contiguous_format);  getitem_359 = None
        var_mean_120 = torch.ops.aten.var_mean.correction(clone_299, [2], correction = 0, keepdim = True)
        getitem_360: "f32[8, 196, 1]" = var_mean_120[0]
        getitem_361: "f32[8, 196, 1]" = var_mean_120[1];  var_mean_120 = None
        sub_120: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_299, getitem_361);  clone_299 = getitem_361 = None
        add_418: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_360, 1e-05);  getitem_360 = None
        rsqrt_120: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        mul_479: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = rsqrt_120 = None
        mul_480: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_479, arg297_1);  mul_479 = arg297_1 = None
        add_419: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_480, arg298_1);  mul_480 = arg298_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:149 in forward, code: v = self.proj(v.transpose(-1, -2))
        permute_299: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_419, [0, 2, 1]);  add_419 = None
        clone_300: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
        view_358: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_300, [6144, 196]);  clone_300 = None
        permute_300: "f32[196, 196]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        mm_59: "f32[6144, 196]" = torch.ops.aten.mm.default(view_358, permute_300);  view_358 = permute_300 = None
        view_359: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_59, [8, 768, 196]);  mm_59 = None
        add_420: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_359, arg300_1);  view_359 = arg300_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:150 in forward, code: return u * v.transpose(-1, -2)
        permute_301: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_420, [0, 2, 1]);  add_420 = None
        mul_481: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_358, permute_301);  getitem_358 = permute_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_360: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_481, [1568, 768]);  mul_481 = None
        permute_302: "f32[768, 256]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[1568, 256]" = torch.ops.aten.mm.default(view_360, permute_302);  view_360 = permute_302 = None
        add_tensor: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mm_default, arg302_1);  mm_default = arg302_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:188 in forward, code: x = self.fc2(x)
        view_361: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_tensor, [8, 196, 256]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:177 in forward, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        add_421: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_414, view_361);  add_414 = view_361 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:341 in forward_features, code: x = self.norm(x)
        clone_302: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_421, memory_format = torch.contiguous_format);  add_421 = None
        var_mean_121 = torch.ops.aten.var_mean.correction(clone_302, [2], correction = 0, keepdim = True)
        getitem_362: "f32[8, 196, 1]" = var_mean_121[0]
        getitem_363: "f32[8, 196, 1]" = var_mean_121[1];  var_mean_121 = None
        sub_121: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_302, getitem_363);  clone_302 = getitem_363 = None
        add_422: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_362, 1e-06);  getitem_362 = None
        rsqrt_121: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
        mul_482: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_121);  sub_121 = rsqrt_121 = None
        mul_483: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_482, arg303_1);  mul_482 = arg303_1 = None
        add_423: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_483, arg304_1);  mul_483 = arg304_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:346 in forward_head, code: x = x.mean(dim=1)
        mean_1: "f32[8, 256]" = torch.ops.aten.mean.dim(add_423, [1]);  add_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mlp_mixer.py:348 in forward_head, code: return x if pre_logits else self.head(x)
        permute_303: "f32[256, 1000]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_121: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg306_1, mean_1, permute_303);  arg306_1 = mean_1 = permute_303 = None
        return (addmm_121,)
        