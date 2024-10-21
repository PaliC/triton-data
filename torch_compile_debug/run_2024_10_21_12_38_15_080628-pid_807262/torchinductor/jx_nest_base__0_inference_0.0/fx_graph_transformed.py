class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[128, 3, 4, 4]", arg2_1: "f32[128]", arg3_1: "f32[1, 16, 196, 128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[384, 128]", arg7_1: "f32[384]", arg8_1: "f32[128, 128]", arg9_1: "f32[128]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[512, 128]", arg13_1: "f32[512]", arg14_1: "f32[128, 512]", arg15_1: "f32[128]", arg16_1: "f32[128]", arg17_1: "f32[128]", arg18_1: "f32[384, 128]", arg19_1: "f32[384]", arg20_1: "f32[128, 128]", arg21_1: "f32[128]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[512, 128]", arg25_1: "f32[512]", arg26_1: "f32[128, 512]", arg27_1: "f32[128]", arg28_1: "f32[256, 128, 3, 3]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[256]", arg32_1: "f32[1, 4, 196, 256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[768, 256]", arg36_1: "f32[768]", arg37_1: "f32[256, 256]", arg38_1: "f32[256]", arg39_1: "f32[256]", arg40_1: "f32[256]", arg41_1: "f32[1024, 256]", arg42_1: "f32[1024]", arg43_1: "f32[256, 1024]", arg44_1: "f32[256]", arg45_1: "f32[256]", arg46_1: "f32[256]", arg47_1: "f32[768, 256]", arg48_1: "f32[768]", arg49_1: "f32[256, 256]", arg50_1: "f32[256]", arg51_1: "f32[256]", arg52_1: "f32[256]", arg53_1: "f32[1024, 256]", arg54_1: "f32[1024]", arg55_1: "f32[256, 1024]", arg56_1: "f32[256]", arg57_1: "f32[512, 256, 3, 3]", arg58_1: "f32[512]", arg59_1: "f32[512]", arg60_1: "f32[512]", arg61_1: "f32[1, 1, 196, 512]", arg62_1: "f32[512]", arg63_1: "f32[512]", arg64_1: "f32[1536, 512]", arg65_1: "f32[1536]", arg66_1: "f32[512, 512]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[2048, 512]", arg71_1: "f32[2048]", arg72_1: "f32[512, 2048]", arg73_1: "f32[512]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[1536, 512]", arg77_1: "f32[1536]", arg78_1: "f32[512, 512]", arg79_1: "f32[512]", arg80_1: "f32[512]", arg81_1: "f32[512]", arg82_1: "f32[2048, 512]", arg83_1: "f32[2048]", arg84_1: "f32[512, 2048]", arg85_1: "f32[512]", arg86_1: "f32[512]", arg87_1: "f32[512]", arg88_1: "f32[1536, 512]", arg89_1: "f32[1536]", arg90_1: "f32[512, 512]", arg91_1: "f32[512]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[2048, 512]", arg95_1: "f32[2048]", arg96_1: "f32[512, 2048]", arg97_1: "f32[512]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[1536, 512]", arg101_1: "f32[1536]", arg102_1: "f32[512, 512]", arg103_1: "f32[512]", arg104_1: "f32[512]", arg105_1: "f32[512]", arg106_1: "f32[2048, 512]", arg107_1: "f32[2048]", arg108_1: "f32[512, 2048]", arg109_1: "f32[512]", arg110_1: "f32[512]", arg111_1: "f32[512]", arg112_1: "f32[1536, 512]", arg113_1: "f32[1536]", arg114_1: "f32[512, 512]", arg115_1: "f32[512]", arg116_1: "f32[512]", arg117_1: "f32[512]", arg118_1: "f32[2048, 512]", arg119_1: "f32[2048]", arg120_1: "f32[512, 2048]", arg121_1: "f32[512]", arg122_1: "f32[512]", arg123_1: "f32[512]", arg124_1: "f32[1536, 512]", arg125_1: "f32[1536]", arg126_1: "f32[512, 512]", arg127_1: "f32[512]", arg128_1: "f32[512]", arg129_1: "f32[512]", arg130_1: "f32[2048, 512]", arg131_1: "f32[2048]", arg132_1: "f32[512, 2048]", arg133_1: "f32[512]", arg134_1: "f32[512]", arg135_1: "f32[512]", arg136_1: "f32[1536, 512]", arg137_1: "f32[1536]", arg138_1: "f32[512, 512]", arg139_1: "f32[512]", arg140_1: "f32[512]", arg141_1: "f32[512]", arg142_1: "f32[2048, 512]", arg143_1: "f32[2048]", arg144_1: "f32[512, 2048]", arg145_1: "f32[512]", arg146_1: "f32[512]", arg147_1: "f32[512]", arg148_1: "f32[1536, 512]", arg149_1: "f32[1536]", arg150_1: "f32[512, 512]", arg151_1: "f32[512]", arg152_1: "f32[512]", arg153_1: "f32[512]", arg154_1: "f32[2048, 512]", arg155_1: "f32[2048]", arg156_1: "f32[512, 2048]", arg157_1: "f32[512]", arg158_1: "f32[512]", arg159_1: "f32[512]", arg160_1: "f32[1536, 512]", arg161_1: "f32[1536]", arg162_1: "f32[512, 512]", arg163_1: "f32[512]", arg164_1: "f32[512]", arg165_1: "f32[512]", arg166_1: "f32[2048, 512]", arg167_1: "f32[2048]", arg168_1: "f32[512, 2048]", arg169_1: "f32[512]", arg170_1: "f32[512]", arg171_1: "f32[512]", arg172_1: "f32[1536, 512]", arg173_1: "f32[1536]", arg174_1: "f32[512, 512]", arg175_1: "f32[512]", arg176_1: "f32[512]", arg177_1: "f32[512]", arg178_1: "f32[2048, 512]", arg179_1: "f32[2048]", arg180_1: "f32[512, 2048]", arg181_1: "f32[512]", arg182_1: "f32[512]", arg183_1: "f32[512]", arg184_1: "f32[1536, 512]", arg185_1: "f32[1536]", arg186_1: "f32[512, 512]", arg187_1: "f32[512]", arg188_1: "f32[512]", arg189_1: "f32[512]", arg190_1: "f32[2048, 512]", arg191_1: "f32[2048]", arg192_1: "f32[512, 2048]", arg193_1: "f32[512]", arg194_1: "f32[512]", arg195_1: "f32[512]", arg196_1: "f32[1536, 512]", arg197_1: "f32[1536]", arg198_1: "f32[512, 512]", arg199_1: "f32[512]", arg200_1: "f32[512]", arg201_1: "f32[512]", arg202_1: "f32[2048, 512]", arg203_1: "f32[2048]", arg204_1: "f32[512, 2048]", arg205_1: "f32[512]", arg206_1: "f32[512]", arg207_1: "f32[512]", arg208_1: "f32[1536, 512]", arg209_1: "f32[1536]", arg210_1: "f32[512, 512]", arg211_1: "f32[512]", arg212_1: "f32[512]", arg213_1: "f32[512]", arg214_1: "f32[2048, 512]", arg215_1: "f32[2048]", arg216_1: "f32[512, 2048]", arg217_1: "f32[512]", arg218_1: "f32[512]", arg219_1: "f32[512]", arg220_1: "f32[1536, 512]", arg221_1: "f32[1536]", arg222_1: "f32[512, 512]", arg223_1: "f32[512]", arg224_1: "f32[512]", arg225_1: "f32[512]", arg226_1: "f32[2048, 512]", arg227_1: "f32[2048]", arg228_1: "f32[512, 2048]", arg229_1: "f32[512]", arg230_1: "f32[512]", arg231_1: "f32[512]", arg232_1: "f32[1536, 512]", arg233_1: "f32[1536]", arg234_1: "f32[512, 512]", arg235_1: "f32[512]", arg236_1: "f32[512]", arg237_1: "f32[512]", arg238_1: "f32[2048, 512]", arg239_1: "f32[2048]", arg240_1: "f32[512, 2048]", arg241_1: "f32[512]", arg242_1: "f32[512]", arg243_1: "f32[512]", arg244_1: "f32[1536, 512]", arg245_1: "f32[1536]", arg246_1: "f32[512, 512]", arg247_1: "f32[512]", arg248_1: "f32[512]", arg249_1: "f32[512]", arg250_1: "f32[2048, 512]", arg251_1: "f32[2048]", arg252_1: "f32[512, 2048]", arg253_1: "f32[512]", arg254_1: "f32[512]", arg255_1: "f32[512]", arg256_1: "f32[1536, 512]", arg257_1: "f32[1536]", arg258_1: "f32[512, 512]", arg259_1: "f32[512]", arg260_1: "f32[512]", arg261_1: "f32[512]", arg262_1: "f32[2048, 512]", arg263_1: "f32[2048]", arg264_1: "f32[512, 2048]", arg265_1: "f32[512]", arg266_1: "f32[512]", arg267_1: "f32[512]", arg268_1: "f32[1536, 512]", arg269_1: "f32[1536]", arg270_1: "f32[512, 512]", arg271_1: "f32[512]", arg272_1: "f32[512]", arg273_1: "f32[512]", arg274_1: "f32[2048, 512]", arg275_1: "f32[2048]", arg276_1: "f32[512, 2048]", arg277_1: "f32[512]", arg278_1: "f32[512]", arg279_1: "f32[512]", arg280_1: "f32[1536, 512]", arg281_1: "f32[1536]", arg282_1: "f32[512, 512]", arg283_1: "f32[512]", arg284_1: "f32[512]", arg285_1: "f32[512]", arg286_1: "f32[2048, 512]", arg287_1: "f32[2048]", arg288_1: "f32[512, 2048]", arg289_1: "f32[512]", arg290_1: "f32[512]", arg291_1: "f32[512]", arg292_1: "f32[1536, 512]", arg293_1: "f32[1536]", arg294_1: "f32[512, 512]", arg295_1: "f32[512]", arg296_1: "f32[512]", arg297_1: "f32[512]", arg298_1: "f32[2048, 512]", arg299_1: "f32[2048]", arg300_1: "f32[512, 2048]", arg301_1: "f32[512]", arg302_1: "f32[512]", arg303_1: "f32[512]", arg304_1: "f32[1000, 512]", arg305_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_3: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:233 in forward, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
        permute_187: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:159 in blockify, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
        view_397: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.reshape.default(permute_187, [8, 4, 14, 4, 14, 128]);  permute_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:160 in blockify, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
        permute_188: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.permute.default(view_397, [0, 1, 3, 2, 4, 5]);  view_397 = None
        clone_173: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_398: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(clone_173, [8, 16, 196, 128]);  clone_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:235 in forward, code: x = x + self.pos_embed
        add_177: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(view_398, arg3_1);  view_398 = arg3_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_51 = torch.ops.aten.var_mean.correction(add_177, [3], correction = 0, keepdim = True)
        getitem_178: "f32[8, 16, 196, 1]" = var_mean_51[0]
        getitem_179: "f32[8, 16, 196, 1]" = var_mean_51[1];  var_mean_51 = None
        sub_75: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_177, getitem_179);  getitem_179 = None
        add_178: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_51: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_222: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_51);  sub_75 = rsqrt_51 = None
        mul_223: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_222, arg4_1);  mul_222 = arg4_1 = None
        add_179: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_223, arg5_1);  mul_223 = arg5_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_399: "f32[25088, 128]" = torch.ops.aten.reshape.default(add_179, [25088, 128]);  add_179 = None
        permute_189: "f32[128, 384]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        
        # No stacktrace found for following nodes
        mm_default_95: "f32[25088, 384]" = torch.ops.aten.mm.default(view_399, permute_189);  view_399 = permute_189 = None
        add_tensor_95: "f32[25088, 384]" = torch.ops.aten.add.Tensor(mm_default_95, arg7_1);  mm_default_95 = arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_400: "f32[8, 16, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_95, [8, 16, 196, 384]);  add_tensor_95 = None
        view_401: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.reshape.default(view_400, [8, 16, 196, 3, 4, 32]);  view_400 = None
        permute_190: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_401, [3, 0, 4, 1, 2, 5]);  view_401 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_24 = torch.ops.aten.unbind.int(permute_190);  permute_190 = None
        getitem_180: "f32[8, 4, 16, 196, 32]" = unbind_24[0]
        getitem_181: "f32[8, 4, 16, 196, 32]" = unbind_24[1]
        getitem_182: "f32[8, 4, 16, 196, 32]" = unbind_24[2];  unbind_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_224: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_180, 0.42044820762685725);  getitem_180 = None
        expand_96: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(mul_224, [8, 4, 16, 196, 32]);  mul_224 = None
        clone_174: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_402: "f32[512, 196, 32]" = torch.ops.aten.reshape.default(clone_174, [512, 196, 32]);  clone_174 = None
        permute_191: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.permute.default(getitem_181, [0, 1, 2, 4, 3]);  getitem_181 = None
        mul_225: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(permute_191, 0.42044820762685725);  permute_191 = None
        expand_97: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.expand.default(mul_225, [8, 4, 16, 32, 196]);  mul_225 = None
        clone_175: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_403: "f32[512, 32, 196]" = torch.ops.aten.reshape.default(clone_175, [512, 32, 196]);  clone_175 = None
        bmm_48: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_402, view_403);  view_402 = view_403 = None
        view_404: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_48, [8, 4, 16, 196, 196]);  bmm_48 = None
        eq_24: "b8[8, 4, 16, 196, 196]" = torch.ops.aten.eq.Scalar(view_404, -inf)
        logical_not_48: "b8[8, 4, 16, 196, 196]" = torch.ops.aten.logical_not.default(eq_24);  eq_24 = None
        any_25: "b8[8, 4, 16, 196, 1]" = torch.ops.aten.any.dim(logical_not_48, -1, True);  logical_not_48 = None
        logical_not_49: "b8[8, 4, 16, 196, 1]" = torch.ops.aten.logical_not.default(any_25);  any_25 = None
        full_default: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.full.default([8, 4, 16, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_24: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.amax.default(view_404, [-1], True)
        sub_76: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(view_404, amax_24);  view_404 = amax_24 = None
        exp_24: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_25: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        where_24: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.where.self(logical_not_49, full_default, div_24);  logical_not_49 = full_default = div_24 = None
        expand_98: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.expand.default(where_24, [8, 4, 16, 196, 196]);  where_24 = None
        view_405: "f32[512, 196, 196]" = torch.ops.aten.reshape.default(expand_98, [512, 196, 196]);  expand_98 = None
        expand_99: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(getitem_182, [8, 4, 16, 196, 32]);  getitem_182 = None
        clone_176: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_406: "f32[512, 196, 32]" = torch.ops.aten.reshape.default(clone_176, [512, 196, 32]);  clone_176 = None
        bmm_49: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_405, view_406);  view_405 = view_406 = None
        view_407: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.reshape.default(bmm_49, [8, 4, 16, 196, 32]);  bmm_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_192: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.permute.default(view_407, [0, 2, 3, 4, 1]);  view_407 = None
        clone_177: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_408: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(clone_177, [8, 16, 196, 128]);  clone_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_409: "f32[25088, 128]" = torch.ops.aten.reshape.default(view_408, [25088, 128]);  view_408 = None
        permute_193: "f32[128, 128]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        
        # No stacktrace found for following nodes
        mm_default_94: "f32[25088, 128]" = torch.ops.aten.mm.default(view_409, permute_193);  view_409 = permute_193 = None
        add_tensor_94: "f32[25088, 128]" = torch.ops.aten.add.Tensor(mm_default_94, arg9_1);  mm_default_94 = arg9_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_410: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(add_tensor_94, [8, 16, 196, 128]);  add_tensor_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_180: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_177, view_410);  add_177 = view_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_52 = torch.ops.aten.var_mean.correction(add_180, [3], correction = 0, keepdim = True)
        getitem_183: "f32[8, 16, 196, 1]" = var_mean_52[0]
        getitem_184: "f32[8, 16, 196, 1]" = var_mean_52[1];  var_mean_52 = None
        sub_77: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_180, getitem_184);  getitem_184 = None
        add_181: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_183, 1e-06);  getitem_183 = None
        rsqrt_52: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_226: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_52);  sub_77 = rsqrt_52 = None
        mul_227: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_226, arg10_1);  mul_226 = arg10_1 = None
        add_182: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_227, arg11_1);  mul_227 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_411: "f32[25088, 128]" = torch.ops.aten.reshape.default(add_182, [25088, 128]);  add_182 = None
        permute_194: "f32[128, 512]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        
        # No stacktrace found for following nodes
        mm_default_93: "f32[25088, 512]" = torch.ops.aten.mm.default(view_411, permute_194);  view_411 = permute_194 = None
        add_tensor_93: "f32[25088, 512]" = torch.ops.aten.add.Tensor(mm_default_93, arg13_1);  mm_default_93 = arg13_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_412: "f32[8, 16, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_93, [8, 16, 196, 512]);  add_tensor_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_228: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_412, 0.5)
        mul_229: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_412, 0.7071067811865476);  view_412 = None
        erf_24: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
        add_183: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_230: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_228, add_183);  mul_228 = add_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_413: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_230, [25088, 512]);  mul_230 = None
        permute_195: "f32[512, 128]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        
        # No stacktrace found for following nodes
        mm_default_92: "f32[25088, 128]" = torch.ops.aten.mm.default(view_413, permute_195);  view_413 = permute_195 = None
        add_tensor_92: "f32[25088, 128]" = torch.ops.aten.add.Tensor(mm_default_92, arg15_1);  mm_default_92 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_414: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(add_tensor_92, [8, 16, 196, 128]);  add_tensor_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_184: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_180, view_414);  add_180 = view_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_53 = torch.ops.aten.var_mean.correction(add_184, [3], correction = 0, keepdim = True)
        getitem_185: "f32[8, 16, 196, 1]" = var_mean_53[0]
        getitem_186: "f32[8, 16, 196, 1]" = var_mean_53[1];  var_mean_53 = None
        sub_78: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_184, getitem_186);  getitem_186 = None
        add_185: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_185, 1e-06);  getitem_185 = None
        rsqrt_53: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        mul_231: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_53);  sub_78 = rsqrt_53 = None
        mul_232: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_231, arg16_1);  mul_231 = arg16_1 = None
        add_186: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_232, arg17_1);  mul_232 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_415: "f32[25088, 128]" = torch.ops.aten.reshape.default(add_186, [25088, 128]);  add_186 = None
        permute_196: "f32[128, 384]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        
        # No stacktrace found for following nodes
        mm_default_91: "f32[25088, 384]" = torch.ops.aten.mm.default(view_415, permute_196);  view_415 = permute_196 = None
        add_tensor_91: "f32[25088, 384]" = torch.ops.aten.add.Tensor(mm_default_91, arg19_1);  mm_default_91 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_416: "f32[8, 16, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_91, [8, 16, 196, 384]);  add_tensor_91 = None
        view_417: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.reshape.default(view_416, [8, 16, 196, 3, 4, 32]);  view_416 = None
        permute_197: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_417, [3, 0, 4, 1, 2, 5]);  view_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_25 = torch.ops.aten.unbind.int(permute_197);  permute_197 = None
        getitem_187: "f32[8, 4, 16, 196, 32]" = unbind_25[0]
        getitem_188: "f32[8, 4, 16, 196, 32]" = unbind_25[1]
        getitem_189: "f32[8, 4, 16, 196, 32]" = unbind_25[2];  unbind_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_233: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_187, 0.42044820762685725);  getitem_187 = None
        expand_100: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(mul_233, [8, 4, 16, 196, 32]);  mul_233 = None
        clone_181: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_418: "f32[512, 196, 32]" = torch.ops.aten.reshape.default(clone_181, [512, 196, 32]);  clone_181 = None
        permute_198: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.permute.default(getitem_188, [0, 1, 2, 4, 3]);  getitem_188 = None
        mul_234: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(permute_198, 0.42044820762685725);  permute_198 = None
        expand_101: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.expand.default(mul_234, [8, 4, 16, 32, 196]);  mul_234 = None
        clone_182: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_419: "f32[512, 32, 196]" = torch.ops.aten.reshape.default(clone_182, [512, 32, 196]);  clone_182 = None
        bmm_50: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_418, view_419);  view_418 = view_419 = None
        view_420: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_50, [8, 4, 16, 196, 196]);  bmm_50 = None
        eq_25: "b8[8, 4, 16, 196, 196]" = torch.ops.aten.eq.Scalar(view_420, -inf)
        logical_not_50: "b8[8, 4, 16, 196, 196]" = torch.ops.aten.logical_not.default(eq_25);  eq_25 = None
        any_26: "b8[8, 4, 16, 196, 1]" = torch.ops.aten.any.dim(logical_not_50, -1, True);  logical_not_50 = None
        logical_not_51: "b8[8, 4, 16, 196, 1]" = torch.ops.aten.logical_not.default(any_26);  any_26 = None
        full_default_1: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.full.default([8, 4, 16, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_25: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.amax.default(view_420, [-1], True)
        sub_79: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(view_420, amax_25);  view_420 = amax_25 = None
        exp_25: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        sum_26: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_25: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        where_25: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.where.self(logical_not_51, full_default_1, div_25);  logical_not_51 = full_default_1 = div_25 = None
        expand_102: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.expand.default(where_25, [8, 4, 16, 196, 196]);  where_25 = None
        view_421: "f32[512, 196, 196]" = torch.ops.aten.reshape.default(expand_102, [512, 196, 196]);  expand_102 = None
        expand_103: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.expand.default(getitem_189, [8, 4, 16, 196, 32]);  getitem_189 = None
        clone_183: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_422: "f32[512, 196, 32]" = torch.ops.aten.reshape.default(clone_183, [512, 196, 32]);  clone_183 = None
        bmm_51: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_421, view_422);  view_421 = view_422 = None
        view_423: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.reshape.default(bmm_51, [8, 4, 16, 196, 32]);  bmm_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_199: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.permute.default(view_423, [0, 2, 3, 4, 1]);  view_423 = None
        clone_184: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_424: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(clone_184, [8, 16, 196, 128]);  clone_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_425: "f32[25088, 128]" = torch.ops.aten.reshape.default(view_424, [25088, 128]);  view_424 = None
        permute_200: "f32[128, 128]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        
        # No stacktrace found for following nodes
        mm_default_90: "f32[25088, 128]" = torch.ops.aten.mm.default(view_425, permute_200);  view_425 = permute_200 = None
        add_tensor_90: "f32[25088, 128]" = torch.ops.aten.add.Tensor(mm_default_90, arg21_1);  mm_default_90 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_426: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(add_tensor_90, [8, 16, 196, 128]);  add_tensor_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_187: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_184, view_426);  add_184 = view_426 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_54 = torch.ops.aten.var_mean.correction(add_187, [3], correction = 0, keepdim = True)
        getitem_190: "f32[8, 16, 196, 1]" = var_mean_54[0]
        getitem_191: "f32[8, 16, 196, 1]" = var_mean_54[1];  var_mean_54 = None
        sub_80: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(add_187, getitem_191);  getitem_191 = None
        add_188: "f32[8, 16, 196, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
        rsqrt_54: "f32[8, 16, 196, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_235: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_54);  sub_80 = rsqrt_54 = None
        mul_236: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_235, arg22_1);  mul_235 = arg22_1 = None
        add_189: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(mul_236, arg23_1);  mul_236 = arg23_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_427: "f32[25088, 128]" = torch.ops.aten.reshape.default(add_189, [25088, 128]);  add_189 = None
        permute_201: "f32[128, 512]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        
        # No stacktrace found for following nodes
        mm_default_89: "f32[25088, 512]" = torch.ops.aten.mm.default(view_427, permute_201);  view_427 = permute_201 = None
        add_tensor_89: "f32[25088, 512]" = torch.ops.aten.add.Tensor(mm_default_89, arg25_1);  mm_default_89 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_428: "f32[8, 16, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_89, [8, 16, 196, 512]);  add_tensor_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_237: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_428, 0.5)
        mul_238: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_428, 0.7071067811865476);  view_428 = None
        erf_25: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_190: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_239: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_237, add_190);  mul_237 = add_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_429: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_239, [25088, 512]);  mul_239 = None
        permute_202: "f32[512, 128]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        
        # No stacktrace found for following nodes
        mm_default_88: "f32[25088, 128]" = torch.ops.aten.mm.default(view_429, permute_202);  view_429 = permute_202 = None
        add_tensor_88: "f32[25088, 128]" = torch.ops.aten.add.Tensor(mm_default_88, arg27_1);  mm_default_88 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_430: "f32[8, 16, 196, 128]" = torch.ops.aten.reshape.default(add_tensor_88, [8, 16, 196, 128]);  add_tensor_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_191: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_187, view_430);  add_187 = view_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:174 in deblockify, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
        view_431: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.reshape.default(add_191, [8, 4, 4, 14, 14, 128]);  add_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:175 in deblockify, code: x = x.transpose(2, 3).reshape(B, height, width, C)
        permute_203: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.permute.default(view_431, [0, 1, 3, 2, 4, 5]);  view_431 = None
        clone_188: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_432: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_188, [8, 56, 56, 128]);  clone_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:242 in forward, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
        permute_204: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_432, [0, 3, 1, 2]);  view_432 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:141 in forward, code: x = self.conv(x)
        convolution_4: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(permute_204, arg28_1, arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_204 = arg28_1 = arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:143 in forward, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_205: "f32[8, 56, 56, 256]" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_55 = torch.ops.aten.var_mean.correction(permute_205, [3], correction = 0, keepdim = True)
        getitem_192: "f32[8, 56, 56, 1]" = var_mean_55[0]
        getitem_193: "f32[8, 56, 56, 1]" = var_mean_55[1];  var_mean_55 = None
        sub_81: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(permute_205, getitem_193);  permute_205 = getitem_193 = None
        add_192: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_55: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_240: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_55);  sub_81 = rsqrt_55 = None
        mul_241: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_240, arg30_1);  mul_240 = arg30_1 = None
        add_193: "f32[8, 56, 56, 256]" = torch.ops.aten.add.Tensor(mul_241, arg31_1);  mul_241 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:143 in forward, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_206: "f32[8, 256, 56, 56]" = torch.ops.aten.permute.default(add_193, [0, 3, 1, 2]);  add_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f32[8, 256, 57, 57]" = torch.ops.aten.constant_pad_nd.default(permute_206, [0, 1, 0, 1], -inf);  permute_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/pool2d_same.py:53 in forward, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
        _low_memory_max_pool2d_with_offsets_2 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_2, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_2 = None
        getitem_194: "f32[8, 256, 28, 28]" = _low_memory_max_pool2d_with_offsets_2[0];  _low_memory_max_pool2d_with_offsets_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:233 in forward, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
        permute_207: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 3, 1]);  getitem_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:159 in blockify, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
        view_433: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.reshape.default(permute_207, [8, 2, 14, 2, 14, 256]);  permute_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:160 in blockify, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
        permute_208: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.permute.default(view_433, [0, 1, 3, 2, 4, 5]);  view_433 = None
        clone_189: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
        view_434: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(clone_189, [8, 4, 196, 256]);  clone_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:235 in forward, code: x = x + self.pos_embed
        add_194: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(view_434, arg32_1);  view_434 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_56 = torch.ops.aten.var_mean.correction(add_194, [3], correction = 0, keepdim = True)
        getitem_196: "f32[8, 4, 196, 1]" = var_mean_56[0]
        getitem_197: "f32[8, 4, 196, 1]" = var_mean_56[1];  var_mean_56 = None
        sub_82: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_194, getitem_197);  getitem_197 = None
        add_195: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-06);  getitem_196 = None
        rsqrt_56: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_242: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_56);  sub_82 = rsqrt_56 = None
        mul_243: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_242, arg33_1);  mul_242 = arg33_1 = None
        add_196: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_243, arg34_1);  mul_243 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_435: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_196, [6272, 256]);  add_196 = None
        permute_209: "f32[256, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        
        # No stacktrace found for following nodes
        mm_default_87: "f32[6272, 768]" = torch.ops.aten.mm.default(view_435, permute_209);  view_435 = permute_209 = None
        add_tensor_87: "f32[6272, 768]" = torch.ops.aten.add.Tensor(mm_default_87, arg36_1);  mm_default_87 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_436: "f32[8, 4, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_87, [8, 4, 196, 768]);  add_tensor_87 = None
        view_437: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.reshape.default(view_436, [8, 4, 196, 3, 8, 32]);  view_436 = None
        permute_210: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_437, [3, 0, 4, 1, 2, 5]);  view_437 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_26 = torch.ops.aten.unbind.int(permute_210);  permute_210 = None
        getitem_198: "f32[8, 8, 4, 196, 32]" = unbind_26[0]
        getitem_199: "f32[8, 8, 4, 196, 32]" = unbind_26[1]
        getitem_200: "f32[8, 8, 4, 196, 32]" = unbind_26[2];  unbind_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_244: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_198, 0.42044820762685725);  getitem_198 = None
        expand_104: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(mul_244, [8, 8, 4, 196, 32]);  mul_244 = None
        clone_190: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_438: "f32[256, 196, 32]" = torch.ops.aten.reshape.default(clone_190, [256, 196, 32]);  clone_190 = None
        permute_211: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.permute.default(getitem_199, [0, 1, 2, 4, 3]);  getitem_199 = None
        mul_245: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(permute_211, 0.42044820762685725);  permute_211 = None
        expand_105: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.expand.default(mul_245, [8, 8, 4, 32, 196]);  mul_245 = None
        clone_191: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_439: "f32[256, 32, 196]" = torch.ops.aten.reshape.default(clone_191, [256, 32, 196]);  clone_191 = None
        bmm_52: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_438, view_439);  view_438 = view_439 = None
        view_440: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.reshape.default(bmm_52, [8, 8, 4, 196, 196]);  bmm_52 = None
        eq_26: "b8[8, 8, 4, 196, 196]" = torch.ops.aten.eq.Scalar(view_440, -inf)
        logical_not_52: "b8[8, 8, 4, 196, 196]" = torch.ops.aten.logical_not.default(eq_26);  eq_26 = None
        any_27: "b8[8, 8, 4, 196, 1]" = torch.ops.aten.any.dim(logical_not_52, -1, True);  logical_not_52 = None
        logical_not_53: "b8[8, 8, 4, 196, 1]" = torch.ops.aten.logical_not.default(any_27);  any_27 = None
        full_default_2: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.full.default([8, 8, 4, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_26: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.amax.default(view_440, [-1], True)
        sub_83: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(view_440, amax_26);  view_440 = amax_26 = None
        exp_26: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
        sum_27: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_26: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        where_26: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.where.self(logical_not_53, full_default_2, div_26);  logical_not_53 = full_default_2 = div_26 = None
        expand_106: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.expand.default(where_26, [8, 8, 4, 196, 196]);  where_26 = None
        view_441: "f32[256, 196, 196]" = torch.ops.aten.reshape.default(expand_106, [256, 196, 196]);  expand_106 = None
        expand_107: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(getitem_200, [8, 8, 4, 196, 32]);  getitem_200 = None
        clone_192: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_442: "f32[256, 196, 32]" = torch.ops.aten.reshape.default(clone_192, [256, 196, 32]);  clone_192 = None
        bmm_53: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_441, view_442);  view_441 = view_442 = None
        view_443: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.reshape.default(bmm_53, [8, 8, 4, 196, 32]);  bmm_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_212: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.permute.default(view_443, [0, 2, 3, 4, 1]);  view_443 = None
        clone_193: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_444: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(clone_193, [8, 4, 196, 256]);  clone_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_445: "f32[6272, 256]" = torch.ops.aten.reshape.default(view_444, [6272, 256]);  view_444 = None
        permute_213: "f32[256, 256]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        
        # No stacktrace found for following nodes
        mm_default_86: "f32[6272, 256]" = torch.ops.aten.mm.default(view_445, permute_213);  view_445 = permute_213 = None
        add_tensor_86: "f32[6272, 256]" = torch.ops.aten.add.Tensor(mm_default_86, arg38_1);  mm_default_86 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_446: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_86, [8, 4, 196, 256]);  add_tensor_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_197: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_194, view_446);  add_194 = view_446 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_57 = torch.ops.aten.var_mean.correction(add_197, [3], correction = 0, keepdim = True)
        getitem_201: "f32[8, 4, 196, 1]" = var_mean_57[0]
        getitem_202: "f32[8, 4, 196, 1]" = var_mean_57[1];  var_mean_57 = None
        sub_84: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_197, getitem_202);  getitem_202 = None
        add_198: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_201, 1e-06);  getitem_201 = None
        rsqrt_57: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        mul_246: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_57);  sub_84 = rsqrt_57 = None
        mul_247: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_246, arg39_1);  mul_246 = arg39_1 = None
        add_199: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_247, arg40_1);  mul_247 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_447: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_199, [6272, 256]);  add_199 = None
        permute_214: "f32[256, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        
        # No stacktrace found for following nodes
        mm_default_85: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_447, permute_214);  view_447 = permute_214 = None
        add_tensor_85: "f32[6272, 1024]" = torch.ops.aten.add.Tensor(mm_default_85, arg42_1);  mm_default_85 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_448: "f32[8, 4, 196, 1024]" = torch.ops.aten.reshape.default(add_tensor_85, [8, 4, 196, 1024]);  add_tensor_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_248: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_448, 0.5)
        mul_249: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_448, 0.7071067811865476);  view_448 = None
        erf_26: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_249);  mul_249 = None
        add_200: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_250: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_248, add_200);  mul_248 = add_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_449: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_250, [6272, 1024]);  mul_250 = None
        permute_215: "f32[1024, 256]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        
        # No stacktrace found for following nodes
        mm_default_84: "f32[6272, 256]" = torch.ops.aten.mm.default(view_449, permute_215);  view_449 = permute_215 = None
        add_tensor_84: "f32[6272, 256]" = torch.ops.aten.add.Tensor(mm_default_84, arg44_1);  mm_default_84 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_450: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_84, [8, 4, 196, 256]);  add_tensor_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_201: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_197, view_450);  add_197 = view_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_58 = torch.ops.aten.var_mean.correction(add_201, [3], correction = 0, keepdim = True)
        getitem_203: "f32[8, 4, 196, 1]" = var_mean_58[0]
        getitem_204: "f32[8, 4, 196, 1]" = var_mean_58[1];  var_mean_58 = None
        sub_85: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_201, getitem_204);  getitem_204 = None
        add_202: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_203, 1e-06);  getitem_203 = None
        rsqrt_58: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        mul_251: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_58);  sub_85 = rsqrt_58 = None
        mul_252: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_251, arg45_1);  mul_251 = arg45_1 = None
        add_203: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_252, arg46_1);  mul_252 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_451: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_203, [6272, 256]);  add_203 = None
        permute_216: "f32[256, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        
        # No stacktrace found for following nodes
        mm_default_83: "f32[6272, 768]" = torch.ops.aten.mm.default(view_451, permute_216);  view_451 = permute_216 = None
        add_tensor_83: "f32[6272, 768]" = torch.ops.aten.add.Tensor(mm_default_83, arg48_1);  mm_default_83 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_452: "f32[8, 4, 196, 768]" = torch.ops.aten.reshape.default(add_tensor_83, [8, 4, 196, 768]);  add_tensor_83 = None
        view_453: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.reshape.default(view_452, [8, 4, 196, 3, 8, 32]);  view_452 = None
        permute_217: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_453, [3, 0, 4, 1, 2, 5]);  view_453 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_27 = torch.ops.aten.unbind.int(permute_217);  permute_217 = None
        getitem_205: "f32[8, 8, 4, 196, 32]" = unbind_27[0]
        getitem_206: "f32[8, 8, 4, 196, 32]" = unbind_27[1]
        getitem_207: "f32[8, 8, 4, 196, 32]" = unbind_27[2];  unbind_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_253: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_205, 0.42044820762685725);  getitem_205 = None
        expand_108: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(mul_253, [8, 8, 4, 196, 32]);  mul_253 = None
        clone_197: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_454: "f32[256, 196, 32]" = torch.ops.aten.reshape.default(clone_197, [256, 196, 32]);  clone_197 = None
        permute_218: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.permute.default(getitem_206, [0, 1, 2, 4, 3]);  getitem_206 = None
        mul_254: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(permute_218, 0.42044820762685725);  permute_218 = None
        expand_109: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.expand.default(mul_254, [8, 8, 4, 32, 196]);  mul_254 = None
        clone_198: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_455: "f32[256, 32, 196]" = torch.ops.aten.reshape.default(clone_198, [256, 32, 196]);  clone_198 = None
        bmm_54: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_454, view_455);  view_454 = view_455 = None
        view_456: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.reshape.default(bmm_54, [8, 8, 4, 196, 196]);  bmm_54 = None
        eq_27: "b8[8, 8, 4, 196, 196]" = torch.ops.aten.eq.Scalar(view_456, -inf)
        logical_not_54: "b8[8, 8, 4, 196, 196]" = torch.ops.aten.logical_not.default(eq_27);  eq_27 = None
        any_28: "b8[8, 8, 4, 196, 1]" = torch.ops.aten.any.dim(logical_not_54, -1, True);  logical_not_54 = None
        logical_not_55: "b8[8, 8, 4, 196, 1]" = torch.ops.aten.logical_not.default(any_28);  any_28 = None
        full_default_3: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.full.default([8, 8, 4, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_27: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.amax.default(view_456, [-1], True)
        sub_86: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(view_456, amax_27);  view_456 = amax_27 = None
        exp_27: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_86);  sub_86 = None
        sum_28: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_27: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        where_27: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.where.self(logical_not_55, full_default_3, div_27);  logical_not_55 = full_default_3 = div_27 = None
        expand_110: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.expand.default(where_27, [8, 8, 4, 196, 196]);  where_27 = None
        view_457: "f32[256, 196, 196]" = torch.ops.aten.reshape.default(expand_110, [256, 196, 196]);  expand_110 = None
        expand_111: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.expand.default(getitem_207, [8, 8, 4, 196, 32]);  getitem_207 = None
        clone_199: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_458: "f32[256, 196, 32]" = torch.ops.aten.reshape.default(clone_199, [256, 196, 32]);  clone_199 = None
        bmm_55: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_457, view_458);  view_457 = view_458 = None
        view_459: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.reshape.default(bmm_55, [8, 8, 4, 196, 32]);  bmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_219: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.permute.default(view_459, [0, 2, 3, 4, 1]);  view_459 = None
        clone_200: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_460: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(clone_200, [8, 4, 196, 256]);  clone_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_461: "f32[6272, 256]" = torch.ops.aten.reshape.default(view_460, [6272, 256]);  view_460 = None
        permute_220: "f32[256, 256]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        
        # No stacktrace found for following nodes
        mm_default_82: "f32[6272, 256]" = torch.ops.aten.mm.default(view_461, permute_220);  view_461 = permute_220 = None
        add_tensor_82: "f32[6272, 256]" = torch.ops.aten.add.Tensor(mm_default_82, arg50_1);  mm_default_82 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_462: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_82, [8, 4, 196, 256]);  add_tensor_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_204: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_201, view_462);  add_201 = view_462 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_59 = torch.ops.aten.var_mean.correction(add_204, [3], correction = 0, keepdim = True)
        getitem_208: "f32[8, 4, 196, 1]" = var_mean_59[0]
        getitem_209: "f32[8, 4, 196, 1]" = var_mean_59[1];  var_mean_59 = None
        sub_87: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(add_204, getitem_209);  getitem_209 = None
        add_205: "f32[8, 4, 196, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-06);  getitem_208 = None
        rsqrt_59: "f32[8, 4, 196, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        mul_255: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_59);  sub_87 = rsqrt_59 = None
        mul_256: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_255, arg51_1);  mul_255 = arg51_1 = None
        add_206: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(mul_256, arg52_1);  mul_256 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_463: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_206, [6272, 256]);  add_206 = None
        permute_221: "f32[256, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        
        # No stacktrace found for following nodes
        mm_default_81: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_463, permute_221);  view_463 = permute_221 = None
        add_tensor_81: "f32[6272, 1024]" = torch.ops.aten.add.Tensor(mm_default_81, arg54_1);  mm_default_81 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_464: "f32[8, 4, 196, 1024]" = torch.ops.aten.reshape.default(add_tensor_81, [8, 4, 196, 1024]);  add_tensor_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_257: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_464, 0.5)
        mul_258: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_464, 0.7071067811865476);  view_464 = None
        erf_27: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_258);  mul_258 = None
        add_207: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_259: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_257, add_207);  mul_257 = add_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_465: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_259, [6272, 1024]);  mul_259 = None
        permute_222: "f32[1024, 256]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        
        # No stacktrace found for following nodes
        mm_default_80: "f32[6272, 256]" = torch.ops.aten.mm.default(view_465, permute_222);  view_465 = permute_222 = None
        add_tensor_80: "f32[6272, 256]" = torch.ops.aten.add.Tensor(mm_default_80, arg56_1);  mm_default_80 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_466: "f32[8, 4, 196, 256]" = torch.ops.aten.reshape.default(add_tensor_80, [8, 4, 196, 256]);  add_tensor_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_208: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_204, view_466);  add_204 = view_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:174 in deblockify, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
        view_467: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.reshape.default(add_208, [8, 2, 2, 14, 14, 256]);  add_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:175 in deblockify, code: x = x.transpose(2, 3).reshape(B, height, width, C)
        permute_223: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.permute.default(view_467, [0, 1, 3, 2, 4, 5]);  view_467 = None
        clone_204: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_468: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_204, [8, 28, 28, 256]);  clone_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:242 in forward, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
        permute_224: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_468, [0, 3, 1, 2]);  view_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:141 in forward, code: x = self.conv(x)
        convolution_5: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_224, arg57_1, arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_224 = arg57_1 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:143 in forward, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_225: "f32[8, 28, 28, 512]" = torch.ops.aten.permute.default(convolution_5, [0, 2, 3, 1]);  convolution_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_60 = torch.ops.aten.var_mean.correction(permute_225, [3], correction = 0, keepdim = True)
        getitem_210: "f32[8, 28, 28, 1]" = var_mean_60[0]
        getitem_211: "f32[8, 28, 28, 1]" = var_mean_60[1];  var_mean_60 = None
        sub_88: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(permute_225, getitem_211);  permute_225 = getitem_211 = None
        add_209: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-06);  getitem_210 = None
        rsqrt_60: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        mul_260: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_60);  sub_88 = rsqrt_60 = None
        mul_261: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_260, arg59_1);  mul_260 = arg59_1 = None
        add_210: "f32[8, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_261, arg60_1);  mul_261 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:143 in forward, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_226: "f32[8, 512, 28, 28]" = torch.ops.aten.permute.default(add_210, [0, 3, 1, 2]);  add_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "f32[8, 512, 29, 29]" = torch.ops.aten.constant_pad_nd.default(permute_226, [0, 1, 0, 1], -inf);  permute_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/pool2d_same.py:53 in forward, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
        _low_memory_max_pool2d_with_offsets_3 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(constant_pad_nd_3, [3, 3], [2, 2], [0, 0], [1, 1], False);  constant_pad_nd_3 = None
        getitem_212: "f32[8, 512, 14, 14]" = _low_memory_max_pool2d_with_offsets_3[0];  _low_memory_max_pool2d_with_offsets_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:233 in forward, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
        permute_227: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(getitem_212, [0, 2, 3, 1]);  getitem_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:159 in blockify, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
        view_469: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.reshape.default(permute_227, [8, 1, 14, 1, 14, 512]);  permute_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:160 in blockify, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
        permute_228: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.permute.default(view_469, [0, 1, 3, 2, 4, 5]);  view_469 = None
        view_470: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(permute_228, [8, 1, -1, 512]);  permute_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:235 in forward, code: x = x + self.pos_embed
        add_211: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(view_470, arg61_1);  view_470 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_61 = torch.ops.aten.var_mean.correction(add_211, [3], correction = 0, keepdim = True)
        getitem_214: "f32[8, 1, 196, 1]" = var_mean_61[0]
        getitem_215: "f32[8, 1, 196, 1]" = var_mean_61[1];  var_mean_61 = None
        sub_89: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_211, getitem_215);  getitem_215 = None
        add_212: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
        rsqrt_61: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        mul_262: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_61);  sub_89 = rsqrt_61 = None
        mul_263: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_262, arg62_1);  mul_262 = arg62_1 = None
        add_213: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_263, arg63_1);  mul_263 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_471: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_213, [1568, 512]);  add_213 = None
        permute_229: "f32[512, 1536]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        
        # No stacktrace found for following nodes
        mm_default_79: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_471, permute_229);  view_471 = permute_229 = None
        add_tensor_79: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_79, arg65_1);  mm_default_79 = arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_472: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_79, [8, 1, 196, 1536]);  add_tensor_79 = None
        view_473: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_472, [8, 1, 196, 3, 16, 32]);  view_472 = None
        permute_230: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_473, [3, 0, 4, 1, 2, 5]);  view_473 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_28 = torch.ops.aten.unbind.int(permute_230);  permute_230 = None
        getitem_216: "f32[8, 16, 1, 196, 32]" = unbind_28[0]
        getitem_217: "f32[8, 16, 1, 196, 32]" = unbind_28[1]
        getitem_218: "f32[8, 16, 1, 196, 32]" = unbind_28[2];  unbind_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_264: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_216, 0.42044820762685725);  getitem_216 = None
        expand_112: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_264, [8, 16, 1, 196, 32]);  mul_264 = None
        clone_205: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
        view_474: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_205, [128, 196, 32]);  clone_205 = None
        permute_231: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_217, [0, 1, 2, 4, 3]);  getitem_217 = None
        mul_265: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_231, 0.42044820762685725);  permute_231 = None
        expand_113: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_265, [8, 16, 1, 32, 196]);  mul_265 = None
        clone_206: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_475: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_206, [128, 32, 196]);  clone_206 = None
        bmm_56: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = view_475 = None
        view_476: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_56, [8, 16, 1, 196, 196]);  bmm_56 = None
        eq_28: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_476, -inf)
        logical_not_56: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_28);  eq_28 = None
        any_29: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_56, -1, True);  logical_not_56 = None
        logical_not_57: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_29);  any_29 = None
        full_default_4: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_28: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_476, [-1], True)
        sub_90: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_476, amax_28);  view_476 = amax_28 = None
        exp_28: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_90);  sub_90 = None
        sum_29: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_28: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
        where_28: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_57, full_default_4, div_28);  logical_not_57 = full_default_4 = div_28 = None
        expand_114: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_28, [8, 16, 1, 196, 196]);  where_28 = None
        view_477: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_114, [128, 196, 196]);  expand_114 = None
        expand_115: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_218, [8, 16, 1, 196, 32]);  getitem_218 = None
        clone_207: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_478: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_207, [128, 196, 32]);  clone_207 = None
        bmm_57: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_477, view_478);  view_477 = view_478 = None
        view_479: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_57, [8, 16, 1, 196, 32]);  bmm_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_232: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_479, [0, 2, 3, 4, 1]);  view_479 = None
        clone_208: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_480: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_208, [8, 1, 196, 512]);  clone_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_481: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_480, [1568, 512]);  view_480 = None
        permute_233: "f32[512, 512]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        
        # No stacktrace found for following nodes
        mm_default_78: "f32[1568, 512]" = torch.ops.aten.mm.default(view_481, permute_233);  view_481 = permute_233 = None
        add_tensor_78: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_78, arg67_1);  mm_default_78 = arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_482: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_78, [8, 1, 196, 512]);  add_tensor_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_214: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_211, view_482);  add_211 = view_482 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_62 = torch.ops.aten.var_mean.correction(add_214, [3], correction = 0, keepdim = True)
        getitem_219: "f32[8, 1, 196, 1]" = var_mean_62[0]
        getitem_220: "f32[8, 1, 196, 1]" = var_mean_62[1];  var_mean_62 = None
        sub_91: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_214, getitem_220);  getitem_220 = None
        add_215: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_219, 1e-06);  getitem_219 = None
        rsqrt_62: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        mul_266: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_62);  sub_91 = rsqrt_62 = None
        mul_267: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_266, arg68_1);  mul_266 = arg68_1 = None
        add_216: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_267, arg69_1);  mul_267 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_483: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_216, [1568, 512]);  add_216 = None
        permute_234: "f32[512, 2048]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        
        # No stacktrace found for following nodes
        mm_default_77: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_483, permute_234);  view_483 = permute_234 = None
        add_tensor_77: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_77, arg71_1);  mm_default_77 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_484: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_77, [8, 1, 196, 2048]);  add_tensor_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_268: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_484, 0.5)
        mul_269: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_484, 0.7071067811865476);  view_484 = None
        erf_28: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_269);  mul_269 = None
        add_217: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_270: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_268, add_217);  mul_268 = add_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_485: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_270, [1568, 2048]);  mul_270 = None
        permute_235: "f32[2048, 512]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        
        # No stacktrace found for following nodes
        mm_default_76: "f32[1568, 512]" = torch.ops.aten.mm.default(view_485, permute_235);  view_485 = permute_235 = None
        add_tensor_76: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_76, arg73_1);  mm_default_76 = arg73_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_486: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_76, [8, 1, 196, 512]);  add_tensor_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_218: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_214, view_486);  add_214 = view_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_63 = torch.ops.aten.var_mean.correction(add_218, [3], correction = 0, keepdim = True)
        getitem_221: "f32[8, 1, 196, 1]" = var_mean_63[0]
        getitem_222: "f32[8, 1, 196, 1]" = var_mean_63[1];  var_mean_63 = None
        sub_92: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_218, getitem_222);  getitem_222 = None
        add_219: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_221, 1e-06);  getitem_221 = None
        rsqrt_63: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_271: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_63);  sub_92 = rsqrt_63 = None
        mul_272: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_271, arg74_1);  mul_271 = arg74_1 = None
        add_220: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_272, arg75_1);  mul_272 = arg75_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_487: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_220, [1568, 512]);  add_220 = None
        permute_236: "f32[512, 1536]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        
        # No stacktrace found for following nodes
        mm_default_75: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_487, permute_236);  view_487 = permute_236 = None
        add_tensor_75: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_75, arg77_1);  mm_default_75 = arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_488: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_75, [8, 1, 196, 1536]);  add_tensor_75 = None
        view_489: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_488, [8, 1, 196, 3, 16, 32]);  view_488 = None
        permute_237: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_489, [3, 0, 4, 1, 2, 5]);  view_489 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_29 = torch.ops.aten.unbind.int(permute_237);  permute_237 = None
        getitem_223: "f32[8, 16, 1, 196, 32]" = unbind_29[0]
        getitem_224: "f32[8, 16, 1, 196, 32]" = unbind_29[1]
        getitem_225: "f32[8, 16, 1, 196, 32]" = unbind_29[2];  unbind_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_273: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_223, 0.42044820762685725);  getitem_223 = None
        expand_116: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_273, [8, 16, 1, 196, 32]);  mul_273 = None
        clone_212: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_490: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_212, [128, 196, 32]);  clone_212 = None
        permute_238: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_224, [0, 1, 2, 4, 3]);  getitem_224 = None
        mul_274: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_238, 0.42044820762685725);  permute_238 = None
        expand_117: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_274, [8, 16, 1, 32, 196]);  mul_274 = None
        clone_213: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_491: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_213, [128, 32, 196]);  clone_213 = None
        bmm_58: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_490, view_491);  view_490 = view_491 = None
        view_492: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_58, [8, 16, 1, 196, 196]);  bmm_58 = None
        eq_29: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_492, -inf)
        logical_not_58: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_29);  eq_29 = None
        any_30: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_58, -1, True);  logical_not_58 = None
        logical_not_59: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_30);  any_30 = None
        full_default_5: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_29: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_492, [-1], True)
        sub_93: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_492, amax_29);  view_492 = amax_29 = None
        exp_29: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_30: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_29: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
        where_29: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_59, full_default_5, div_29);  logical_not_59 = full_default_5 = div_29 = None
        expand_118: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_29, [8, 16, 1, 196, 196]);  where_29 = None
        view_493: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_118, [128, 196, 196]);  expand_118 = None
        expand_119: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_225, [8, 16, 1, 196, 32]);  getitem_225 = None
        clone_214: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
        view_494: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_214, [128, 196, 32]);  clone_214 = None
        bmm_59: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_493, view_494);  view_493 = view_494 = None
        view_495: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_59, [8, 16, 1, 196, 32]);  bmm_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_239: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_495, [0, 2, 3, 4, 1]);  view_495 = None
        clone_215: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_496: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_215, [8, 1, 196, 512]);  clone_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_497: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_496, [1568, 512]);  view_496 = None
        permute_240: "f32[512, 512]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        
        # No stacktrace found for following nodes
        mm_default_74: "f32[1568, 512]" = torch.ops.aten.mm.default(view_497, permute_240);  view_497 = permute_240 = None
        add_tensor_74: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_74, arg79_1);  mm_default_74 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_498: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_74, [8, 1, 196, 512]);  add_tensor_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_221: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_218, view_498);  add_218 = view_498 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_64 = torch.ops.aten.var_mean.correction(add_221, [3], correction = 0, keepdim = True)
        getitem_226: "f32[8, 1, 196, 1]" = var_mean_64[0]
        getitem_227: "f32[8, 1, 196, 1]" = var_mean_64[1];  var_mean_64 = None
        sub_94: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_221, getitem_227);  getitem_227 = None
        add_222: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
        rsqrt_64: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        mul_275: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_64);  sub_94 = rsqrt_64 = None
        mul_276: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_275, arg80_1);  mul_275 = arg80_1 = None
        add_223: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_276, arg81_1);  mul_276 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_499: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_223, [1568, 512]);  add_223 = None
        permute_241: "f32[512, 2048]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        
        # No stacktrace found for following nodes
        mm_default_73: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_499, permute_241);  view_499 = permute_241 = None
        add_tensor_73: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_73, arg83_1);  mm_default_73 = arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_500: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_73, [8, 1, 196, 2048]);  add_tensor_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_277: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_500, 0.5)
        mul_278: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_500, 0.7071067811865476);  view_500 = None
        erf_29: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_278);  mul_278 = None
        add_224: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_279: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_277, add_224);  mul_277 = add_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_501: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_279, [1568, 2048]);  mul_279 = None
        permute_242: "f32[2048, 512]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        
        # No stacktrace found for following nodes
        mm_default_72: "f32[1568, 512]" = torch.ops.aten.mm.default(view_501, permute_242);  view_501 = permute_242 = None
        add_tensor_72: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_72, arg85_1);  mm_default_72 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_502: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_72, [8, 1, 196, 512]);  add_tensor_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_225: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_221, view_502);  add_221 = view_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_65 = torch.ops.aten.var_mean.correction(add_225, [3], correction = 0, keepdim = True)
        getitem_228: "f32[8, 1, 196, 1]" = var_mean_65[0]
        getitem_229: "f32[8, 1, 196, 1]" = var_mean_65[1];  var_mean_65 = None
        sub_95: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_225, getitem_229);  getitem_229 = None
        add_226: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-06);  getitem_228 = None
        rsqrt_65: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        mul_280: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_65);  sub_95 = rsqrt_65 = None
        mul_281: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_280, arg86_1);  mul_280 = arg86_1 = None
        add_227: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_281, arg87_1);  mul_281 = arg87_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_503: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_227, [1568, 512]);  add_227 = None
        permute_243: "f32[512, 1536]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        
        # No stacktrace found for following nodes
        mm_default_71: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_503, permute_243);  view_503 = permute_243 = None
        add_tensor_71: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_71, arg89_1);  mm_default_71 = arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_504: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_71, [8, 1, 196, 1536]);  add_tensor_71 = None
        view_505: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_504, [8, 1, 196, 3, 16, 32]);  view_504 = None
        permute_244: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_505, [3, 0, 4, 1, 2, 5]);  view_505 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_30 = torch.ops.aten.unbind.int(permute_244);  permute_244 = None
        getitem_230: "f32[8, 16, 1, 196, 32]" = unbind_30[0]
        getitem_231: "f32[8, 16, 1, 196, 32]" = unbind_30[1]
        getitem_232: "f32[8, 16, 1, 196, 32]" = unbind_30[2];  unbind_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_282: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_230, 0.42044820762685725);  getitem_230 = None
        expand_120: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_282, [8, 16, 1, 196, 32]);  mul_282 = None
        clone_219: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
        view_506: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_219, [128, 196, 32]);  clone_219 = None
        permute_245: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_231, [0, 1, 2, 4, 3]);  getitem_231 = None
        mul_283: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_245, 0.42044820762685725);  permute_245 = None
        expand_121: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_283, [8, 16, 1, 32, 196]);  mul_283 = None
        clone_220: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_507: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_220, [128, 32, 196]);  clone_220 = None
        bmm_60: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_506, view_507);  view_506 = view_507 = None
        view_508: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_60, [8, 16, 1, 196, 196]);  bmm_60 = None
        eq_30: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_508, -inf)
        logical_not_60: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_30);  eq_30 = None
        any_31: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_60, -1, True);  logical_not_60 = None
        logical_not_61: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_31);  any_31 = None
        full_default_6: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_30: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_508, [-1], True)
        sub_96: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_508, amax_30);  view_508 = amax_30 = None
        exp_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_96);  sub_96 = None
        sum_31: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
        where_30: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_61, full_default_6, div_30);  logical_not_61 = full_default_6 = div_30 = None
        expand_122: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_30, [8, 16, 1, 196, 196]);  where_30 = None
        view_509: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_122, [128, 196, 196]);  expand_122 = None
        expand_123: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_232, [8, 16, 1, 196, 32]);  getitem_232 = None
        clone_221: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_510: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_221, [128, 196, 32]);  clone_221 = None
        bmm_61: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_509, view_510);  view_509 = view_510 = None
        view_511: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_61, [8, 16, 1, 196, 32]);  bmm_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_246: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_511, [0, 2, 3, 4, 1]);  view_511 = None
        clone_222: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_512: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_222, [8, 1, 196, 512]);  clone_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_513: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_512, [1568, 512]);  view_512 = None
        permute_247: "f32[512, 512]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        
        # No stacktrace found for following nodes
        mm_default_70: "f32[1568, 512]" = torch.ops.aten.mm.default(view_513, permute_247);  view_513 = permute_247 = None
        add_tensor_70: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_70, arg91_1);  mm_default_70 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_514: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_70, [8, 1, 196, 512]);  add_tensor_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_228: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_225, view_514);  add_225 = view_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_66 = torch.ops.aten.var_mean.correction(add_228, [3], correction = 0, keepdim = True)
        getitem_233: "f32[8, 1, 196, 1]" = var_mean_66[0]
        getitem_234: "f32[8, 1, 196, 1]" = var_mean_66[1];  var_mean_66 = None
        sub_97: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_228, getitem_234);  getitem_234 = None
        add_229: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_233, 1e-06);  getitem_233 = None
        rsqrt_66: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_284: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_66);  sub_97 = rsqrt_66 = None
        mul_285: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_284, arg92_1);  mul_284 = arg92_1 = None
        add_230: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_285, arg93_1);  mul_285 = arg93_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_515: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_230, [1568, 512]);  add_230 = None
        permute_248: "f32[512, 2048]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        
        # No stacktrace found for following nodes
        mm_default_69: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_515, permute_248);  view_515 = permute_248 = None
        add_tensor_69: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_69, arg95_1);  mm_default_69 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_516: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_69, [8, 1, 196, 2048]);  add_tensor_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_286: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, 0.5)
        mul_287: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, 0.7071067811865476);  view_516 = None
        erf_30: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_287);  mul_287 = None
        add_231: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_288: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_286, add_231);  mul_286 = add_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_517: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_288, [1568, 2048]);  mul_288 = None
        permute_249: "f32[2048, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        
        # No stacktrace found for following nodes
        mm_default_68: "f32[1568, 512]" = torch.ops.aten.mm.default(view_517, permute_249);  view_517 = permute_249 = None
        add_tensor_68: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_68, arg97_1);  mm_default_68 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_518: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_68, [8, 1, 196, 512]);  add_tensor_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_232: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_228, view_518);  add_228 = view_518 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_67 = torch.ops.aten.var_mean.correction(add_232, [3], correction = 0, keepdim = True)
        getitem_235: "f32[8, 1, 196, 1]" = var_mean_67[0]
        getitem_236: "f32[8, 1, 196, 1]" = var_mean_67[1];  var_mean_67 = None
        sub_98: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_232, getitem_236);  getitem_236 = None
        add_233: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_235, 1e-06);  getitem_235 = None
        rsqrt_67: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        mul_289: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_67);  sub_98 = rsqrt_67 = None
        mul_290: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_289, arg98_1);  mul_289 = arg98_1 = None
        add_234: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_290, arg99_1);  mul_290 = arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_519: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_234, [1568, 512]);  add_234 = None
        permute_250: "f32[512, 1536]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        
        # No stacktrace found for following nodes
        mm_default_67: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_519, permute_250);  view_519 = permute_250 = None
        add_tensor_67: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_67, arg101_1);  mm_default_67 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_520: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_67, [8, 1, 196, 1536]);  add_tensor_67 = None
        view_521: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_520, [8, 1, 196, 3, 16, 32]);  view_520 = None
        permute_251: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_521, [3, 0, 4, 1, 2, 5]);  view_521 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_31 = torch.ops.aten.unbind.int(permute_251);  permute_251 = None
        getitem_237: "f32[8, 16, 1, 196, 32]" = unbind_31[0]
        getitem_238: "f32[8, 16, 1, 196, 32]" = unbind_31[1]
        getitem_239: "f32[8, 16, 1, 196, 32]" = unbind_31[2];  unbind_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_291: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_237, 0.42044820762685725);  getitem_237 = None
        expand_124: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_291, [8, 16, 1, 196, 32]);  mul_291 = None
        clone_226: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
        view_522: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_226, [128, 196, 32]);  clone_226 = None
        permute_252: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_238, [0, 1, 2, 4, 3]);  getitem_238 = None
        mul_292: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_252, 0.42044820762685725);  permute_252 = None
        expand_125: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_292, [8, 16, 1, 32, 196]);  mul_292 = None
        clone_227: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_523: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_227, [128, 32, 196]);  clone_227 = None
        bmm_62: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_522, view_523);  view_522 = view_523 = None
        view_524: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_62, [8, 16, 1, 196, 196]);  bmm_62 = None
        eq_31: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_524, -inf)
        logical_not_62: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_31);  eq_31 = None
        any_32: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_62, -1, True);  logical_not_62 = None
        logical_not_63: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_32);  any_32 = None
        full_default_7: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_31: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_524, [-1], True)
        sub_99: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_524, amax_31);  view_524 = amax_31 = None
        exp_31: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_99);  sub_99 = None
        sum_32: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_31: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
        where_31: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_63, full_default_7, div_31);  logical_not_63 = full_default_7 = div_31 = None
        expand_126: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_31, [8, 16, 1, 196, 196]);  where_31 = None
        view_525: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_126, [128, 196, 196]);  expand_126 = None
        expand_127: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_239, [8, 16, 1, 196, 32]);  getitem_239 = None
        clone_228: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_526: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_228, [128, 196, 32]);  clone_228 = None
        bmm_63: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_525, view_526);  view_525 = view_526 = None
        view_527: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_63, [8, 16, 1, 196, 32]);  bmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_253: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_527, [0, 2, 3, 4, 1]);  view_527 = None
        clone_229: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_528: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_229, [8, 1, 196, 512]);  clone_229 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_529: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_528, [1568, 512]);  view_528 = None
        permute_254: "f32[512, 512]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        
        # No stacktrace found for following nodes
        mm_default_66: "f32[1568, 512]" = torch.ops.aten.mm.default(view_529, permute_254);  view_529 = permute_254 = None
        add_tensor_66: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_66, arg103_1);  mm_default_66 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_530: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_66, [8, 1, 196, 512]);  add_tensor_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_235: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_232, view_530);  add_232 = view_530 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_68 = torch.ops.aten.var_mean.correction(add_235, [3], correction = 0, keepdim = True)
        getitem_240: "f32[8, 1, 196, 1]" = var_mean_68[0]
        getitem_241: "f32[8, 1, 196, 1]" = var_mean_68[1];  var_mean_68 = None
        sub_100: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_235, getitem_241);  getitem_241 = None
        add_236: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-06);  getitem_240 = None
        rsqrt_68: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_293: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_68);  sub_100 = rsqrt_68 = None
        mul_294: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_293, arg104_1);  mul_293 = arg104_1 = None
        add_237: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_294, arg105_1);  mul_294 = arg105_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_531: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_237, [1568, 512]);  add_237 = None
        permute_255: "f32[512, 2048]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        
        # No stacktrace found for following nodes
        mm_default_65: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_531, permute_255);  view_531 = permute_255 = None
        add_tensor_65: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_65, arg107_1);  mm_default_65 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_532: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_65, [8, 1, 196, 2048]);  add_tensor_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_295: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_532, 0.5)
        mul_296: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_532, 0.7071067811865476);  view_532 = None
        erf_31: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_296);  mul_296 = None
        add_238: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_297: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_295, add_238);  mul_295 = add_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_533: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_297, [1568, 2048]);  mul_297 = None
        permute_256: "f32[2048, 512]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        
        # No stacktrace found for following nodes
        mm_default_64: "f32[1568, 512]" = torch.ops.aten.mm.default(view_533, permute_256);  view_533 = permute_256 = None
        add_tensor_64: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_64, arg109_1);  mm_default_64 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_534: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_64, [8, 1, 196, 512]);  add_tensor_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_239: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_235, view_534);  add_235 = view_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_69 = torch.ops.aten.var_mean.correction(add_239, [3], correction = 0, keepdim = True)
        getitem_242: "f32[8, 1, 196, 1]" = var_mean_69[0]
        getitem_243: "f32[8, 1, 196, 1]" = var_mean_69[1];  var_mean_69 = None
        sub_101: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_239, getitem_243);  getitem_243 = None
        add_240: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
        rsqrt_69: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        mul_298: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_69);  sub_101 = rsqrt_69 = None
        mul_299: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_298, arg110_1);  mul_298 = arg110_1 = None
        add_241: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_299, arg111_1);  mul_299 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_535: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_241, [1568, 512]);  add_241 = None
        permute_257: "f32[512, 1536]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        
        # No stacktrace found for following nodes
        mm_default_63: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_535, permute_257);  view_535 = permute_257 = None
        add_tensor_63: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_63, arg113_1);  mm_default_63 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_536: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_63, [8, 1, 196, 1536]);  add_tensor_63 = None
        view_537: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_536, [8, 1, 196, 3, 16, 32]);  view_536 = None
        permute_258: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_537, [3, 0, 4, 1, 2, 5]);  view_537 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_32 = torch.ops.aten.unbind.int(permute_258);  permute_258 = None
        getitem_244: "f32[8, 16, 1, 196, 32]" = unbind_32[0]
        getitem_245: "f32[8, 16, 1, 196, 32]" = unbind_32[1]
        getitem_246: "f32[8, 16, 1, 196, 32]" = unbind_32[2];  unbind_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_300: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_244, 0.42044820762685725);  getitem_244 = None
        expand_128: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_300, [8, 16, 1, 196, 32]);  mul_300 = None
        clone_233: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
        view_538: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_233, [128, 196, 32]);  clone_233 = None
        permute_259: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_245, [0, 1, 2, 4, 3]);  getitem_245 = None
        mul_301: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_259, 0.42044820762685725);  permute_259 = None
        expand_129: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_301, [8, 16, 1, 32, 196]);  mul_301 = None
        clone_234: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_539: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_234, [128, 32, 196]);  clone_234 = None
        bmm_64: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_538, view_539);  view_538 = view_539 = None
        view_540: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_64, [8, 16, 1, 196, 196]);  bmm_64 = None
        eq_32: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_540, -inf)
        logical_not_64: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_32);  eq_32 = None
        any_33: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_64, -1, True);  logical_not_64 = None
        logical_not_65: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_33);  any_33 = None
        full_default_8: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_32: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_540, [-1], True)
        sub_102: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_540, amax_32);  view_540 = amax_32 = None
        exp_32: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_102);  sub_102 = None
        sum_33: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_32: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
        where_32: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_65, full_default_8, div_32);  logical_not_65 = full_default_8 = div_32 = None
        expand_130: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_32, [8, 16, 1, 196, 196]);  where_32 = None
        view_541: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_130, [128, 196, 196]);  expand_130 = None
        expand_131: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_246, [8, 16, 1, 196, 32]);  getitem_246 = None
        clone_235: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_542: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_235, [128, 196, 32]);  clone_235 = None
        bmm_65: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_541, view_542);  view_541 = view_542 = None
        view_543: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_65, [8, 16, 1, 196, 32]);  bmm_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_260: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_543, [0, 2, 3, 4, 1]);  view_543 = None
        clone_236: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_544: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_236, [8, 1, 196, 512]);  clone_236 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_545: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_544, [1568, 512]);  view_544 = None
        permute_261: "f32[512, 512]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        
        # No stacktrace found for following nodes
        mm_default_62: "f32[1568, 512]" = torch.ops.aten.mm.default(view_545, permute_261);  view_545 = permute_261 = None
        add_tensor_62: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_62, arg115_1);  mm_default_62 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_546: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_62, [8, 1, 196, 512]);  add_tensor_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_242: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_239, view_546);  add_239 = view_546 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_70 = torch.ops.aten.var_mean.correction(add_242, [3], correction = 0, keepdim = True)
        getitem_247: "f32[8, 1, 196, 1]" = var_mean_70[0]
        getitem_248: "f32[8, 1, 196, 1]" = var_mean_70[1];  var_mean_70 = None
        sub_103: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_242, getitem_248);  getitem_248 = None
        add_243: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_247, 1e-06);  getitem_247 = None
        rsqrt_70: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_302: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_70);  sub_103 = rsqrt_70 = None
        mul_303: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_302, arg116_1);  mul_302 = arg116_1 = None
        add_244: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_303, arg117_1);  mul_303 = arg117_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_547: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_244, [1568, 512]);  add_244 = None
        permute_262: "f32[512, 2048]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        
        # No stacktrace found for following nodes
        mm_default_61: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_547, permute_262);  view_547 = permute_262 = None
        add_tensor_61: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_61, arg119_1);  mm_default_61 = arg119_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_548: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_61, [8, 1, 196, 2048]);  add_tensor_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_304: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_548, 0.5)
        mul_305: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_548, 0.7071067811865476);  view_548 = None
        erf_32: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_305);  mul_305 = None
        add_245: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_306: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_304, add_245);  mul_304 = add_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_549: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_306, [1568, 2048]);  mul_306 = None
        permute_263: "f32[2048, 512]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        
        # No stacktrace found for following nodes
        mm_default_60: "f32[1568, 512]" = torch.ops.aten.mm.default(view_549, permute_263);  view_549 = permute_263 = None
        add_tensor_60: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_60, arg121_1);  mm_default_60 = arg121_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_550: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_60, [8, 1, 196, 512]);  add_tensor_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_246: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_242, view_550);  add_242 = view_550 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_71 = torch.ops.aten.var_mean.correction(add_246, [3], correction = 0, keepdim = True)
        getitem_249: "f32[8, 1, 196, 1]" = var_mean_71[0]
        getitem_250: "f32[8, 1, 196, 1]" = var_mean_71[1];  var_mean_71 = None
        sub_104: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_246, getitem_250);  getitem_250 = None
        add_247: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_249, 1e-06);  getitem_249 = None
        rsqrt_71: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_307: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_71);  sub_104 = rsqrt_71 = None
        mul_308: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_307, arg122_1);  mul_307 = arg122_1 = None
        add_248: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_308, arg123_1);  mul_308 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_551: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_248, [1568, 512]);  add_248 = None
        permute_264: "f32[512, 1536]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        
        # No stacktrace found for following nodes
        mm_default_59: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_551, permute_264);  view_551 = permute_264 = None
        add_tensor_59: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_59, arg125_1);  mm_default_59 = arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_552: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 1, 196, 1536]);  add_tensor_59 = None
        view_553: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_552, [8, 1, 196, 3, 16, 32]);  view_552 = None
        permute_265: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_553, [3, 0, 4, 1, 2, 5]);  view_553 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_33 = torch.ops.aten.unbind.int(permute_265);  permute_265 = None
        getitem_251: "f32[8, 16, 1, 196, 32]" = unbind_33[0]
        getitem_252: "f32[8, 16, 1, 196, 32]" = unbind_33[1]
        getitem_253: "f32[8, 16, 1, 196, 32]" = unbind_33[2];  unbind_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_309: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_251, 0.42044820762685725);  getitem_251 = None
        expand_132: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_309, [8, 16, 1, 196, 32]);  mul_309 = None
        clone_240: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
        view_554: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_240, [128, 196, 32]);  clone_240 = None
        permute_266: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_252, [0, 1, 2, 4, 3]);  getitem_252 = None
        mul_310: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_266, 0.42044820762685725);  permute_266 = None
        expand_133: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_310, [8, 16, 1, 32, 196]);  mul_310 = None
        clone_241: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_555: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_241, [128, 32, 196]);  clone_241 = None
        bmm_66: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_554, view_555);  view_554 = view_555 = None
        view_556: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_66, [8, 16, 1, 196, 196]);  bmm_66 = None
        eq_33: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_556, -inf)
        logical_not_66: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_33);  eq_33 = None
        any_34: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_66, -1, True);  logical_not_66 = None
        logical_not_67: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_34);  any_34 = None
        full_default_9: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_33: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_556, [-1], True)
        sub_105: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_556, amax_33);  view_556 = amax_33 = None
        exp_33: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_105);  sub_105 = None
        sum_34: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_33: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
        where_33: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_67, full_default_9, div_33);  logical_not_67 = full_default_9 = div_33 = None
        expand_134: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_33, [8, 16, 1, 196, 196]);  where_33 = None
        view_557: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_134, [128, 196, 196]);  expand_134 = None
        expand_135: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_253, [8, 16, 1, 196, 32]);  getitem_253 = None
        clone_242: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
        view_558: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_242, [128, 196, 32]);  clone_242 = None
        bmm_67: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_557, view_558);  view_557 = view_558 = None
        view_559: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_67, [8, 16, 1, 196, 32]);  bmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_267: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_559, [0, 2, 3, 4, 1]);  view_559 = None
        clone_243: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_560: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_243, [8, 1, 196, 512]);  clone_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_561: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_560, [1568, 512]);  view_560 = None
        permute_268: "f32[512, 512]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        
        # No stacktrace found for following nodes
        mm_default_58: "f32[1568, 512]" = torch.ops.aten.mm.default(view_561, permute_268);  view_561 = permute_268 = None
        add_tensor_58: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_58, arg127_1);  mm_default_58 = arg127_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_562: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 1, 196, 512]);  add_tensor_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_249: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_246, view_562);  add_246 = view_562 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_72 = torch.ops.aten.var_mean.correction(add_249, [3], correction = 0, keepdim = True)
        getitem_254: "f32[8, 1, 196, 1]" = var_mean_72[0]
        getitem_255: "f32[8, 1, 196, 1]" = var_mean_72[1];  var_mean_72 = None
        sub_106: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_249, getitem_255);  getitem_255 = None
        add_250: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_72: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        mul_311: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_72);  sub_106 = rsqrt_72 = None
        mul_312: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_311, arg128_1);  mul_311 = arg128_1 = None
        add_251: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_312, arg129_1);  mul_312 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_563: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_251, [1568, 512]);  add_251 = None
        permute_269: "f32[512, 2048]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        
        # No stacktrace found for following nodes
        mm_default_57: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_563, permute_269);  view_563 = permute_269 = None
        add_tensor_57: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_57, arg131_1);  mm_default_57 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_564: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_57, [8, 1, 196, 2048]);  add_tensor_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_313: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_564, 0.5)
        mul_314: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_564, 0.7071067811865476);  view_564 = None
        erf_33: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_314);  mul_314 = None
        add_252: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_315: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_313, add_252);  mul_313 = add_252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_565: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_315, [1568, 2048]);  mul_315 = None
        permute_270: "f32[2048, 512]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        
        # No stacktrace found for following nodes
        mm_default_56: "f32[1568, 512]" = torch.ops.aten.mm.default(view_565, permute_270);  view_565 = permute_270 = None
        add_tensor_56: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_56, arg133_1);  mm_default_56 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_566: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_56, [8, 1, 196, 512]);  add_tensor_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_253: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_249, view_566);  add_249 = view_566 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_73 = torch.ops.aten.var_mean.correction(add_253, [3], correction = 0, keepdim = True)
        getitem_256: "f32[8, 1, 196, 1]" = var_mean_73[0]
        getitem_257: "f32[8, 1, 196, 1]" = var_mean_73[1];  var_mean_73 = None
        sub_107: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_253, getitem_257);  getitem_257 = None
        add_254: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-06);  getitem_256 = None
        rsqrt_73: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        mul_316: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_73);  sub_107 = rsqrt_73 = None
        mul_317: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_316, arg134_1);  mul_316 = arg134_1 = None
        add_255: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_317, arg135_1);  mul_317 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_567: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_255, [1568, 512]);  add_255 = None
        permute_271: "f32[512, 1536]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        
        # No stacktrace found for following nodes
        mm_default_55: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_567, permute_271);  view_567 = permute_271 = None
        add_tensor_55: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_55, arg137_1);  mm_default_55 = arg137_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_568: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_55, [8, 1, 196, 1536]);  add_tensor_55 = None
        view_569: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_568, [8, 1, 196, 3, 16, 32]);  view_568 = None
        permute_272: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_569, [3, 0, 4, 1, 2, 5]);  view_569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_34 = torch.ops.aten.unbind.int(permute_272);  permute_272 = None
        getitem_258: "f32[8, 16, 1, 196, 32]" = unbind_34[0]
        getitem_259: "f32[8, 16, 1, 196, 32]" = unbind_34[1]
        getitem_260: "f32[8, 16, 1, 196, 32]" = unbind_34[2];  unbind_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_318: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_258, 0.42044820762685725);  getitem_258 = None
        expand_136: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_318, [8, 16, 1, 196, 32]);  mul_318 = None
        clone_247: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
        view_570: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_247, [128, 196, 32]);  clone_247 = None
        permute_273: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_259, [0, 1, 2, 4, 3]);  getitem_259 = None
        mul_319: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_273, 0.42044820762685725);  permute_273 = None
        expand_137: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_319, [8, 16, 1, 32, 196]);  mul_319 = None
        clone_248: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_571: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_248, [128, 32, 196]);  clone_248 = None
        bmm_68: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_570, view_571);  view_570 = view_571 = None
        view_572: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_68, [8, 16, 1, 196, 196]);  bmm_68 = None
        eq_34: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_572, -inf)
        logical_not_68: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_34);  eq_34 = None
        any_35: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_68, -1, True);  logical_not_68 = None
        logical_not_69: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_35);  any_35 = None
        full_default_10: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_34: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_572, [-1], True)
        sub_108: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_572, amax_34);  view_572 = amax_34 = None
        exp_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_108);  sub_108 = None
        sum_35: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
        where_34: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_69, full_default_10, div_34);  logical_not_69 = full_default_10 = div_34 = None
        expand_138: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_34, [8, 16, 1, 196, 196]);  where_34 = None
        view_573: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_138, [128, 196, 196]);  expand_138 = None
        expand_139: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_260, [8, 16, 1, 196, 32]);  getitem_260 = None
        clone_249: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_574: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_249, [128, 196, 32]);  clone_249 = None
        bmm_69: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_573, view_574);  view_573 = view_574 = None
        view_575: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_69, [8, 16, 1, 196, 32]);  bmm_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_274: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_575, [0, 2, 3, 4, 1]);  view_575 = None
        clone_250: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
        view_576: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_250, [8, 1, 196, 512]);  clone_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_577: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_576, [1568, 512]);  view_576 = None
        permute_275: "f32[512, 512]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        
        # No stacktrace found for following nodes
        mm_default_54: "f32[1568, 512]" = torch.ops.aten.mm.default(view_577, permute_275);  view_577 = permute_275 = None
        add_tensor_54: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_54, arg139_1);  mm_default_54 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_578: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 1, 196, 512]);  add_tensor_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_256: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_253, view_578);  add_253 = view_578 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_74 = torch.ops.aten.var_mean.correction(add_256, [3], correction = 0, keepdim = True)
        getitem_261: "f32[8, 1, 196, 1]" = var_mean_74[0]
        getitem_262: "f32[8, 1, 196, 1]" = var_mean_74[1];  var_mean_74 = None
        sub_109: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_256, getitem_262);  getitem_262 = None
        add_257: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_261, 1e-06);  getitem_261 = None
        rsqrt_74: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        mul_320: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_74);  sub_109 = rsqrt_74 = None
        mul_321: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_320, arg140_1);  mul_320 = arg140_1 = None
        add_258: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_321, arg141_1);  mul_321 = arg141_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_579: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_258, [1568, 512]);  add_258 = None
        permute_276: "f32[512, 2048]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        
        # No stacktrace found for following nodes
        mm_default_53: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_579, permute_276);  view_579 = permute_276 = None
        add_tensor_53: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_53, arg143_1);  mm_default_53 = arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_580: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 1, 196, 2048]);  add_tensor_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_322: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_580, 0.5)
        mul_323: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_580, 0.7071067811865476);  view_580 = None
        erf_34: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_323);  mul_323 = None
        add_259: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_324: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_322, add_259);  mul_322 = add_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_581: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_324, [1568, 2048]);  mul_324 = None
        permute_277: "f32[2048, 512]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        
        # No stacktrace found for following nodes
        mm_default_52: "f32[1568, 512]" = torch.ops.aten.mm.default(view_581, permute_277);  view_581 = permute_277 = None
        add_tensor_52: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_52, arg145_1);  mm_default_52 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_582: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 1, 196, 512]);  add_tensor_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_260: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_256, view_582);  add_256 = view_582 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_75 = torch.ops.aten.var_mean.correction(add_260, [3], correction = 0, keepdim = True)
        getitem_263: "f32[8, 1, 196, 1]" = var_mean_75[0]
        getitem_264: "f32[8, 1, 196, 1]" = var_mean_75[1];  var_mean_75 = None
        sub_110: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_260, getitem_264);  getitem_264 = None
        add_261: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_263, 1e-06);  getitem_263 = None
        rsqrt_75: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        mul_325: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_75);  sub_110 = rsqrt_75 = None
        mul_326: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_325, arg146_1);  mul_325 = arg146_1 = None
        add_262: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_326, arg147_1);  mul_326 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_583: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_262, [1568, 512]);  add_262 = None
        permute_278: "f32[512, 1536]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        
        # No stacktrace found for following nodes
        mm_default_51: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_583, permute_278);  view_583 = permute_278 = None
        add_tensor_51: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_51, arg149_1);  mm_default_51 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_584: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 1, 196, 1536]);  add_tensor_51 = None
        view_585: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_584, [8, 1, 196, 3, 16, 32]);  view_584 = None
        permute_279: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_585, [3, 0, 4, 1, 2, 5]);  view_585 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_35 = torch.ops.aten.unbind.int(permute_279);  permute_279 = None
        getitem_265: "f32[8, 16, 1, 196, 32]" = unbind_35[0]
        getitem_266: "f32[8, 16, 1, 196, 32]" = unbind_35[1]
        getitem_267: "f32[8, 16, 1, 196, 32]" = unbind_35[2];  unbind_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_327: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_265, 0.42044820762685725);  getitem_265 = None
        expand_140: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_327, [8, 16, 1, 196, 32]);  mul_327 = None
        clone_254: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
        view_586: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_254, [128, 196, 32]);  clone_254 = None
        permute_280: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_266, [0, 1, 2, 4, 3]);  getitem_266 = None
        mul_328: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_280, 0.42044820762685725);  permute_280 = None
        expand_141: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_328, [8, 16, 1, 32, 196]);  mul_328 = None
        clone_255: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
        view_587: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_255, [128, 32, 196]);  clone_255 = None
        bmm_70: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_586, view_587);  view_586 = view_587 = None
        view_588: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_70, [8, 16, 1, 196, 196]);  bmm_70 = None
        eq_35: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_588, -inf)
        logical_not_70: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_35);  eq_35 = None
        any_36: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_70, -1, True);  logical_not_70 = None
        logical_not_71: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_36);  any_36 = None
        full_default_11: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_35: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_588, [-1], True)
        sub_111: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_588, amax_35);  view_588 = amax_35 = None
        exp_35: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_111);  sub_111 = None
        sum_36: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_35: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
        where_35: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_71, full_default_11, div_35);  logical_not_71 = full_default_11 = div_35 = None
        expand_142: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_35, [8, 16, 1, 196, 196]);  where_35 = None
        view_589: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_142, [128, 196, 196]);  expand_142 = None
        expand_143: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_267, [8, 16, 1, 196, 32]);  getitem_267 = None
        clone_256: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
        view_590: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_256, [128, 196, 32]);  clone_256 = None
        bmm_71: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_589, view_590);  view_589 = view_590 = None
        view_591: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_71, [8, 16, 1, 196, 32]);  bmm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_281: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_591, [0, 2, 3, 4, 1]);  view_591 = None
        clone_257: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_592: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_257, [8, 1, 196, 512]);  clone_257 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_593: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_592, [1568, 512]);  view_592 = None
        permute_282: "f32[512, 512]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        
        # No stacktrace found for following nodes
        mm_default_50: "f32[1568, 512]" = torch.ops.aten.mm.default(view_593, permute_282);  view_593 = permute_282 = None
        add_tensor_50: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_50, arg151_1);  mm_default_50 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_594: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_50, [8, 1, 196, 512]);  add_tensor_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_263: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_260, view_594);  add_260 = view_594 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_76 = torch.ops.aten.var_mean.correction(add_263, [3], correction = 0, keepdim = True)
        getitem_268: "f32[8, 1, 196, 1]" = var_mean_76[0]
        getitem_269: "f32[8, 1, 196, 1]" = var_mean_76[1];  var_mean_76 = None
        sub_112: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_263, getitem_269);  getitem_269 = None
        add_264: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-06);  getitem_268 = None
        rsqrt_76: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        mul_329: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_76);  sub_112 = rsqrt_76 = None
        mul_330: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_329, arg152_1);  mul_329 = arg152_1 = None
        add_265: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_330, arg153_1);  mul_330 = arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_595: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_265, [1568, 512]);  add_265 = None
        permute_283: "f32[512, 2048]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        
        # No stacktrace found for following nodes
        mm_default_49: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_595, permute_283);  view_595 = permute_283 = None
        add_tensor_49: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_49, arg155_1);  mm_default_49 = arg155_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_596: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_49, [8, 1, 196, 2048]);  add_tensor_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_331: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_596, 0.5)
        mul_332: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_596, 0.7071067811865476);  view_596 = None
        erf_35: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_332);  mul_332 = None
        add_266: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_333: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_331, add_266);  mul_331 = add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_597: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_333, [1568, 2048]);  mul_333 = None
        permute_284: "f32[2048, 512]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        
        # No stacktrace found for following nodes
        mm_default_48: "f32[1568, 512]" = torch.ops.aten.mm.default(view_597, permute_284);  view_597 = permute_284 = None
        add_tensor_48: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_48, arg157_1);  mm_default_48 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_598: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_48, [8, 1, 196, 512]);  add_tensor_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_267: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_263, view_598);  add_263 = view_598 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_77 = torch.ops.aten.var_mean.correction(add_267, [3], correction = 0, keepdim = True)
        getitem_270: "f32[8, 1, 196, 1]" = var_mean_77[0]
        getitem_271: "f32[8, 1, 196, 1]" = var_mean_77[1];  var_mean_77 = None
        sub_113: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_267, getitem_271);  getitem_271 = None
        add_268: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_77: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        mul_334: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_77);  sub_113 = rsqrt_77 = None
        mul_335: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_334, arg158_1);  mul_334 = arg158_1 = None
        add_269: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_335, arg159_1);  mul_335 = arg159_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_599: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_269, [1568, 512]);  add_269 = None
        permute_285: "f32[512, 1536]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        
        # No stacktrace found for following nodes
        mm_default_47: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_599, permute_285);  view_599 = permute_285 = None
        add_tensor_47: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_47, arg161_1);  mm_default_47 = arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_600: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 1, 196, 1536]);  add_tensor_47 = None
        view_601: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_600, [8, 1, 196, 3, 16, 32]);  view_600 = None
        permute_286: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_601, [3, 0, 4, 1, 2, 5]);  view_601 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_36 = torch.ops.aten.unbind.int(permute_286);  permute_286 = None
        getitem_272: "f32[8, 16, 1, 196, 32]" = unbind_36[0]
        getitem_273: "f32[8, 16, 1, 196, 32]" = unbind_36[1]
        getitem_274: "f32[8, 16, 1, 196, 32]" = unbind_36[2];  unbind_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_336: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_272, 0.42044820762685725);  getitem_272 = None
        expand_144: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_336, [8, 16, 1, 196, 32]);  mul_336 = None
        clone_261: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_144, memory_format = torch.contiguous_format);  expand_144 = None
        view_602: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_261, [128, 196, 32]);  clone_261 = None
        permute_287: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_273, [0, 1, 2, 4, 3]);  getitem_273 = None
        mul_337: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_287, 0.42044820762685725);  permute_287 = None
        expand_145: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_337, [8, 16, 1, 32, 196]);  mul_337 = None
        clone_262: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_603: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_262, [128, 32, 196]);  clone_262 = None
        bmm_72: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_602, view_603);  view_602 = view_603 = None
        view_604: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_72, [8, 16, 1, 196, 196]);  bmm_72 = None
        eq_36: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_604, -inf)
        logical_not_72: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_36);  eq_36 = None
        any_37: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_72, -1, True);  logical_not_72 = None
        logical_not_73: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_37);  any_37 = None
        full_default_12: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_36: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_604, [-1], True)
        sub_114: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_604, amax_36);  view_604 = amax_36 = None
        exp_36: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_37: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_36: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
        where_36: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_73, full_default_12, div_36);  logical_not_73 = full_default_12 = div_36 = None
        expand_146: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_36, [8, 16, 1, 196, 196]);  where_36 = None
        view_605: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_146, [128, 196, 196]);  expand_146 = None
        expand_147: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_274, [8, 16, 1, 196, 32]);  getitem_274 = None
        clone_263: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_606: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_263, [128, 196, 32]);  clone_263 = None
        bmm_73: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_605, view_606);  view_605 = view_606 = None
        view_607: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_73, [8, 16, 1, 196, 32]);  bmm_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_288: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_607, [0, 2, 3, 4, 1]);  view_607 = None
        clone_264: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_608: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_264, [8, 1, 196, 512]);  clone_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_609: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_608, [1568, 512]);  view_608 = None
        permute_289: "f32[512, 512]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        
        # No stacktrace found for following nodes
        mm_default_46: "f32[1568, 512]" = torch.ops.aten.mm.default(view_609, permute_289);  view_609 = permute_289 = None
        add_tensor_46: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_46, arg163_1);  mm_default_46 = arg163_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_610: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 1, 196, 512]);  add_tensor_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_270: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_267, view_610);  add_267 = view_610 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_78 = torch.ops.aten.var_mean.correction(add_270, [3], correction = 0, keepdim = True)
        getitem_275: "f32[8, 1, 196, 1]" = var_mean_78[0]
        getitem_276: "f32[8, 1, 196, 1]" = var_mean_78[1];  var_mean_78 = None
        sub_115: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_270, getitem_276);  getitem_276 = None
        add_271: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_275, 1e-06);  getitem_275 = None
        rsqrt_78: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        mul_338: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_78);  sub_115 = rsqrt_78 = None
        mul_339: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_338, arg164_1);  mul_338 = arg164_1 = None
        add_272: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_339, arg165_1);  mul_339 = arg165_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_611: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_272, [1568, 512]);  add_272 = None
        permute_290: "f32[512, 2048]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        
        # No stacktrace found for following nodes
        mm_default_45: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_611, permute_290);  view_611 = permute_290 = None
        add_tensor_45: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_45, arg167_1);  mm_default_45 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_612: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 1, 196, 2048]);  add_tensor_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_340: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_612, 0.5)
        mul_341: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_612, 0.7071067811865476);  view_612 = None
        erf_36: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_341);  mul_341 = None
        add_273: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_342: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_340, add_273);  mul_340 = add_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_613: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_342, [1568, 2048]);  mul_342 = None
        permute_291: "f32[2048, 512]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        
        # No stacktrace found for following nodes
        mm_default_44: "f32[1568, 512]" = torch.ops.aten.mm.default(view_613, permute_291);  view_613 = permute_291 = None
        add_tensor_44: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_44, arg169_1);  mm_default_44 = arg169_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_614: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 1, 196, 512]);  add_tensor_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_274: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_270, view_614);  add_270 = view_614 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_79 = torch.ops.aten.var_mean.correction(add_274, [3], correction = 0, keepdim = True)
        getitem_277: "f32[8, 1, 196, 1]" = var_mean_79[0]
        getitem_278: "f32[8, 1, 196, 1]" = var_mean_79[1];  var_mean_79 = None
        sub_116: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_274, getitem_278);  getitem_278 = None
        add_275: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_277, 1e-06);  getitem_277 = None
        rsqrt_79: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        mul_343: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_79);  sub_116 = rsqrt_79 = None
        mul_344: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_343, arg170_1);  mul_343 = arg170_1 = None
        add_276: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_344, arg171_1);  mul_344 = arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_615: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_276, [1568, 512]);  add_276 = None
        permute_292: "f32[512, 1536]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        
        # No stacktrace found for following nodes
        mm_default_43: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_615, permute_292);  view_615 = permute_292 = None
        add_tensor_43: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_43, arg173_1);  mm_default_43 = arg173_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_616: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_43, [8, 1, 196, 1536]);  add_tensor_43 = None
        view_617: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_616, [8, 1, 196, 3, 16, 32]);  view_616 = None
        permute_293: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_617, [3, 0, 4, 1, 2, 5]);  view_617 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_37 = torch.ops.aten.unbind.int(permute_293);  permute_293 = None
        getitem_279: "f32[8, 16, 1, 196, 32]" = unbind_37[0]
        getitem_280: "f32[8, 16, 1, 196, 32]" = unbind_37[1]
        getitem_281: "f32[8, 16, 1, 196, 32]" = unbind_37[2];  unbind_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_345: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_279, 0.42044820762685725);  getitem_279 = None
        expand_148: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_345, [8, 16, 1, 196, 32]);  mul_345 = None
        clone_268: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_618: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_268, [128, 196, 32]);  clone_268 = None
        permute_294: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_280, [0, 1, 2, 4, 3]);  getitem_280 = None
        mul_346: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_294, 0.42044820762685725);  permute_294 = None
        expand_149: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_346, [8, 16, 1, 32, 196]);  mul_346 = None
        clone_269: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_619: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_269, [128, 32, 196]);  clone_269 = None
        bmm_74: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_618, view_619);  view_618 = view_619 = None
        view_620: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_74, [8, 16, 1, 196, 196]);  bmm_74 = None
        eq_37: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_620, -inf)
        logical_not_74: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_37);  eq_37 = None
        any_38: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_74, -1, True);  logical_not_74 = None
        logical_not_75: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_38);  any_38 = None
        full_default_13: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_37: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_620, [-1], True)
        sub_117: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_620, amax_37);  view_620 = amax_37 = None
        exp_37: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_117);  sub_117 = None
        sum_38: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_37: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_37, sum_38);  exp_37 = sum_38 = None
        where_37: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_75, full_default_13, div_37);  logical_not_75 = full_default_13 = div_37 = None
        expand_150: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_37, [8, 16, 1, 196, 196]);  where_37 = None
        view_621: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_150, [128, 196, 196]);  expand_150 = None
        expand_151: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_281, [8, 16, 1, 196, 32]);  getitem_281 = None
        clone_270: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_622: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_270, [128, 196, 32]);  clone_270 = None
        bmm_75: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_621, view_622);  view_621 = view_622 = None
        view_623: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_75, [8, 16, 1, 196, 32]);  bmm_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_295: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_623, [0, 2, 3, 4, 1]);  view_623 = None
        clone_271: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
        view_624: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_271, [8, 1, 196, 512]);  clone_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_625: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_624, [1568, 512]);  view_624 = None
        permute_296: "f32[512, 512]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        
        # No stacktrace found for following nodes
        mm_default_42: "f32[1568, 512]" = torch.ops.aten.mm.default(view_625, permute_296);  view_625 = permute_296 = None
        add_tensor_42: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_42, arg175_1);  mm_default_42 = arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_626: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_42, [8, 1, 196, 512]);  add_tensor_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_277: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_274, view_626);  add_274 = view_626 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_80 = torch.ops.aten.var_mean.correction(add_277, [3], correction = 0, keepdim = True)
        getitem_282: "f32[8, 1, 196, 1]" = var_mean_80[0]
        getitem_283: "f32[8, 1, 196, 1]" = var_mean_80[1];  var_mean_80 = None
        sub_118: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_277, getitem_283);  getitem_283 = None
        add_278: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-06);  getitem_282 = None
        rsqrt_80: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
        mul_347: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_80);  sub_118 = rsqrt_80 = None
        mul_348: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_347, arg176_1);  mul_347 = arg176_1 = None
        add_279: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_348, arg177_1);  mul_348 = arg177_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_627: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_279, [1568, 512]);  add_279 = None
        permute_297: "f32[512, 2048]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        
        # No stacktrace found for following nodes
        mm_default_41: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_627, permute_297);  view_627 = permute_297 = None
        add_tensor_41: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_41, arg179_1);  mm_default_41 = arg179_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_628: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_41, [8, 1, 196, 2048]);  add_tensor_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_349: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_628, 0.5)
        mul_350: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_628, 0.7071067811865476);  view_628 = None
        erf_37: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_280: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_351: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_349, add_280);  mul_349 = add_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_629: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_351, [1568, 2048]);  mul_351 = None
        permute_298: "f32[2048, 512]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        
        # No stacktrace found for following nodes
        mm_default_40: "f32[1568, 512]" = torch.ops.aten.mm.default(view_629, permute_298);  view_629 = permute_298 = None
        add_tensor_40: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_40, arg181_1);  mm_default_40 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_630: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 1, 196, 512]);  add_tensor_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_281: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_277, view_630);  add_277 = view_630 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_81 = torch.ops.aten.var_mean.correction(add_281, [3], correction = 0, keepdim = True)
        getitem_284: "f32[8, 1, 196, 1]" = var_mean_81[0]
        getitem_285: "f32[8, 1, 196, 1]" = var_mean_81[1];  var_mean_81 = None
        sub_119: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_281, getitem_285);  getitem_285 = None
        add_282: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
        rsqrt_81: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        mul_352: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_81);  sub_119 = rsqrt_81 = None
        mul_353: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_352, arg182_1);  mul_352 = arg182_1 = None
        add_283: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_353, arg183_1);  mul_353 = arg183_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_631: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_283, [1568, 512]);  add_283 = None
        permute_299: "f32[512, 1536]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        
        # No stacktrace found for following nodes
        mm_default_39: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_631, permute_299);  view_631 = permute_299 = None
        add_tensor_39: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_39, arg185_1);  mm_default_39 = arg185_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_632: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 1, 196, 1536]);  add_tensor_39 = None
        view_633: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_632, [8, 1, 196, 3, 16, 32]);  view_632 = None
        permute_300: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_633, [3, 0, 4, 1, 2, 5]);  view_633 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_38 = torch.ops.aten.unbind.int(permute_300);  permute_300 = None
        getitem_286: "f32[8, 16, 1, 196, 32]" = unbind_38[0]
        getitem_287: "f32[8, 16, 1, 196, 32]" = unbind_38[1]
        getitem_288: "f32[8, 16, 1, 196, 32]" = unbind_38[2];  unbind_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_354: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_286, 0.42044820762685725);  getitem_286 = None
        expand_152: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_354, [8, 16, 1, 196, 32]);  mul_354 = None
        clone_275: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
        view_634: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_275, [128, 196, 32]);  clone_275 = None
        permute_301: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_287, [0, 1, 2, 4, 3]);  getitem_287 = None
        mul_355: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_301, 0.42044820762685725);  permute_301 = None
        expand_153: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_355, [8, 16, 1, 32, 196]);  mul_355 = None
        clone_276: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_635: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_276, [128, 32, 196]);  clone_276 = None
        bmm_76: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_634, view_635);  view_634 = view_635 = None
        view_636: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_76, [8, 16, 1, 196, 196]);  bmm_76 = None
        eq_38: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_636, -inf)
        logical_not_76: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_38);  eq_38 = None
        any_39: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_76, -1, True);  logical_not_76 = None
        logical_not_77: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_39);  any_39 = None
        full_default_14: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_38: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_636, [-1], True)
        sub_120: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_636, amax_38);  view_636 = amax_38 = None
        exp_38: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_120);  sub_120 = None
        sum_39: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_38: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
        where_38: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_77, full_default_14, div_38);  logical_not_77 = full_default_14 = div_38 = None
        expand_154: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_38, [8, 16, 1, 196, 196]);  where_38 = None
        view_637: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_154, [128, 196, 196]);  expand_154 = None
        expand_155: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_288, [8, 16, 1, 196, 32]);  getitem_288 = None
        clone_277: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_638: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_277, [128, 196, 32]);  clone_277 = None
        bmm_77: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_637, view_638);  view_637 = view_638 = None
        view_639: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_77, [8, 16, 1, 196, 32]);  bmm_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_302: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_639, [0, 2, 3, 4, 1]);  view_639 = None
        clone_278: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_640: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_278, [8, 1, 196, 512]);  clone_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_641: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_640, [1568, 512]);  view_640 = None
        permute_303: "f32[512, 512]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        
        # No stacktrace found for following nodes
        mm_default_38: "f32[1568, 512]" = torch.ops.aten.mm.default(view_641, permute_303);  view_641 = permute_303 = None
        add_tensor_38: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_38, arg187_1);  mm_default_38 = arg187_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_642: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 1, 196, 512]);  add_tensor_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_284: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_281, view_642);  add_281 = view_642 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_82 = torch.ops.aten.var_mean.correction(add_284, [3], correction = 0, keepdim = True)
        getitem_289: "f32[8, 1, 196, 1]" = var_mean_82[0]
        getitem_290: "f32[8, 1, 196, 1]" = var_mean_82[1];  var_mean_82 = None
        sub_121: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_284, getitem_290);  getitem_290 = None
        add_285: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_289, 1e-06);  getitem_289 = None
        rsqrt_82: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
        mul_356: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_82);  sub_121 = rsqrt_82 = None
        mul_357: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_356, arg188_1);  mul_356 = arg188_1 = None
        add_286: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_357, arg189_1);  mul_357 = arg189_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_643: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_286, [1568, 512]);  add_286 = None
        permute_304: "f32[512, 2048]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_643, permute_304);  view_643 = permute_304 = None
        add_tensor_37: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_37, arg191_1);  mm_default_37 = arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_644: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 1, 196, 2048]);  add_tensor_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_358: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_644, 0.5)
        mul_359: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_644, 0.7071067811865476);  view_644 = None
        erf_38: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_359);  mul_359 = None
        add_287: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_360: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_358, add_287);  mul_358 = add_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_645: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_360, [1568, 2048]);  mul_360 = None
        permute_305: "f32[2048, 512]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[1568, 512]" = torch.ops.aten.mm.default(view_645, permute_305);  view_645 = permute_305 = None
        add_tensor_36: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_36, arg193_1);  mm_default_36 = arg193_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_646: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 1, 196, 512]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_288: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_284, view_646);  add_284 = view_646 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_83 = torch.ops.aten.var_mean.correction(add_288, [3], correction = 0, keepdim = True)
        getitem_291: "f32[8, 1, 196, 1]" = var_mean_83[0]
        getitem_292: "f32[8, 1, 196, 1]" = var_mean_83[1];  var_mean_83 = None
        sub_122: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_288, getitem_292);  getitem_292 = None
        add_289: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_291, 1e-06);  getitem_291 = None
        rsqrt_83: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
        mul_361: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_83);  sub_122 = rsqrt_83 = None
        mul_362: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_361, arg194_1);  mul_361 = arg194_1 = None
        add_290: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_362, arg195_1);  mul_362 = arg195_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_647: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_290, [1568, 512]);  add_290 = None
        permute_306: "f32[512, 1536]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_647, permute_306);  view_647 = permute_306 = None
        add_tensor_35: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_35, arg197_1);  mm_default_35 = arg197_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_648: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 1, 196, 1536]);  add_tensor_35 = None
        view_649: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_648, [8, 1, 196, 3, 16, 32]);  view_648 = None
        permute_307: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_649, [3, 0, 4, 1, 2, 5]);  view_649 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_39 = torch.ops.aten.unbind.int(permute_307);  permute_307 = None
        getitem_293: "f32[8, 16, 1, 196, 32]" = unbind_39[0]
        getitem_294: "f32[8, 16, 1, 196, 32]" = unbind_39[1]
        getitem_295: "f32[8, 16, 1, 196, 32]" = unbind_39[2];  unbind_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_363: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_293, 0.42044820762685725);  getitem_293 = None
        expand_156: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_363, [8, 16, 1, 196, 32]);  mul_363 = None
        clone_282: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_650: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_282, [128, 196, 32]);  clone_282 = None
        permute_308: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_294, [0, 1, 2, 4, 3]);  getitem_294 = None
        mul_364: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_308, 0.42044820762685725);  permute_308 = None
        expand_157: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_364, [8, 16, 1, 32, 196]);  mul_364 = None
        clone_283: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_651: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_283, [128, 32, 196]);  clone_283 = None
        bmm_78: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_650, view_651);  view_650 = view_651 = None
        view_652: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_78, [8, 16, 1, 196, 196]);  bmm_78 = None
        eq_39: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_652, -inf)
        logical_not_78: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_39);  eq_39 = None
        any_40: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_78, -1, True);  logical_not_78 = None
        logical_not_79: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_40);  any_40 = None
        full_default_15: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_39: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_652, [-1], True)
        sub_123: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_652, amax_39);  view_652 = amax_39 = None
        exp_39: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_123);  sub_123 = None
        sum_40: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_39: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_39, sum_40);  exp_39 = sum_40 = None
        where_39: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_79, full_default_15, div_39);  logical_not_79 = full_default_15 = div_39 = None
        expand_158: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_39, [8, 16, 1, 196, 196]);  where_39 = None
        view_653: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_158, [128, 196, 196]);  expand_158 = None
        expand_159: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_295, [8, 16, 1, 196, 32]);  getitem_295 = None
        clone_284: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_654: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_284, [128, 196, 32]);  clone_284 = None
        bmm_79: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_653, view_654);  view_653 = view_654 = None
        view_655: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_79, [8, 16, 1, 196, 32]);  bmm_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_309: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_655, [0, 2, 3, 4, 1]);  view_655 = None
        clone_285: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
        view_656: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_285, [8, 1, 196, 512]);  clone_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_657: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_656, [1568, 512]);  view_656 = None
        permute_310: "f32[512, 512]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[1568, 512]" = torch.ops.aten.mm.default(view_657, permute_310);  view_657 = permute_310 = None
        add_tensor_34: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_34, arg199_1);  mm_default_34 = arg199_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_658: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 1, 196, 512]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_291: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_288, view_658);  add_288 = view_658 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_84 = torch.ops.aten.var_mean.correction(add_291, [3], correction = 0, keepdim = True)
        getitem_296: "f32[8, 1, 196, 1]" = var_mean_84[0]
        getitem_297: "f32[8, 1, 196, 1]" = var_mean_84[1];  var_mean_84 = None
        sub_124: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_291, getitem_297);  getitem_297 = None
        add_292: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-06);  getitem_296 = None
        rsqrt_84: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        mul_365: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_84);  sub_124 = rsqrt_84 = None
        mul_366: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_365, arg200_1);  mul_365 = arg200_1 = None
        add_293: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_366, arg201_1);  mul_366 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_659: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_293, [1568, 512]);  add_293 = None
        permute_311: "f32[512, 2048]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_659, permute_311);  view_659 = permute_311 = None
        add_tensor_33: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_33, arg203_1);  mm_default_33 = arg203_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_660: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 1, 196, 2048]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_367: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_660, 0.5)
        mul_368: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_660, 0.7071067811865476);  view_660 = None
        erf_39: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_368);  mul_368 = None
        add_294: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_369: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_367, add_294);  mul_367 = add_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_661: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_369, [1568, 2048]);  mul_369 = None
        permute_312: "f32[2048, 512]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[1568, 512]" = torch.ops.aten.mm.default(view_661, permute_312);  view_661 = permute_312 = None
        add_tensor_32: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_32, arg205_1);  mm_default_32 = arg205_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_662: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 1, 196, 512]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_295: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_291, view_662);  add_291 = view_662 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_85 = torch.ops.aten.var_mean.correction(add_295, [3], correction = 0, keepdim = True)
        getitem_298: "f32[8, 1, 196, 1]" = var_mean_85[0]
        getitem_299: "f32[8, 1, 196, 1]" = var_mean_85[1];  var_mean_85 = None
        sub_125: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_295, getitem_299);  getitem_299 = None
        add_296: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_298, 1e-06);  getitem_298 = None
        rsqrt_85: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        mul_370: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_85);  sub_125 = rsqrt_85 = None
        mul_371: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_370, arg206_1);  mul_370 = arg206_1 = None
        add_297: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_371, arg207_1);  mul_371 = arg207_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_663: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_297, [1568, 512]);  add_297 = None
        permute_313: "f32[512, 1536]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_663, permute_313);  view_663 = permute_313 = None
        add_tensor_31: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_31, arg209_1);  mm_default_31 = arg209_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_664: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 1, 196, 1536]);  add_tensor_31 = None
        view_665: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_664, [8, 1, 196, 3, 16, 32]);  view_664 = None
        permute_314: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_665, [3, 0, 4, 1, 2, 5]);  view_665 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_40 = torch.ops.aten.unbind.int(permute_314);  permute_314 = None
        getitem_300: "f32[8, 16, 1, 196, 32]" = unbind_40[0]
        getitem_301: "f32[8, 16, 1, 196, 32]" = unbind_40[1]
        getitem_302: "f32[8, 16, 1, 196, 32]" = unbind_40[2];  unbind_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_372: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_300, 0.42044820762685725);  getitem_300 = None
        expand_160: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_372, [8, 16, 1, 196, 32]);  mul_372 = None
        clone_289: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
        view_666: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_289, [128, 196, 32]);  clone_289 = None
        permute_315: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_301, [0, 1, 2, 4, 3]);  getitem_301 = None
        mul_373: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_315, 0.42044820762685725);  permute_315 = None
        expand_161: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_373, [8, 16, 1, 32, 196]);  mul_373 = None
        clone_290: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_667: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_290, [128, 32, 196]);  clone_290 = None
        bmm_80: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_666, view_667);  view_666 = view_667 = None
        view_668: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_80, [8, 16, 1, 196, 196]);  bmm_80 = None
        eq_40: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_668, -inf)
        logical_not_80: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_40);  eq_40 = None
        any_41: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_80, -1, True);  logical_not_80 = None
        logical_not_81: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_41);  any_41 = None
        full_default_16: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_40: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_668, [-1], True)
        sub_126: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_668, amax_40);  view_668 = amax_40 = None
        exp_40: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_126);  sub_126 = None
        sum_41: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_40: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
        where_40: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_81, full_default_16, div_40);  logical_not_81 = full_default_16 = div_40 = None
        expand_162: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_40, [8, 16, 1, 196, 196]);  where_40 = None
        view_669: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_162, [128, 196, 196]);  expand_162 = None
        expand_163: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_302, [8, 16, 1, 196, 32]);  getitem_302 = None
        clone_291: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_670: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_291, [128, 196, 32]);  clone_291 = None
        bmm_81: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_669, view_670);  view_669 = view_670 = None
        view_671: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_81, [8, 16, 1, 196, 32]);  bmm_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_316: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_671, [0, 2, 3, 4, 1]);  view_671 = None
        clone_292: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
        view_672: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_292, [8, 1, 196, 512]);  clone_292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_673: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_672, [1568, 512]);  view_672 = None
        permute_317: "f32[512, 512]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[1568, 512]" = torch.ops.aten.mm.default(view_673, permute_317);  view_673 = permute_317 = None
        add_tensor_30: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_30, arg211_1);  mm_default_30 = arg211_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_674: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 1, 196, 512]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_298: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_295, view_674);  add_295 = view_674 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_86 = torch.ops.aten.var_mean.correction(add_298, [3], correction = 0, keepdim = True)
        getitem_303: "f32[8, 1, 196, 1]" = var_mean_86[0]
        getitem_304: "f32[8, 1, 196, 1]" = var_mean_86[1];  var_mean_86 = None
        sub_127: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_298, getitem_304);  getitem_304 = None
        add_299: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_303, 1e-06);  getitem_303 = None
        rsqrt_86: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        mul_374: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_86);  sub_127 = rsqrt_86 = None
        mul_375: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_374, arg212_1);  mul_374 = arg212_1 = None
        add_300: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_375, arg213_1);  mul_375 = arg213_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_675: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_300, [1568, 512]);  add_300 = None
        permute_318: "f32[512, 2048]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_675, permute_318);  view_675 = permute_318 = None
        add_tensor_29: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_29, arg215_1);  mm_default_29 = arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_676: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 1, 196, 2048]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_376: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_676, 0.5)
        mul_377: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_676, 0.7071067811865476);  view_676 = None
        erf_40: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_377);  mul_377 = None
        add_301: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_378: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_376, add_301);  mul_376 = add_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_677: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_378, [1568, 2048]);  mul_378 = None
        permute_319: "f32[2048, 512]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[1568, 512]" = torch.ops.aten.mm.default(view_677, permute_319);  view_677 = permute_319 = None
        add_tensor_28: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_28, arg217_1);  mm_default_28 = arg217_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_678: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 1, 196, 512]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_302: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_298, view_678);  add_298 = view_678 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_87 = torch.ops.aten.var_mean.correction(add_302, [3], correction = 0, keepdim = True)
        getitem_305: "f32[8, 1, 196, 1]" = var_mean_87[0]
        getitem_306: "f32[8, 1, 196, 1]" = var_mean_87[1];  var_mean_87 = None
        sub_128: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_302, getitem_306);  getitem_306 = None
        add_303: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_305, 1e-06);  getitem_305 = None
        rsqrt_87: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        mul_379: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_87);  sub_128 = rsqrt_87 = None
        mul_380: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_379, arg218_1);  mul_379 = arg218_1 = None
        add_304: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_380, arg219_1);  mul_380 = arg219_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_679: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_304, [1568, 512]);  add_304 = None
        permute_320: "f32[512, 1536]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_679, permute_320);  view_679 = permute_320 = None
        add_tensor_27: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_27, arg221_1);  mm_default_27 = arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_680: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 1, 196, 1536]);  add_tensor_27 = None
        view_681: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_680, [8, 1, 196, 3, 16, 32]);  view_680 = None
        permute_321: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_681, [3, 0, 4, 1, 2, 5]);  view_681 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_41 = torch.ops.aten.unbind.int(permute_321);  permute_321 = None
        getitem_307: "f32[8, 16, 1, 196, 32]" = unbind_41[0]
        getitem_308: "f32[8, 16, 1, 196, 32]" = unbind_41[1]
        getitem_309: "f32[8, 16, 1, 196, 32]" = unbind_41[2];  unbind_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_381: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_307, 0.42044820762685725);  getitem_307 = None
        expand_164: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_381, [8, 16, 1, 196, 32]);  mul_381 = None
        clone_296: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_682: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_296, [128, 196, 32]);  clone_296 = None
        permute_322: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_308, [0, 1, 2, 4, 3]);  getitem_308 = None
        mul_382: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_322, 0.42044820762685725);  permute_322 = None
        expand_165: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_382, [8, 16, 1, 32, 196]);  mul_382 = None
        clone_297: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_683: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_297, [128, 32, 196]);  clone_297 = None
        bmm_82: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_682, view_683);  view_682 = view_683 = None
        view_684: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_82, [8, 16, 1, 196, 196]);  bmm_82 = None
        eq_41: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_684, -inf)
        logical_not_82: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_41);  eq_41 = None
        any_42: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_82, -1, True);  logical_not_82 = None
        logical_not_83: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_42);  any_42 = None
        full_default_17: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_41: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_684, [-1], True)
        sub_129: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_684, amax_41);  view_684 = amax_41 = None
        exp_41: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_129);  sub_129 = None
        sum_42: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_41: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_41, sum_42);  exp_41 = sum_42 = None
        where_41: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_83, full_default_17, div_41);  logical_not_83 = full_default_17 = div_41 = None
        expand_166: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_41, [8, 16, 1, 196, 196]);  where_41 = None
        view_685: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_166, [128, 196, 196]);  expand_166 = None
        expand_167: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_309, [8, 16, 1, 196, 32]);  getitem_309 = None
        clone_298: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_167, memory_format = torch.contiguous_format);  expand_167 = None
        view_686: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_298, [128, 196, 32]);  clone_298 = None
        bmm_83: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_685, view_686);  view_685 = view_686 = None
        view_687: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_83, [8, 16, 1, 196, 32]);  bmm_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_323: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_687, [0, 2, 3, 4, 1]);  view_687 = None
        clone_299: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
        view_688: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_299, [8, 1, 196, 512]);  clone_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_689: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_688, [1568, 512]);  view_688 = None
        permute_324: "f32[512, 512]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[1568, 512]" = torch.ops.aten.mm.default(view_689, permute_324);  view_689 = permute_324 = None
        add_tensor_26: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_26, arg223_1);  mm_default_26 = arg223_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_690: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 1, 196, 512]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_305: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_302, view_690);  add_302 = view_690 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_88 = torch.ops.aten.var_mean.correction(add_305, [3], correction = 0, keepdim = True)
        getitem_310: "f32[8, 1, 196, 1]" = var_mean_88[0]
        getitem_311: "f32[8, 1, 196, 1]" = var_mean_88[1];  var_mean_88 = None
        sub_130: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_305, getitem_311);  getitem_311 = None
        add_306: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-06);  getitem_310 = None
        rsqrt_88: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        mul_383: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_88);  sub_130 = rsqrt_88 = None
        mul_384: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_383, arg224_1);  mul_383 = arg224_1 = None
        add_307: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_384, arg225_1);  mul_384 = arg225_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_691: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_307, [1568, 512]);  add_307 = None
        permute_325: "f32[512, 2048]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_691, permute_325);  view_691 = permute_325 = None
        add_tensor_25: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_25, arg227_1);  mm_default_25 = arg227_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_692: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 1, 196, 2048]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_385: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_692, 0.5)
        mul_386: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_692, 0.7071067811865476);  view_692 = None
        erf_41: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_386);  mul_386 = None
        add_308: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_387: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_385, add_308);  mul_385 = add_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_693: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_387, [1568, 2048]);  mul_387 = None
        permute_326: "f32[2048, 512]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[1568, 512]" = torch.ops.aten.mm.default(view_693, permute_326);  view_693 = permute_326 = None
        add_tensor_24: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_24, arg229_1);  mm_default_24 = arg229_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_694: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 1, 196, 512]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_309: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_305, view_694);  add_305 = view_694 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_89 = torch.ops.aten.var_mean.correction(add_309, [3], correction = 0, keepdim = True)
        getitem_312: "f32[8, 1, 196, 1]" = var_mean_89[0]
        getitem_313: "f32[8, 1, 196, 1]" = var_mean_89[1];  var_mean_89 = None
        sub_131: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_309, getitem_313);  getitem_313 = None
        add_310: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_312, 1e-06);  getitem_312 = None
        rsqrt_89: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        mul_388: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_89);  sub_131 = rsqrt_89 = None
        mul_389: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_388, arg230_1);  mul_388 = arg230_1 = None
        add_311: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_389, arg231_1);  mul_389 = arg231_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_695: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_311, [1568, 512]);  add_311 = None
        permute_327: "f32[512, 1536]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_695, permute_327);  view_695 = permute_327 = None
        add_tensor_23: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_23, arg233_1);  mm_default_23 = arg233_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_696: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 1, 196, 1536]);  add_tensor_23 = None
        view_697: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_696, [8, 1, 196, 3, 16, 32]);  view_696 = None
        permute_328: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_697, [3, 0, 4, 1, 2, 5]);  view_697 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_42 = torch.ops.aten.unbind.int(permute_328);  permute_328 = None
        getitem_314: "f32[8, 16, 1, 196, 32]" = unbind_42[0]
        getitem_315: "f32[8, 16, 1, 196, 32]" = unbind_42[1]
        getitem_316: "f32[8, 16, 1, 196, 32]" = unbind_42[2];  unbind_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_390: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_314, 0.42044820762685725);  getitem_314 = None
        expand_168: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_390, [8, 16, 1, 196, 32]);  mul_390 = None
        clone_303: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
        view_698: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_303, [128, 196, 32]);  clone_303 = None
        permute_329: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_315, [0, 1, 2, 4, 3]);  getitem_315 = None
        mul_391: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_329, 0.42044820762685725);  permute_329 = None
        expand_169: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_391, [8, 16, 1, 32, 196]);  mul_391 = None
        clone_304: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_699: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_304, [128, 32, 196]);  clone_304 = None
        bmm_84: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_698, view_699);  view_698 = view_699 = None
        view_700: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_84, [8, 16, 1, 196, 196]);  bmm_84 = None
        eq_42: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_700, -inf)
        logical_not_84: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_42);  eq_42 = None
        any_43: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_84, -1, True);  logical_not_84 = None
        logical_not_85: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_43);  any_43 = None
        full_default_18: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_42: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_700, [-1], True)
        sub_132: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_700, amax_42);  view_700 = amax_42 = None
        exp_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_132);  sub_132 = None
        sum_43: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
        where_42: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_85, full_default_18, div_42);  logical_not_85 = full_default_18 = div_42 = None
        expand_170: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_42, [8, 16, 1, 196, 196]);  where_42 = None
        view_701: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_170, [128, 196, 196]);  expand_170 = None
        expand_171: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_316, [8, 16, 1, 196, 32]);  getitem_316 = None
        clone_305: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_702: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_305, [128, 196, 32]);  clone_305 = None
        bmm_85: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_701, view_702);  view_701 = view_702 = None
        view_703: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_85, [8, 16, 1, 196, 32]);  bmm_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_330: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_703, [0, 2, 3, 4, 1]);  view_703 = None
        clone_306: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_704: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_306, [8, 1, 196, 512]);  clone_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_705: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_704, [1568, 512]);  view_704 = None
        permute_331: "f32[512, 512]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[1568, 512]" = torch.ops.aten.mm.default(view_705, permute_331);  view_705 = permute_331 = None
        add_tensor_22: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_22, arg235_1);  mm_default_22 = arg235_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_706: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 1, 196, 512]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_312: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_309, view_706);  add_309 = view_706 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_90 = torch.ops.aten.var_mean.correction(add_312, [3], correction = 0, keepdim = True)
        getitem_317: "f32[8, 1, 196, 1]" = var_mean_90[0]
        getitem_318: "f32[8, 1, 196, 1]" = var_mean_90[1];  var_mean_90 = None
        sub_133: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_312, getitem_318);  getitem_318 = None
        add_313: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_317, 1e-06);  getitem_317 = None
        rsqrt_90: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        mul_392: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_90);  sub_133 = rsqrt_90 = None
        mul_393: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_392, arg236_1);  mul_392 = arg236_1 = None
        add_314: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_393, arg237_1);  mul_393 = arg237_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_707: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_314, [1568, 512]);  add_314 = None
        permute_332: "f32[512, 2048]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_707, permute_332);  view_707 = permute_332 = None
        add_tensor_21: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_21, arg239_1);  mm_default_21 = arg239_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_708: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 1, 196, 2048]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_394: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_708, 0.5)
        mul_395: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_708, 0.7071067811865476);  view_708 = None
        erf_42: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_395);  mul_395 = None
        add_315: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_396: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_394, add_315);  mul_394 = add_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_709: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_396, [1568, 2048]);  mul_396 = None
        permute_333: "f32[2048, 512]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[1568, 512]" = torch.ops.aten.mm.default(view_709, permute_333);  view_709 = permute_333 = None
        add_tensor_20: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_20, arg241_1);  mm_default_20 = arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_710: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 1, 196, 512]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_316: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_312, view_710);  add_312 = view_710 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_91 = torch.ops.aten.var_mean.correction(add_316, [3], correction = 0, keepdim = True)
        getitem_319: "f32[8, 1, 196, 1]" = var_mean_91[0]
        getitem_320: "f32[8, 1, 196, 1]" = var_mean_91[1];  var_mean_91 = None
        sub_134: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_316, getitem_320);  getitem_320 = None
        add_317: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_319, 1e-06);  getitem_319 = None
        rsqrt_91: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        mul_397: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_91);  sub_134 = rsqrt_91 = None
        mul_398: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_397, arg242_1);  mul_397 = arg242_1 = None
        add_318: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_398, arg243_1);  mul_398 = arg243_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_711: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_318, [1568, 512]);  add_318 = None
        permute_334: "f32[512, 1536]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_711, permute_334);  view_711 = permute_334 = None
        add_tensor_19: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_19, arg245_1);  mm_default_19 = arg245_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_712: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 1, 196, 1536]);  add_tensor_19 = None
        view_713: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_712, [8, 1, 196, 3, 16, 32]);  view_712 = None
        permute_335: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_713, [3, 0, 4, 1, 2, 5]);  view_713 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_43 = torch.ops.aten.unbind.int(permute_335);  permute_335 = None
        getitem_321: "f32[8, 16, 1, 196, 32]" = unbind_43[0]
        getitem_322: "f32[8, 16, 1, 196, 32]" = unbind_43[1]
        getitem_323: "f32[8, 16, 1, 196, 32]" = unbind_43[2];  unbind_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_399: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_321, 0.42044820762685725);  getitem_321 = None
        expand_172: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_399, [8, 16, 1, 196, 32]);  mul_399 = None
        clone_310: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_714: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_310, [128, 196, 32]);  clone_310 = None
        permute_336: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_322, [0, 1, 2, 4, 3]);  getitem_322 = None
        mul_400: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_336, 0.42044820762685725);  permute_336 = None
        expand_173: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_400, [8, 16, 1, 32, 196]);  mul_400 = None
        clone_311: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_715: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_311, [128, 32, 196]);  clone_311 = None
        bmm_86: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_714, view_715);  view_714 = view_715 = None
        view_716: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_86, [8, 16, 1, 196, 196]);  bmm_86 = None
        eq_43: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_716, -inf)
        logical_not_86: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_43);  eq_43 = None
        any_44: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_86, -1, True);  logical_not_86 = None
        logical_not_87: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_44);  any_44 = None
        full_default_19: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_43: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_716, [-1], True)
        sub_135: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_716, amax_43);  view_716 = amax_43 = None
        exp_43: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        sum_44: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_43: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_43, sum_44);  exp_43 = sum_44 = None
        where_43: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_87, full_default_19, div_43);  logical_not_87 = full_default_19 = div_43 = None
        expand_174: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_43, [8, 16, 1, 196, 196]);  where_43 = None
        view_717: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_174, [128, 196, 196]);  expand_174 = None
        expand_175: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_323, [8, 16, 1, 196, 32]);  getitem_323 = None
        clone_312: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_175, memory_format = torch.contiguous_format);  expand_175 = None
        view_718: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_312, [128, 196, 32]);  clone_312 = None
        bmm_87: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_717, view_718);  view_717 = view_718 = None
        view_719: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_87, [8, 16, 1, 196, 32]);  bmm_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_337: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_719, [0, 2, 3, 4, 1]);  view_719 = None
        clone_313: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_720: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_313, [8, 1, 196, 512]);  clone_313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_721: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_720, [1568, 512]);  view_720 = None
        permute_338: "f32[512, 512]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[1568, 512]" = torch.ops.aten.mm.default(view_721, permute_338);  view_721 = permute_338 = None
        add_tensor_18: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_18, arg247_1);  mm_default_18 = arg247_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_722: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 1, 196, 512]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_319: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_316, view_722);  add_316 = view_722 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_92 = torch.ops.aten.var_mean.correction(add_319, [3], correction = 0, keepdim = True)
        getitem_324: "f32[8, 1, 196, 1]" = var_mean_92[0]
        getitem_325: "f32[8, 1, 196, 1]" = var_mean_92[1];  var_mean_92 = None
        sub_136: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_319, getitem_325);  getitem_325 = None
        add_320: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_324, 1e-06);  getitem_324 = None
        rsqrt_92: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        mul_401: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_92);  sub_136 = rsqrt_92 = None
        mul_402: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_401, arg248_1);  mul_401 = arg248_1 = None
        add_321: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_402, arg249_1);  mul_402 = arg249_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_723: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_321, [1568, 512]);  add_321 = None
        permute_339: "f32[512, 2048]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_723, permute_339);  view_723 = permute_339 = None
        add_tensor_17: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_17, arg251_1);  mm_default_17 = arg251_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_724: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 1, 196, 2048]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_403: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_724, 0.5)
        mul_404: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_724, 0.7071067811865476);  view_724 = None
        erf_43: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_404);  mul_404 = None
        add_322: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_405: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_403, add_322);  mul_403 = add_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_725: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_405, [1568, 2048]);  mul_405 = None
        permute_340: "f32[2048, 512]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[1568, 512]" = torch.ops.aten.mm.default(view_725, permute_340);  view_725 = permute_340 = None
        add_tensor_16: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_16, arg253_1);  mm_default_16 = arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_726: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 1, 196, 512]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_323: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_319, view_726);  add_319 = view_726 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_93 = torch.ops.aten.var_mean.correction(add_323, [3], correction = 0, keepdim = True)
        getitem_326: "f32[8, 1, 196, 1]" = var_mean_93[0]
        getitem_327: "f32[8, 1, 196, 1]" = var_mean_93[1];  var_mean_93 = None
        sub_137: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_323, getitem_327);  getitem_327 = None
        add_324: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_326, 1e-06);  getitem_326 = None
        rsqrt_93: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        mul_406: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_93);  sub_137 = rsqrt_93 = None
        mul_407: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_406, arg254_1);  mul_406 = arg254_1 = None
        add_325: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_407, arg255_1);  mul_407 = arg255_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_727: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_325, [1568, 512]);  add_325 = None
        permute_341: "f32[512, 1536]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_727, permute_341);  view_727 = permute_341 = None
        add_tensor_15: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_15, arg257_1);  mm_default_15 = arg257_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_728: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 1, 196, 1536]);  add_tensor_15 = None
        view_729: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_728, [8, 1, 196, 3, 16, 32]);  view_728 = None
        permute_342: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_729, [3, 0, 4, 1, 2, 5]);  view_729 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_44 = torch.ops.aten.unbind.int(permute_342);  permute_342 = None
        getitem_328: "f32[8, 16, 1, 196, 32]" = unbind_44[0]
        getitem_329: "f32[8, 16, 1, 196, 32]" = unbind_44[1]
        getitem_330: "f32[8, 16, 1, 196, 32]" = unbind_44[2];  unbind_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_408: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_328, 0.42044820762685725);  getitem_328 = None
        expand_176: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_408, [8, 16, 1, 196, 32]);  mul_408 = None
        clone_317: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
        view_730: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_317, [128, 196, 32]);  clone_317 = None
        permute_343: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_329, [0, 1, 2, 4, 3]);  getitem_329 = None
        mul_409: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_343, 0.42044820762685725);  permute_343 = None
        expand_177: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_409, [8, 16, 1, 32, 196]);  mul_409 = None
        clone_318: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_731: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_318, [128, 32, 196]);  clone_318 = None
        bmm_88: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_730, view_731);  view_730 = view_731 = None
        view_732: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_88, [8, 16, 1, 196, 196]);  bmm_88 = None
        eq_44: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_732, -inf)
        logical_not_88: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_44);  eq_44 = None
        any_45: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_88, -1, True);  logical_not_88 = None
        logical_not_89: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_45);  any_45 = None
        full_default_20: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_44: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_732, [-1], True)
        sub_138: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_732, amax_44);  view_732 = amax_44 = None
        exp_44: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_138);  sub_138 = None
        sum_45: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_44: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
        where_44: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_89, full_default_20, div_44);  logical_not_89 = full_default_20 = div_44 = None
        expand_178: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_44, [8, 16, 1, 196, 196]);  where_44 = None
        view_733: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_178, [128, 196, 196]);  expand_178 = None
        expand_179: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_330, [8, 16, 1, 196, 32]);  getitem_330 = None
        clone_319: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_734: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_319, [128, 196, 32]);  clone_319 = None
        bmm_89: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_733, view_734);  view_733 = view_734 = None
        view_735: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_89, [8, 16, 1, 196, 32]);  bmm_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_344: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_735, [0, 2, 3, 4, 1]);  view_735 = None
        clone_320: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_344, memory_format = torch.contiguous_format);  permute_344 = None
        view_736: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_320, [8, 1, 196, 512]);  clone_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_737: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_736, [1568, 512]);  view_736 = None
        permute_345: "f32[512, 512]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[1568, 512]" = torch.ops.aten.mm.default(view_737, permute_345);  view_737 = permute_345 = None
        add_tensor_14: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_14, arg259_1);  mm_default_14 = arg259_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_738: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 1, 196, 512]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_326: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_323, view_738);  add_323 = view_738 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_94 = torch.ops.aten.var_mean.correction(add_326, [3], correction = 0, keepdim = True)
        getitem_331: "f32[8, 1, 196, 1]" = var_mean_94[0]
        getitem_332: "f32[8, 1, 196, 1]" = var_mean_94[1];  var_mean_94 = None
        sub_139: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_326, getitem_332);  getitem_332 = None
        add_327: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_331, 1e-06);  getitem_331 = None
        rsqrt_94: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        mul_410: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_94);  sub_139 = rsqrt_94 = None
        mul_411: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_410, arg260_1);  mul_410 = arg260_1 = None
        add_328: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_411, arg261_1);  mul_411 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_739: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_328, [1568, 512]);  add_328 = None
        permute_346: "f32[512, 2048]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_739, permute_346);  view_739 = permute_346 = None
        add_tensor_13: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_13, arg263_1);  mm_default_13 = arg263_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_740: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 1, 196, 2048]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_412: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_740, 0.5)
        mul_413: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_740, 0.7071067811865476);  view_740 = None
        erf_44: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_413);  mul_413 = None
        add_329: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_414: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_412, add_329);  mul_412 = add_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_741: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_414, [1568, 2048]);  mul_414 = None
        permute_347: "f32[2048, 512]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[1568, 512]" = torch.ops.aten.mm.default(view_741, permute_347);  view_741 = permute_347 = None
        add_tensor_12: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_12, arg265_1);  mm_default_12 = arg265_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_742: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 1, 196, 512]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_330: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_326, view_742);  add_326 = view_742 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_95 = torch.ops.aten.var_mean.correction(add_330, [3], correction = 0, keepdim = True)
        getitem_333: "f32[8, 1, 196, 1]" = var_mean_95[0]
        getitem_334: "f32[8, 1, 196, 1]" = var_mean_95[1];  var_mean_95 = None
        sub_140: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_330, getitem_334);  getitem_334 = None
        add_331: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_333, 1e-06);  getitem_333 = None
        rsqrt_95: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        mul_415: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_95);  sub_140 = rsqrt_95 = None
        mul_416: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_415, arg266_1);  mul_415 = arg266_1 = None
        add_332: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_416, arg267_1);  mul_416 = arg267_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_743: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_332, [1568, 512]);  add_332 = None
        permute_348: "f32[512, 1536]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_743, permute_348);  view_743 = permute_348 = None
        add_tensor_11: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_11, arg269_1);  mm_default_11 = arg269_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_744: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 1, 196, 1536]);  add_tensor_11 = None
        view_745: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_744, [8, 1, 196, 3, 16, 32]);  view_744 = None
        permute_349: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_745, [3, 0, 4, 1, 2, 5]);  view_745 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_45 = torch.ops.aten.unbind.int(permute_349);  permute_349 = None
        getitem_335: "f32[8, 16, 1, 196, 32]" = unbind_45[0]
        getitem_336: "f32[8, 16, 1, 196, 32]" = unbind_45[1]
        getitem_337: "f32[8, 16, 1, 196, 32]" = unbind_45[2];  unbind_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_417: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_335, 0.42044820762685725);  getitem_335 = None
        expand_180: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_417, [8, 16, 1, 196, 32]);  mul_417 = None
        clone_324: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_746: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_324, [128, 196, 32]);  clone_324 = None
        permute_350: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_336, [0, 1, 2, 4, 3]);  getitem_336 = None
        mul_418: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_350, 0.42044820762685725);  permute_350 = None
        expand_181: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_418, [8, 16, 1, 32, 196]);  mul_418 = None
        clone_325: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_747: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_325, [128, 32, 196]);  clone_325 = None
        bmm_90: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_746, view_747);  view_746 = view_747 = None
        view_748: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_90, [8, 16, 1, 196, 196]);  bmm_90 = None
        eq_45: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_748, -inf)
        logical_not_90: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_45);  eq_45 = None
        any_46: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_90, -1, True);  logical_not_90 = None
        logical_not_91: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_46);  any_46 = None
        full_default_21: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_45: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_748, [-1], True)
        sub_141: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_748, amax_45);  view_748 = amax_45 = None
        exp_45: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_141);  sub_141 = None
        sum_46: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_45: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_45, sum_46);  exp_45 = sum_46 = None
        where_45: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_91, full_default_21, div_45);  logical_not_91 = full_default_21 = div_45 = None
        expand_182: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_45, [8, 16, 1, 196, 196]);  where_45 = None
        view_749: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_182, [128, 196, 196]);  expand_182 = None
        expand_183: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_337, [8, 16, 1, 196, 32]);  getitem_337 = None
        clone_326: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_750: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_326, [128, 196, 32]);  clone_326 = None
        bmm_91: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_749, view_750);  view_749 = view_750 = None
        view_751: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_91, [8, 16, 1, 196, 32]);  bmm_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_351: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_751, [0, 2, 3, 4, 1]);  view_751 = None
        clone_327: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
        view_752: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_327, [8, 1, 196, 512]);  clone_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_753: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_752, [1568, 512]);  view_752 = None
        permute_352: "f32[512, 512]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[1568, 512]" = torch.ops.aten.mm.default(view_753, permute_352);  view_753 = permute_352 = None
        add_tensor_10: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_10, arg271_1);  mm_default_10 = arg271_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_754: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 1, 196, 512]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_333: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_330, view_754);  add_330 = view_754 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_96 = torch.ops.aten.var_mean.correction(add_333, [3], correction = 0, keepdim = True)
        getitem_338: "f32[8, 1, 196, 1]" = var_mean_96[0]
        getitem_339: "f32[8, 1, 196, 1]" = var_mean_96[1];  var_mean_96 = None
        sub_142: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_333, getitem_339);  getitem_339 = None
        add_334: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-06);  getitem_338 = None
        rsqrt_96: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        mul_419: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_96);  sub_142 = rsqrt_96 = None
        mul_420: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_419, arg272_1);  mul_419 = arg272_1 = None
        add_335: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_420, arg273_1);  mul_420 = arg273_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_755: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_335, [1568, 512]);  add_335 = None
        permute_353: "f32[512, 2048]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_755, permute_353);  view_755 = permute_353 = None
        add_tensor_9: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_9, arg275_1);  mm_default_9 = arg275_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_756: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 1, 196, 2048]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_421: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_756, 0.5)
        mul_422: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_756, 0.7071067811865476);  view_756 = None
        erf_45: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_422);  mul_422 = None
        add_336: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_423: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_421, add_336);  mul_421 = add_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_757: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_423, [1568, 2048]);  mul_423 = None
        permute_354: "f32[2048, 512]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[1568, 512]" = torch.ops.aten.mm.default(view_757, permute_354);  view_757 = permute_354 = None
        add_tensor_8: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_8, arg277_1);  mm_default_8 = arg277_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_758: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 1, 196, 512]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_337: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_333, view_758);  add_333 = view_758 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_97 = torch.ops.aten.var_mean.correction(add_337, [3], correction = 0, keepdim = True)
        getitem_340: "f32[8, 1, 196, 1]" = var_mean_97[0]
        getitem_341: "f32[8, 1, 196, 1]" = var_mean_97[1];  var_mean_97 = None
        sub_143: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_337, getitem_341);  getitem_341 = None
        add_338: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_340, 1e-06);  getitem_340 = None
        rsqrt_97: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
        mul_424: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_97);  sub_143 = rsqrt_97 = None
        mul_425: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_424, arg278_1);  mul_424 = arg278_1 = None
        add_339: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_425, arg279_1);  mul_425 = arg279_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_759: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_339, [1568, 512]);  add_339 = None
        permute_355: "f32[512, 1536]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_759, permute_355);  view_759 = permute_355 = None
        add_tensor_7: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_7, arg281_1);  mm_default_7 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_760: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 1, 196, 1536]);  add_tensor_7 = None
        view_761: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_760, [8, 1, 196, 3, 16, 32]);  view_760 = None
        permute_356: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_761, [3, 0, 4, 1, 2, 5]);  view_761 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_46 = torch.ops.aten.unbind.int(permute_356);  permute_356 = None
        getitem_342: "f32[8, 16, 1, 196, 32]" = unbind_46[0]
        getitem_343: "f32[8, 16, 1, 196, 32]" = unbind_46[1]
        getitem_344: "f32[8, 16, 1, 196, 32]" = unbind_46[2];  unbind_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_426: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_342, 0.42044820762685725);  getitem_342 = None
        expand_184: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_426, [8, 16, 1, 196, 32]);  mul_426 = None
        clone_331: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
        view_762: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_331, [128, 196, 32]);  clone_331 = None
        permute_357: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_343, [0, 1, 2, 4, 3]);  getitem_343 = None
        mul_427: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_357, 0.42044820762685725);  permute_357 = None
        expand_185: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_427, [8, 16, 1, 32, 196]);  mul_427 = None
        clone_332: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_763: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_332, [128, 32, 196]);  clone_332 = None
        bmm_92: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_762, view_763);  view_762 = view_763 = None
        view_764: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_92, [8, 16, 1, 196, 196]);  bmm_92 = None
        eq_46: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_764, -inf)
        logical_not_92: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_46);  eq_46 = None
        any_47: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_92, -1, True);  logical_not_92 = None
        logical_not_93: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_47);  any_47 = None
        full_default_22: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_46: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_764, [-1], True)
        sub_144: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_764, amax_46);  view_764 = amax_46 = None
        exp_46: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_144);  sub_144 = None
        sum_47: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_46: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
        where_46: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_93, full_default_22, div_46);  logical_not_93 = full_default_22 = div_46 = None
        expand_186: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_46, [8, 16, 1, 196, 196]);  where_46 = None
        view_765: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_186, [128, 196, 196]);  expand_186 = None
        expand_187: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_344, [8, 16, 1, 196, 32]);  getitem_344 = None
        clone_333: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_766: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_333, [128, 196, 32]);  clone_333 = None
        bmm_93: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_765, view_766);  view_765 = view_766 = None
        view_767: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_93, [8, 16, 1, 196, 32]);  bmm_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_358: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_767, [0, 2, 3, 4, 1]);  view_767 = None
        clone_334: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_768: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_334, [8, 1, 196, 512]);  clone_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_769: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_768, [1568, 512]);  view_768 = None
        permute_359: "f32[512, 512]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[1568, 512]" = torch.ops.aten.mm.default(view_769, permute_359);  view_769 = permute_359 = None
        add_tensor_6: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_6, arg283_1);  mm_default_6 = arg283_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_770: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 1, 196, 512]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_340: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_337, view_770);  add_337 = view_770 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_98 = torch.ops.aten.var_mean.correction(add_340, [3], correction = 0, keepdim = True)
        getitem_345: "f32[8, 1, 196, 1]" = var_mean_98[0]
        getitem_346: "f32[8, 1, 196, 1]" = var_mean_98[1];  var_mean_98 = None
        sub_145: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_340, getitem_346);  getitem_346 = None
        add_341: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_345, 1e-06);  getitem_345 = None
        rsqrt_98: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        mul_428: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_98);  sub_145 = rsqrt_98 = None
        mul_429: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_428, arg284_1);  mul_428 = arg284_1 = None
        add_342: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_429, arg285_1);  mul_429 = arg285_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_771: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_342, [1568, 512]);  add_342 = None
        permute_360: "f32[512, 2048]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_771, permute_360);  view_771 = permute_360 = None
        add_tensor_5: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_5, arg287_1);  mm_default_5 = arg287_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_772: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 1, 196, 2048]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_430: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_772, 0.5)
        mul_431: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_772, 0.7071067811865476);  view_772 = None
        erf_46: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_431);  mul_431 = None
        add_343: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_432: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_430, add_343);  mul_430 = add_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_773: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_432, [1568, 2048]);  mul_432 = None
        permute_361: "f32[2048, 512]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[1568, 512]" = torch.ops.aten.mm.default(view_773, permute_361);  view_773 = permute_361 = None
        add_tensor_4: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_4, arg289_1);  mm_default_4 = arg289_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_774: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 1, 196, 512]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_344: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_340, view_774);  add_340 = view_774 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_99 = torch.ops.aten.var_mean.correction(add_344, [3], correction = 0, keepdim = True)
        getitem_347: "f32[8, 1, 196, 1]" = var_mean_99[0]
        getitem_348: "f32[8, 1, 196, 1]" = var_mean_99[1];  var_mean_99 = None
        sub_146: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_344, getitem_348);  getitem_348 = None
        add_345: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_347, 1e-06);  getitem_347 = None
        rsqrt_99: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
        mul_433: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_99);  sub_146 = rsqrt_99 = None
        mul_434: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_433, arg290_1);  mul_433 = arg290_1 = None
        add_346: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_434, arg291_1);  mul_434 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_775: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_346, [1568, 512]);  add_346 = None
        permute_362: "f32[512, 1536]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[1568, 1536]" = torch.ops.aten.mm.default(view_775, permute_362);  view_775 = permute_362 = None
        add_tensor_3: "f32[1568, 1536]" = torch.ops.aten.add.Tensor(mm_default_3, arg293_1);  mm_default_3 = arg293_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:65 in forward, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        view_776: "f32[8, 1, 196, 1536]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 1, 196, 1536]);  add_tensor_3 = None
        view_777: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.reshape.default(view_776, [8, 1, 196, 3, 16, 32]);  view_776 = None
        permute_363: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_777, [3, 0, 4, 1, 2, 5]);  view_777 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:66 in forward, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        unbind_47 = torch.ops.aten.unbind.int(permute_363);  permute_363 = None
        getitem_349: "f32[8, 16, 1, 196, 32]" = unbind_47[0]
        getitem_350: "f32[8, 16, 1, 196, 32]" = unbind_47[1]
        getitem_351: "f32[8, 16, 1, 196, 32]" = unbind_47[2];  unbind_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:69 in forward, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        mul_435: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(getitem_349, 0.42044820762685725);  getitem_349 = None
        expand_188: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(mul_435, [8, 16, 1, 196, 32]);  mul_435 = None
        clone_338: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_188, memory_format = torch.contiguous_format);  expand_188 = None
        view_778: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_338, [128, 196, 32]);  clone_338 = None
        permute_364: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.permute.default(getitem_350, [0, 1, 2, 4, 3]);  getitem_350 = None
        mul_436: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(permute_364, 0.42044820762685725);  permute_364 = None
        expand_189: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.expand.default(mul_436, [8, 16, 1, 32, 196]);  mul_436 = None
        clone_339: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_779: "f32[128, 32, 196]" = torch.ops.aten.reshape.default(clone_339, [128, 32, 196]);  clone_339 = None
        bmm_94: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_778, view_779);  view_778 = view_779 = None
        view_780: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.reshape.default(bmm_94, [8, 16, 1, 196, 196]);  bmm_94 = None
        eq_47: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.eq.Scalar(view_780, -inf)
        logical_not_94: "b8[8, 16, 1, 196, 196]" = torch.ops.aten.logical_not.default(eq_47);  eq_47 = None
        any_48: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.any.dim(logical_not_94, -1, True);  logical_not_94 = None
        logical_not_95: "b8[8, 16, 1, 196, 1]" = torch.ops.aten.logical_not.default(any_48);  any_48 = None
        full_default_23: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.full.default([8, 16, 1, 196, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_47: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.amax.default(view_780, [-1], True)
        sub_147: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(view_780, amax_47);  view_780 = amax_47 = None
        exp_47: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.exp.default(sub_147);  sub_147 = None
        sum_48: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_47: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.div.Tensor(exp_47, sum_48);  exp_47 = sum_48 = None
        where_47: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.where.self(logical_not_95, full_default_23, div_47);  logical_not_95 = full_default_23 = div_47 = None
        expand_190: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.expand.default(where_47, [8, 16, 1, 196, 196]);  where_47 = None
        view_781: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(expand_190, [128, 196, 196]);  expand_190 = None
        expand_191: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.expand.default(getitem_351, [8, 16, 1, 196, 32]);  getitem_351 = None
        clone_340: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(expand_191, memory_format = torch.contiguous_format);  expand_191 = None
        view_782: "f32[128, 196, 32]" = torch.ops.aten.reshape.default(clone_340, [128, 196, 32]);  clone_340 = None
        bmm_95: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_781, view_782);  view_781 = view_782 = None
        view_783: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.reshape.default(bmm_95, [8, 16, 1, 196, 32]);  bmm_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:78 in forward, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        permute_365: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.permute.default(view_783, [0, 2, 3, 4, 1]);  view_783 = None
        clone_341: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_784: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(clone_341, [8, 1, 196, 512]);  clone_341 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_785: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_784, [1568, 512]);  view_784 = None
        permute_366: "f32[512, 512]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[1568, 512]" = torch.ops.aten.mm.default(view_785, permute_366);  view_785 = permute_366 = None
        add_tensor_2: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default_2, arg295_1);  mm_default_2 = arg295_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
        view_786: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 1, 196, 512]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:123 in forward, code: x = x + self.drop_path(self.attn(y))
        add_347: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_344, view_786);  add_344 = view_786 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_100 = torch.ops.aten.var_mean.correction(add_347, [3], correction = 0, keepdim = True)
        getitem_352: "f32[8, 1, 196, 1]" = var_mean_100[0]
        getitem_353: "f32[8, 1, 196, 1]" = var_mean_100[1];  var_mean_100 = None
        sub_148: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(add_347, getitem_353);  getitem_353 = None
        add_348: "f32[8, 1, 196, 1]" = torch.ops.aten.add.Tensor(getitem_352, 1e-06);  getitem_352 = None
        rsqrt_100: "f32[8, 1, 196, 1]" = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        mul_437: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_100);  sub_148 = rsqrt_100 = None
        mul_438: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_437, arg296_1);  mul_437 = arg296_1 = None
        add_349: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(mul_438, arg297_1);  mul_438 = arg297_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_787: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_349, [1568, 512]);  add_349 = None
        permute_367: "f32[512, 2048]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_787, permute_367);  view_787 = permute_367 = None
        add_tensor_1: "f32[1568, 2048]" = torch.ops.aten.add.Tensor(mm_default_1, arg299_1);  mm_default_1 = arg299_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_788: "f32[8, 1, 196, 2048]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 1, 196, 2048]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_439: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_788, 0.5)
        mul_440: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_788, 0.7071067811865476);  view_788 = None
        erf_47: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_440);  mul_440 = None
        add_350: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_441: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_439, add_350);  mul_439 = add_350 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_789: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_441, [1568, 2048]);  mul_441 = None
        permute_368: "f32[2048, 512]" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[1568, 512]" = torch.ops.aten.mm.default(view_789, permute_368);  view_789 = permute_368 = None
        add_tensor: "f32[1568, 512]" = torch.ops.aten.add.Tensor(mm_default, arg301_1);  mm_default = arg301_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_790: "f32[8, 1, 196, 512]" = torch.ops.aten.reshape.default(add_tensor, [8, 1, 196, 512]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:124 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_351: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_347, view_790);  add_347 = view_790 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:174 in deblockify, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
        view_791: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.reshape.default(add_351, [8, 1, 1, 14, 14, 512]);  add_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:175 in deblockify, code: x = x.transpose(2, 3).reshape(B, height, width, C)
        permute_369: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.permute.default(view_791, [0, 1, 3, 2, 4, 5]);  view_791 = None
        view_792: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(permute_369, [8, 14, 14, 512]);  permute_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:242 in forward, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
        permute_370: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_792, [0, 3, 1, 2]);  view_792 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:427 in forward_features, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_371: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(permute_370, [0, 2, 3, 1]);  permute_370 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_101 = torch.ops.aten.var_mean.correction(permute_371, [3], correction = 0, keepdim = True)
        getitem_354: "f32[8, 14, 14, 1]" = var_mean_101[0]
        getitem_355: "f32[8, 14, 14, 1]" = var_mean_101[1];  var_mean_101 = None
        sub_149: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_371, getitem_355);  permute_371 = getitem_355 = None
        add_352: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_354, 1e-06);  getitem_354 = None
        rsqrt_101: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_352);  add_352 = None
        mul_442: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_101);  sub_149 = rsqrt_101 = None
        mul_443: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_442, arg302_1);  mul_442 = arg302_1 = None
        add_353: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_443, arg303_1);  mul_443 = arg303_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:427 in forward_features, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        permute_372: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(add_353, [0, 3, 1, 2]);  add_353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(permute_372, [-1, -2], True);  permute_372 = None
        as_strided_1: "f32[8, 512, 1, 1]" = torch.ops.aten.as_strided.default(mean_1, [8, 512, 1, 1], [512, 1, 512, 512]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_793: "f32[8, 512]" = torch.ops.aten.reshape.default(as_strided_1, [8, 512]);  as_strided_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nest.py:433 in forward_head, code: return x if pre_logits else self.head(x)
        permute_373: "f32[512, 1000]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        addmm_193: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg305_1, view_793, permute_373);  arg305_1 = view_793 = permute_373 = None
        return (addmm_193,)
        