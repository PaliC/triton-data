class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[1, 24, 4, 4]", arg2_1: "f32[24, 3, 7, 7]", arg3_1: "f32[24]", arg4_1: "f32[384]", arg5_1: "f32[384]", arg6_1: "f32[384, 384]", arg7_1: "f32[384]", arg8_1: "f32[384]", arg9_1: "f32[384]", arg10_1: "f32[1, 1, 384]", arg11_1: "f32[1, 197, 384]", arg12_1: "f32[24]", arg13_1: "f32[24]", arg14_1: "f32[48, 24]", arg15_1: "f32[24, 24]", arg16_1: "f32[24, 24]", arg17_1: "f32[24]", arg18_1: "f32[24]", arg19_1: "f32[24]", arg20_1: "f32[96, 24]", arg21_1: "f32[96]", arg22_1: "f32[24, 96]", arg23_1: "f32[24]", arg24_1: "f32[24]", arg25_1: "f32[24]", arg26_1: "f32[384, 384]", arg27_1: "f32[384]", arg28_1: "f32[384]", arg29_1: "f32[384]", arg30_1: "f32[768, 384]", arg31_1: "f32[384, 384]", arg32_1: "f32[384, 384]", arg33_1: "f32[384]", arg34_1: "f32[384]", arg35_1: "f32[384]", arg36_1: "f32[1536, 384]", arg37_1: "f32[1536]", arg38_1: "f32[384, 1536]", arg39_1: "f32[384]", arg40_1: "f32[24]", arg41_1: "f32[24]", arg42_1: "f32[48, 24]", arg43_1: "f32[24, 24]", arg44_1: "f32[24, 24]", arg45_1: "f32[24]", arg46_1: "f32[24]", arg47_1: "f32[24]", arg48_1: "f32[96, 24]", arg49_1: "f32[96]", arg50_1: "f32[24, 96]", arg51_1: "f32[24]", arg52_1: "f32[24]", arg53_1: "f32[24]", arg54_1: "f32[384, 384]", arg55_1: "f32[384]", arg56_1: "f32[384]", arg57_1: "f32[384]", arg58_1: "f32[768, 384]", arg59_1: "f32[384, 384]", arg60_1: "f32[384, 384]", arg61_1: "f32[384]", arg62_1: "f32[384]", arg63_1: "f32[384]", arg64_1: "f32[1536, 384]", arg65_1: "f32[1536]", arg66_1: "f32[384, 1536]", arg67_1: "f32[384]", arg68_1: "f32[24]", arg69_1: "f32[24]", arg70_1: "f32[48, 24]", arg71_1: "f32[24, 24]", arg72_1: "f32[24, 24]", arg73_1: "f32[24]", arg74_1: "f32[24]", arg75_1: "f32[24]", arg76_1: "f32[96, 24]", arg77_1: "f32[96]", arg78_1: "f32[24, 96]", arg79_1: "f32[24]", arg80_1: "f32[24]", arg81_1: "f32[24]", arg82_1: "f32[384, 384]", arg83_1: "f32[384]", arg84_1: "f32[384]", arg85_1: "f32[384]", arg86_1: "f32[768, 384]", arg87_1: "f32[384, 384]", arg88_1: "f32[384, 384]", arg89_1: "f32[384]", arg90_1: "f32[384]", arg91_1: "f32[384]", arg92_1: "f32[1536, 384]", arg93_1: "f32[1536]", arg94_1: "f32[384, 1536]", arg95_1: "f32[384]", arg96_1: "f32[24]", arg97_1: "f32[24]", arg98_1: "f32[48, 24]", arg99_1: "f32[24, 24]", arg100_1: "f32[24, 24]", arg101_1: "f32[24]", arg102_1: "f32[24]", arg103_1: "f32[24]", arg104_1: "f32[96, 24]", arg105_1: "f32[96]", arg106_1: "f32[24, 96]", arg107_1: "f32[24]", arg108_1: "f32[24]", arg109_1: "f32[24]", arg110_1: "f32[384, 384]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[384]", arg114_1: "f32[768, 384]", arg115_1: "f32[384, 384]", arg116_1: "f32[384, 384]", arg117_1: "f32[384]", arg118_1: "f32[384]", arg119_1: "f32[384]", arg120_1: "f32[1536, 384]", arg121_1: "f32[1536]", arg122_1: "f32[384, 1536]", arg123_1: "f32[384]", arg124_1: "f32[24]", arg125_1: "f32[24]", arg126_1: "f32[48, 24]", arg127_1: "f32[24, 24]", arg128_1: "f32[24, 24]", arg129_1: "f32[24]", arg130_1: "f32[24]", arg131_1: "f32[24]", arg132_1: "f32[96, 24]", arg133_1: "f32[96]", arg134_1: "f32[24, 96]", arg135_1: "f32[24]", arg136_1: "f32[24]", arg137_1: "f32[24]", arg138_1: "f32[384, 384]", arg139_1: "f32[384]", arg140_1: "f32[384]", arg141_1: "f32[384]", arg142_1: "f32[768, 384]", arg143_1: "f32[384, 384]", arg144_1: "f32[384, 384]", arg145_1: "f32[384]", arg146_1: "f32[384]", arg147_1: "f32[384]", arg148_1: "f32[1536, 384]", arg149_1: "f32[1536]", arg150_1: "f32[384, 1536]", arg151_1: "f32[384]", arg152_1: "f32[24]", arg153_1: "f32[24]", arg154_1: "f32[48, 24]", arg155_1: "f32[24, 24]", arg156_1: "f32[24, 24]", arg157_1: "f32[24]", arg158_1: "f32[24]", arg159_1: "f32[24]", arg160_1: "f32[96, 24]", arg161_1: "f32[96]", arg162_1: "f32[24, 96]", arg163_1: "f32[24]", arg164_1: "f32[24]", arg165_1: "f32[24]", arg166_1: "f32[384, 384]", arg167_1: "f32[384]", arg168_1: "f32[384]", arg169_1: "f32[384]", arg170_1: "f32[768, 384]", arg171_1: "f32[384, 384]", arg172_1: "f32[384, 384]", arg173_1: "f32[384]", arg174_1: "f32[384]", arg175_1: "f32[384]", arg176_1: "f32[1536, 384]", arg177_1: "f32[1536]", arg178_1: "f32[384, 1536]", arg179_1: "f32[384]", arg180_1: "f32[24]", arg181_1: "f32[24]", arg182_1: "f32[48, 24]", arg183_1: "f32[24, 24]", arg184_1: "f32[24, 24]", arg185_1: "f32[24]", arg186_1: "f32[24]", arg187_1: "f32[24]", arg188_1: "f32[96, 24]", arg189_1: "f32[96]", arg190_1: "f32[24, 96]", arg191_1: "f32[24]", arg192_1: "f32[24]", arg193_1: "f32[24]", arg194_1: "f32[384, 384]", arg195_1: "f32[384]", arg196_1: "f32[384]", arg197_1: "f32[384]", arg198_1: "f32[768, 384]", arg199_1: "f32[384, 384]", arg200_1: "f32[384, 384]", arg201_1: "f32[384]", arg202_1: "f32[384]", arg203_1: "f32[384]", arg204_1: "f32[1536, 384]", arg205_1: "f32[1536]", arg206_1: "f32[384, 1536]", arg207_1: "f32[384]", arg208_1: "f32[24]", arg209_1: "f32[24]", arg210_1: "f32[48, 24]", arg211_1: "f32[24, 24]", arg212_1: "f32[24, 24]", arg213_1: "f32[24]", arg214_1: "f32[24]", arg215_1: "f32[24]", arg216_1: "f32[96, 24]", arg217_1: "f32[96]", arg218_1: "f32[24, 96]", arg219_1: "f32[24]", arg220_1: "f32[24]", arg221_1: "f32[24]", arg222_1: "f32[384, 384]", arg223_1: "f32[384]", arg224_1: "f32[384]", arg225_1: "f32[384]", arg226_1: "f32[768, 384]", arg227_1: "f32[384, 384]", arg228_1: "f32[384, 384]", arg229_1: "f32[384]", arg230_1: "f32[384]", arg231_1: "f32[384]", arg232_1: "f32[1536, 384]", arg233_1: "f32[1536]", arg234_1: "f32[384, 1536]", arg235_1: "f32[384]", arg236_1: "f32[24]", arg237_1: "f32[24]", arg238_1: "f32[48, 24]", arg239_1: "f32[24, 24]", arg240_1: "f32[24, 24]", arg241_1: "f32[24]", arg242_1: "f32[24]", arg243_1: "f32[24]", arg244_1: "f32[96, 24]", arg245_1: "f32[96]", arg246_1: "f32[24, 96]", arg247_1: "f32[24]", arg248_1: "f32[24]", arg249_1: "f32[24]", arg250_1: "f32[384, 384]", arg251_1: "f32[384]", arg252_1: "f32[384]", arg253_1: "f32[384]", arg254_1: "f32[768, 384]", arg255_1: "f32[384, 384]", arg256_1: "f32[384, 384]", arg257_1: "f32[384]", arg258_1: "f32[384]", arg259_1: "f32[384]", arg260_1: "f32[1536, 384]", arg261_1: "f32[1536]", arg262_1: "f32[384, 1536]", arg263_1: "f32[384]", arg264_1: "f32[24]", arg265_1: "f32[24]", arg266_1: "f32[48, 24]", arg267_1: "f32[24, 24]", arg268_1: "f32[24, 24]", arg269_1: "f32[24]", arg270_1: "f32[24]", arg271_1: "f32[24]", arg272_1: "f32[96, 24]", arg273_1: "f32[96]", arg274_1: "f32[24, 96]", arg275_1: "f32[24]", arg276_1: "f32[24]", arg277_1: "f32[24]", arg278_1: "f32[384, 384]", arg279_1: "f32[384]", arg280_1: "f32[384]", arg281_1: "f32[384]", arg282_1: "f32[768, 384]", arg283_1: "f32[384, 384]", arg284_1: "f32[384, 384]", arg285_1: "f32[384]", arg286_1: "f32[384]", arg287_1: "f32[384]", arg288_1: "f32[1536, 384]", arg289_1: "f32[1536]", arg290_1: "f32[384, 1536]", arg291_1: "f32[384]", arg292_1: "f32[24]", arg293_1: "f32[24]", arg294_1: "f32[48, 24]", arg295_1: "f32[24, 24]", arg296_1: "f32[24, 24]", arg297_1: "f32[24]", arg298_1: "f32[24]", arg299_1: "f32[24]", arg300_1: "f32[96, 24]", arg301_1: "f32[96]", arg302_1: "f32[24, 96]", arg303_1: "f32[24]", arg304_1: "f32[24]", arg305_1: "f32[24]", arg306_1: "f32[384, 384]", arg307_1: "f32[384]", arg308_1: "f32[384]", arg309_1: "f32[384]", arg310_1: "f32[768, 384]", arg311_1: "f32[384, 384]", arg312_1: "f32[384, 384]", arg313_1: "f32[384]", arg314_1: "f32[384]", arg315_1: "f32[384]", arg316_1: "f32[1536, 384]", arg317_1: "f32[1536]", arg318_1: "f32[384, 1536]", arg319_1: "f32[384]", arg320_1: "f32[24]", arg321_1: "f32[24]", arg322_1: "f32[48, 24]", arg323_1: "f32[24, 24]", arg324_1: "f32[24, 24]", arg325_1: "f32[24]", arg326_1: "f32[24]", arg327_1: "f32[24]", arg328_1: "f32[96, 24]", arg329_1: "f32[96]", arg330_1: "f32[24, 96]", arg331_1: "f32[24]", arg332_1: "f32[24]", arg333_1: "f32[24]", arg334_1: "f32[384, 384]", arg335_1: "f32[384]", arg336_1: "f32[384]", arg337_1: "f32[384]", arg338_1: "f32[768, 384]", arg339_1: "f32[384, 384]", arg340_1: "f32[384, 384]", arg341_1: "f32[384]", arg342_1: "f32[384]", arg343_1: "f32[384]", arg344_1: "f32[1536, 384]", arg345_1: "f32[1536]", arg346_1: "f32[384, 1536]", arg347_1: "f32[384]", arg348_1: "f32[384]", arg349_1: "f32[384]", arg350_1: "f32[1000, 384]", arg351_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:182 in forward, code: x = self.proj(x)
        convolution_1: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(arg0_1, arg2_1, arg3_1, [4, 4], [3, 3], [1, 1], False, [0, 0], 1);  arg0_1 = arg2_1 = arg3_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:183 in forward, code: x = self.unfold(x)
        iota_4: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        iota_5: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_7: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        add_214: "i64[4, 14]" = torch.ops.aten.add.Tensor(unsqueeze_6, unsqueeze_7);  unsqueeze_6 = unsqueeze_7 = None
        unsqueeze_10: "i64[4, 14, 1]" = torch.ops.aten.unsqueeze.default(add_214, -1);  add_214 = None
        unsqueeze_11: "i64[4, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        iota_6: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_8: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = None
        iota_7: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_9: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        add_215: "i64[4, 14]" = torch.ops.aten.add.Tensor(unsqueeze_8, unsqueeze_9);  unsqueeze_8 = unsqueeze_9 = None
        index_1: "f32[8, 24, 4, 14, 4, 14]" = torch.ops.aten.index.Tensor(convolution_1, [None, None, unsqueeze_11, add_215]);  convolution_1 = unsqueeze_11 = add_215 = None
        permute_233: "f32[8, 24, 4, 4, 14, 14]" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
        clone_185: "f32[8, 24, 4, 4, 14, 14]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
        view_498: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(clone_185, [8, 384, 196]);  clone_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:184 in forward, code: x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
        permute_234: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view_498, [0, 2, 1]);  view_498 = None
        clone_186: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_499: "f32[1568, 24, 4, 4]" = torch.ops.aten.reshape.default(clone_186, [1568, 24, 4, 4]);  clone_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:185 in forward, code: x = x + pixel_pos
        add_216: "f32[1568, 24, 4, 4]" = torch.ops.aten.add.Tensor(view_499, arg1_1);  view_499 = arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:186 in forward, code: x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
        view_500: "f32[1568, 24, 16]" = torch.ops.aten.reshape.default(add_216, [1568, 24, -1]);  add_216 = None
        permute_235: "f32[1568, 16, 24]" = torch.ops.aten.permute.default(view_500, [0, 2, 1]);  view_500 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:313 in forward_features, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        clone_187: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format)
        view_501: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(clone_187, [8, 196, 384]);  clone_187 = None
        var_mean_63 = torch.ops.aten.var_mean.correction(view_501, [2], correction = 0, keepdim = True)
        getitem_174: "f32[8, 196, 1]" = var_mean_63[0]
        getitem_175: "f32[8, 196, 1]" = var_mean_63[1];  var_mean_63 = None
        sub_87: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(view_501, getitem_175);  view_501 = getitem_175 = None
        add_217: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_63: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        mul_222: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_63);  sub_87 = rsqrt_63 = None
        mul_223: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_222, arg4_1);  mul_222 = arg4_1 = None
        add_218: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_223, arg5_1);  mul_223 = arg5_1 = None
        view_502: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_218, [1568, 384]);  add_218 = None
        permute_236: "f32[384, 384]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_86: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg7_1, view_502, permute_236);  arg7_1 = view_502 = permute_236 = None
        view_503: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm_86, [8, 196, 384]);  addmm_86 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(view_503, [2], correction = 0, keepdim = True)
        getitem_176: "f32[8, 196, 1]" = var_mean_64[0]
        getitem_177: "f32[8, 196, 1]" = var_mean_64[1];  var_mean_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_189: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format)
        var_mean_65 = torch.ops.aten.var_mean.correction(clone_189, [2], correction = 0, keepdim = True)
        getitem_178: "f32[1568, 16, 1]" = var_mean_65[0]
        getitem_179: "f32[1568, 16, 1]" = var_mean_65[1];  var_mean_65 = None
        sub_89: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_189, getitem_179);  clone_189 = getitem_179 = None
        add_222: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_65: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        mul_226: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_65);  sub_89 = rsqrt_65 = None
        mul_227: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_226, arg12_1);  mul_226 = arg12_1 = None
        add_223: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_227, arg13_1);  mul_227 = arg13_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_504: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_223, [25088, 24])
        permute_237: "f32[24, 48]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_48: "f32[25088, 48]" = torch.ops.aten.mm.default(view_504, permute_237);  view_504 = permute_237 = None
        view_505: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_48, [1568, 16, 48]);  mm_48 = None
        view_506: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_505, [1568, 16, 2, 4, 6]);  view_505 = None
        permute_238: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_506, [2, 0, 3, 1, 4]);  view_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_24 = torch.ops.aten.unbind.int(permute_238);  permute_238 = None
        getitem_180: "f32[1568, 4, 16, 6]" = unbind_24[0]
        getitem_181: "f32[1568, 4, 16, 6]" = unbind_24[1];  unbind_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_98: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_180, [1568, 4, 16, 6]);  getitem_180 = None
        clone_190: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_510: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_190, [6272, 16, 6]);  clone_190 = None
        permute_241: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_181, [0, 1, 3, 2]);  getitem_181 = None
        expand_99: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_241, [1568, 4, 6, 16]);  permute_241 = None
        clone_191: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_511: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_191, [6272, 6, 16]);  clone_191 = None
        bmm_48: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_510, view_511);  view_510 = view_511 = None
        view_512: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_48, [1568, 4, 16, 16]);  bmm_48 = None
        
        # No stacktrace found for following nodes
        mul_tensor_46: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_512, 1);  view_512 = None
        amax_default_23: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_46, [-1], True)
        sub_tensor_23: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_46, amax_default_23);  mul_tensor_46 = amax_default_23 = None
        mul_tensor_47: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_23, 0.408248290463863);  sub_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_24: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_47);  mul_tensor_47 = None
        sum_25: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_100: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_24, [1568, 4, 16, 16]);  div_24 = None
        view_513: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_100, [6272, 16, 16]);  expand_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_507: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_223, [25088, 24]);  add_223 = None
        permute_239: "f32[24, 24]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_49: "f32[25088, 24]" = torch.ops.aten.mm.default(view_507, permute_239);  view_507 = permute_239 = None
        view_508: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_49, [1568, 16, 24]);  mm_49 = None
        view_509: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_508, [1568, 16, 4, -1]);  view_508 = None
        permute_240: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_101: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_240, [1568, 4, 16, 6]);  permute_240 = None
        clone_192: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_514: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_192, [6272, 16, 6]);  clone_192 = None
        bmm_49: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_513, view_514);  view_513 = view_514 = None
        view_515: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_49, [1568, 4, 16, 6]);  bmm_49 = None
        permute_242: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
        clone_193: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_516: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_193, [1568, 16, 24]);  clone_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_517: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_516, [25088, 24]);  view_516 = None
        permute_243: "f32[24, 24]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        
        # No stacktrace found for following nodes
        mm_default_83: "f32[25088, 24]" = torch.ops.aten.mm.default(view_517, permute_243);  view_517 = permute_243 = None
        add_tensor_83: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_83, arg17_1);  mm_default_83 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_518: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_83, [1568, 16, 24]);  add_tensor_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_224: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(permute_235, view_518);  permute_235 = view_518 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_194: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_224, memory_format = torch.contiguous_format)
        var_mean_66 = torch.ops.aten.var_mean.correction(clone_194, [2], correction = 0, keepdim = True)
        getitem_182: "f32[1568, 16, 1]" = var_mean_66[0]
        getitem_183: "f32[1568, 16, 1]" = var_mean_66[1];  var_mean_66 = None
        sub_91: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_194, getitem_183);  clone_194 = getitem_183 = None
        add_225: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_66: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_229: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_66);  sub_91 = rsqrt_66 = None
        mul_230: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_229, arg18_1);  mul_229 = arg18_1 = None
        add_226: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_230, arg19_1);  mul_230 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_519: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_226, [25088, 24]);  add_226 = None
        permute_244: "f32[24, 96]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        
        # No stacktrace found for following nodes
        mm_default_82: "f32[25088, 96]" = torch.ops.aten.mm.default(view_519, permute_244);  view_519 = permute_244 = None
        add_tensor_82: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_82, arg21_1);  mm_default_82 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_520: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_82, [1568, 16, 96]);  add_tensor_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_231: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_520, 0.5)
        mul_232: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_520, 0.7071067811865476);  view_520 = None
        erf_24: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_227: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_233: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_231, add_227);  mul_231 = add_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_521: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_233, [25088, 96]);  mul_233 = None
        permute_245: "f32[96, 24]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        
        # No stacktrace found for following nodes
        mm_default_81: "f32[25088, 24]" = torch.ops.aten.mm.default(view_521, permute_245);  view_521 = permute_245 = None
        add_tensor_81: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_81, arg23_1);  mm_default_81 = arg23_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_522: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_81, [1568, 16, 24]);  add_tensor_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_228: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_224, view_522);  add_224 = view_522 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_197: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_228, memory_format = torch.contiguous_format)
        var_mean_67 = torch.ops.aten.var_mean.correction(clone_197, [2], correction = 0, keepdim = True)
        getitem_184: "f32[1568, 16, 1]" = var_mean_67[0]
        getitem_185: "f32[1568, 16, 1]" = var_mean_67[1];  var_mean_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:314 in forward_features, code: patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        expand_97: "f32[8, 1, 384]" = torch.ops.aten.expand.default(arg10_1, [8, -1, -1]);  arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:313 in forward_features, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        sub_88: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(view_503, getitem_177);  view_503 = getitem_177 = None
        add_219: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_64: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_224: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_64);  sub_88 = rsqrt_64 = None
        mul_225: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_224, arg8_1);  mul_224 = arg8_1 = None
        add_220: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_225, arg9_1);  mul_225 = arg9_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:314 in forward_features, code: patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        cat_13: "f32[8, 197, 384]" = torch.ops.aten.cat.default([expand_97, add_220], 1);  expand_97 = add_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:315 in forward_features, code: patch_embed = patch_embed + self.patch_pos
        add_221: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_13, arg11_1);  cat_13 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_55: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_221, 1, 0, 1)
        slice_57: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_221, 1, 1, 9223372036854775807);  add_221 = None
        sub_92: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_197, getitem_185);  clone_197 = getitem_185 = None
        add_229: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_67: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_234: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_67);  sub_92 = rsqrt_67 = None
        mul_235: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_234, arg24_1);  mul_234 = arg24_1 = None
        add_230: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_235, arg25_1);  mul_235 = arg25_1 = None
        view_523: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_230, [8, 196, -1]);  add_230 = None
        view_524: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_523, [1568, 384]);  view_523 = None
        permute_246: "f32[384, 384]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        
        # No stacktrace found for following nodes
        mm_default_80: "f32[1568, 384]" = torch.ops.aten.mm.default(view_524, permute_246);  view_524 = permute_246 = None
        add_tensor_80: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_80, arg27_1);  mm_default_80 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_525: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_80, [8, 196, 384]);  add_tensor_80 = None
        add_231: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_57, view_525);  slice_57 = view_525 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_14: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_55, add_231], 1);  slice_55 = add_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_68 = torch.ops.aten.var_mean.correction(cat_14, [2], correction = 0, keepdim = True)
        getitem_186: "f32[8, 197, 1]" = var_mean_68[0]
        getitem_187: "f32[8, 197, 1]" = var_mean_68[1];  var_mean_68 = None
        sub_93: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_14, getitem_187);  getitem_187 = None
        add_232: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_68: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        mul_236: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_68);  sub_93 = rsqrt_68 = None
        mul_237: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_236, arg28_1);  mul_236 = arg28_1 = None
        add_233: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_237, arg29_1);  mul_237 = arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_526: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_233, [1576, 384])
        permute_247: "f32[384, 768]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_50: "f32[1576, 768]" = torch.ops.aten.mm.default(view_526, permute_247);  view_526 = permute_247 = None
        view_527: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_50, [8, 197, 768]);  mm_50 = None
        view_528: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_527, [8, 197, 2, 6, 64]);  view_527 = None
        permute_248: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_528, [2, 0, 3, 1, 4]);  view_528 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_25 = torch.ops.aten.unbind.int(permute_248);  permute_248 = None
        getitem_188: "f32[8, 6, 197, 64]" = unbind_25[0]
        getitem_189: "f32[8, 6, 197, 64]" = unbind_25[1];  unbind_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_102: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_188, [8, 6, 197, 64]);  getitem_188 = None
        clone_198: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
        view_532: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_198, [48, 197, 64]);  clone_198 = None
        permute_251: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_189, [0, 1, 3, 2]);  getitem_189 = None
        expand_103: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_251, [8, 6, 64, 197]);  permute_251 = None
        clone_199: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_533: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_199, [48, 64, 197]);  clone_199 = None
        bmm_50: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_532, view_533);  view_532 = view_533 = None
        view_534: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_50, [8, 6, 197, 197]);  bmm_50 = None
        
        # No stacktrace found for following nodes
        mul_tensor_44: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_534, 1);  view_534 = None
        amax_default_22: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_44, [-1], True)
        sub_tensor_22: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_44, amax_default_22);  mul_tensor_44 = amax_default_22 = None
        mul_tensor_45: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_22, 0.125);  sub_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_25: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_45);  mul_tensor_45 = None
        sum_26: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_25: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_104: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_25, [8, 6, 197, 197]);  div_25 = None
        view_535: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_104, [48, 197, 197]);  expand_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_529: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_233, [1576, 384]);  add_233 = None
        permute_249: "f32[384, 384]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_51: "f32[1576, 384]" = torch.ops.aten.mm.default(view_529, permute_249);  view_529 = permute_249 = None
        view_530: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_51, [8, 197, 384]);  mm_51 = None
        view_531: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_530, [8, 197, 6, -1]);  view_530 = None
        permute_250: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_105: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_250, [8, 6, 197, 64]);  permute_250 = None
        clone_200: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_536: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_200, [48, 197, 64]);  clone_200 = None
        bmm_51: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_535, view_536);  view_535 = view_536 = None
        view_537: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_51, [8, 6, 197, 64]);  bmm_51 = None
        permute_252: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
        clone_201: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_538: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_201, [8, 197, 384]);  clone_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_539: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_538, [1576, 384]);  view_538 = None
        permute_253: "f32[384, 384]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        
        # No stacktrace found for following nodes
        mm_default_79: "f32[1576, 384]" = torch.ops.aten.mm.default(view_539, permute_253);  view_539 = permute_253 = None
        add_tensor_79: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_79, arg33_1);  mm_default_79 = arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_540: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_79, [8, 197, 384]);  add_tensor_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_234: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_14, view_540);  cat_14 = view_540 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_69 = torch.ops.aten.var_mean.correction(add_234, [2], correction = 0, keepdim = True)
        getitem_190: "f32[8, 197, 1]" = var_mean_69[0]
        getitem_191: "f32[8, 197, 1]" = var_mean_69[1];  var_mean_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_204: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_228, memory_format = torch.contiguous_format)
        var_mean_70 = torch.ops.aten.var_mean.correction(clone_204, [2], correction = 0, keepdim = True)
        getitem_192: "f32[1568, 16, 1]" = var_mean_70[0]
        getitem_193: "f32[1568, 16, 1]" = var_mean_70[1];  var_mean_70 = None
        sub_96: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_204, getitem_193);  clone_204 = getitem_193 = None
        add_239: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_70: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_244: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_70);  sub_96 = rsqrt_70 = None
        mul_245: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_244, arg40_1);  mul_244 = arg40_1 = None
        add_240: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_245, arg41_1);  mul_245 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_545: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_240, [25088, 24])
        permute_256: "f32[24, 48]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_52: "f32[25088, 48]" = torch.ops.aten.mm.default(view_545, permute_256);  view_545 = permute_256 = None
        view_546: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_52, [1568, 16, 48]);  mm_52 = None
        view_547: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_546, [1568, 16, 2, 4, 6]);  view_546 = None
        permute_257: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_547, [2, 0, 3, 1, 4]);  view_547 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_26 = torch.ops.aten.unbind.int(permute_257);  permute_257 = None
        getitem_194: "f32[1568, 4, 16, 6]" = unbind_26[0]
        getitem_195: "f32[1568, 4, 16, 6]" = unbind_26[1];  unbind_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_106: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_194, [1568, 4, 16, 6]);  getitem_194 = None
        clone_205: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_551: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_205, [6272, 16, 6]);  clone_205 = None
        permute_260: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_195, [0, 1, 3, 2]);  getitem_195 = None
        expand_107: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_260, [1568, 4, 6, 16]);  permute_260 = None
        clone_206: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_552: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_206, [6272, 6, 16]);  clone_206 = None
        bmm_52: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_551, view_552);  view_551 = view_552 = None
        view_553: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_52, [1568, 4, 16, 16]);  bmm_52 = None
        
        # No stacktrace found for following nodes
        mul_tensor_42: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_553, 1);  view_553 = None
        amax_default_21: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_42, [-1], True)
        sub_tensor_21: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_42, amax_default_21);  mul_tensor_42 = amax_default_21 = None
        mul_tensor_43: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_21, 0.408248290463863);  sub_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_26: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_43);  mul_tensor_43 = None
        sum_27: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_26: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_108: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_26, [1568, 4, 16, 16]);  div_26 = None
        view_554: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_108, [6272, 16, 16]);  expand_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_548: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_240, [25088, 24]);  add_240 = None
        permute_258: "f32[24, 24]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_53: "f32[25088, 24]" = torch.ops.aten.mm.default(view_548, permute_258);  view_548 = permute_258 = None
        view_549: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_53, [1568, 16, 24]);  mm_53 = None
        view_550: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_549, [1568, 16, 4, -1]);  view_549 = None
        permute_259: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_109: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_259, [1568, 4, 16, 6]);  permute_259 = None
        clone_207: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_555: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_207, [6272, 16, 6]);  clone_207 = None
        bmm_53: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_554, view_555);  view_554 = view_555 = None
        view_556: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_53, [1568, 4, 16, 6]);  bmm_53 = None
        permute_261: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        clone_208: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_557: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_208, [1568, 16, 24]);  clone_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_558: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_557, [25088, 24]);  view_557 = None
        permute_262: "f32[24, 24]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        
        # No stacktrace found for following nodes
        mm_default_78: "f32[25088, 24]" = torch.ops.aten.mm.default(view_558, permute_262);  view_558 = permute_262 = None
        add_tensor_78: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_78, arg45_1);  mm_default_78 = arg45_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_559: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_78, [1568, 16, 24]);  add_tensor_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_241: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_228, view_559);  add_228 = view_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_209: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_241, memory_format = torch.contiguous_format)
        var_mean_71 = torch.ops.aten.var_mean.correction(clone_209, [2], correction = 0, keepdim = True)
        getitem_196: "f32[1568, 16, 1]" = var_mean_71[0]
        getitem_197: "f32[1568, 16, 1]" = var_mean_71[1];  var_mean_71 = None
        sub_98: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_209, getitem_197);  clone_209 = getitem_197 = None
        add_242: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_71: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
        mul_247: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_71);  sub_98 = rsqrt_71 = None
        mul_248: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_247, arg46_1);  mul_247 = arg46_1 = None
        add_243: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_248, arg47_1);  mul_248 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_560: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_243, [25088, 24]);  add_243 = None
        permute_263: "f32[24, 96]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        
        # No stacktrace found for following nodes
        mm_default_77: "f32[25088, 96]" = torch.ops.aten.mm.default(view_560, permute_263);  view_560 = permute_263 = None
        add_tensor_77: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_77, arg49_1);  mm_default_77 = arg49_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_561: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_77, [1568, 16, 96]);  add_tensor_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_249: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_561, 0.5)
        mul_250: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_561, 0.7071067811865476);  view_561 = None
        erf_26: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_250);  mul_250 = None
        add_244: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_251: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_249, add_244);  mul_249 = add_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_562: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_251, [25088, 96]);  mul_251 = None
        permute_264: "f32[96, 24]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        
        # No stacktrace found for following nodes
        mm_default_76: "f32[25088, 24]" = torch.ops.aten.mm.default(view_562, permute_264);  view_562 = permute_264 = None
        add_tensor_76: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_76, arg51_1);  mm_default_76 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_563: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_76, [1568, 16, 24]);  add_tensor_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_245: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_241, view_563);  add_241 = view_563 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_212: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_245, memory_format = torch.contiguous_format)
        var_mean_72 = torch.ops.aten.var_mean.correction(clone_212, [2], correction = 0, keepdim = True)
        getitem_198: "f32[1568, 16, 1]" = var_mean_72[0]
        getitem_199: "f32[1568, 16, 1]" = var_mean_72[1];  var_mean_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_95: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_234, getitem_191);  getitem_191 = None
        add_235: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_69: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
        mul_239: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_69);  sub_95 = rsqrt_69 = None
        mul_240: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_239, arg34_1);  mul_239 = arg34_1 = None
        add_236: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_240, arg35_1);  mul_240 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_541: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_236, [1576, 384]);  add_236 = None
        permute_254: "f32[384, 1536]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        
        # No stacktrace found for following nodes
        mm_default_75: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_541, permute_254);  view_541 = permute_254 = None
        add_tensor_75: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_75, arg37_1);  mm_default_75 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_542: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_75, [8, 197, 1536]);  add_tensor_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_241: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_542, 0.5)
        mul_242: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_542, 0.7071067811865476);  view_542 = None
        erf_25: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_242);  mul_242 = None
        add_237: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_243: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_241, add_237);  mul_241 = add_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_543: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_243, [1576, 1536]);  mul_243 = None
        permute_255: "f32[1536, 384]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        
        # No stacktrace found for following nodes
        mm_default_74: "f32[1576, 384]" = torch.ops.aten.mm.default(view_543, permute_255);  view_543 = permute_255 = None
        add_tensor_74: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_74, arg39_1);  mm_default_74 = arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_544: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_74, [8, 197, 384]);  add_tensor_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_238: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_234, view_544);  add_234 = view_544 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_59: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_238, 1, 0, 1)
        slice_61: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_238, 1, 1, 9223372036854775807);  add_238 = None
        sub_99: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_212, getitem_199);  clone_212 = getitem_199 = None
        add_246: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_72: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        mul_252: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_72);  sub_99 = rsqrt_72 = None
        mul_253: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_252, arg52_1);  mul_252 = arg52_1 = None
        add_247: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_253, arg53_1);  mul_253 = arg53_1 = None
        view_564: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_247, [8, 196, -1]);  add_247 = None
        view_565: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_564, [1568, 384]);  view_564 = None
        permute_265: "f32[384, 384]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        
        # No stacktrace found for following nodes
        mm_default_73: "f32[1568, 384]" = torch.ops.aten.mm.default(view_565, permute_265);  view_565 = permute_265 = None
        add_tensor_73: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_73, arg55_1);  mm_default_73 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_566: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_73, [8, 196, 384]);  add_tensor_73 = None
        add_248: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_61, view_566);  slice_61 = view_566 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_15: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_59, add_248], 1);  slice_59 = add_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_73 = torch.ops.aten.var_mean.correction(cat_15, [2], correction = 0, keepdim = True)
        getitem_200: "f32[8, 197, 1]" = var_mean_73[0]
        getitem_201: "f32[8, 197, 1]" = var_mean_73[1];  var_mean_73 = None
        sub_100: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_15, getitem_201);  getitem_201 = None
        add_249: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_73: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
        mul_254: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_73);  sub_100 = rsqrt_73 = None
        mul_255: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_254, arg56_1);  mul_254 = arg56_1 = None
        add_250: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_255, arg57_1);  mul_255 = arg57_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_567: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_250, [1576, 384])
        permute_266: "f32[384, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_54: "f32[1576, 768]" = torch.ops.aten.mm.default(view_567, permute_266);  view_567 = permute_266 = None
        view_568: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_54, [8, 197, 768]);  mm_54 = None
        view_569: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_568, [8, 197, 2, 6, 64]);  view_568 = None
        permute_267: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_569, [2, 0, 3, 1, 4]);  view_569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_27 = torch.ops.aten.unbind.int(permute_267);  permute_267 = None
        getitem_202: "f32[8, 6, 197, 64]" = unbind_27[0]
        getitem_203: "f32[8, 6, 197, 64]" = unbind_27[1];  unbind_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_110: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_202, [8, 6, 197, 64]);  getitem_202 = None
        clone_213: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
        view_573: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_213, [48, 197, 64]);  clone_213 = None
        permute_270: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_203, [0, 1, 3, 2]);  getitem_203 = None
        expand_111: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_270, [8, 6, 64, 197]);  permute_270 = None
        clone_214: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_574: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_214, [48, 64, 197]);  clone_214 = None
        bmm_54: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_573, view_574);  view_573 = view_574 = None
        view_575: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_54, [8, 6, 197, 197]);  bmm_54 = None
        
        # No stacktrace found for following nodes
        mul_tensor_40: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_575, 1);  view_575 = None
        amax_default_20: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_40, [-1], True)
        sub_tensor_20: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_40, amax_default_20);  mul_tensor_40 = amax_default_20 = None
        mul_tensor_41: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_20, 0.125);  sub_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_27: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_41);  mul_tensor_41 = None
        sum_28: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_27: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_112: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_27, [8, 6, 197, 197]);  div_27 = None
        view_576: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_112, [48, 197, 197]);  expand_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_570: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_250, [1576, 384]);  add_250 = None
        permute_268: "f32[384, 384]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_55: "f32[1576, 384]" = torch.ops.aten.mm.default(view_570, permute_268);  view_570 = permute_268 = None
        view_571: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_55, [8, 197, 384]);  mm_55 = None
        view_572: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_571, [8, 197, 6, -1]);  view_571 = None
        permute_269: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_113: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_269, [8, 6, 197, 64]);  permute_269 = None
        clone_215: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_577: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_215, [48, 197, 64]);  clone_215 = None
        bmm_55: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_576, view_577);  view_576 = view_577 = None
        view_578: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_55, [8, 6, 197, 64]);  bmm_55 = None
        permute_271: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
        clone_216: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_579: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_216, [8, 197, 384]);  clone_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_580: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_579, [1576, 384]);  view_579 = None
        permute_272: "f32[384, 384]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        
        # No stacktrace found for following nodes
        mm_default_72: "f32[1576, 384]" = torch.ops.aten.mm.default(view_580, permute_272);  view_580 = permute_272 = None
        add_tensor_72: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_72, arg61_1);  mm_default_72 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_581: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_72, [8, 197, 384]);  add_tensor_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_251: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_15, view_581);  cat_15 = view_581 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_74 = torch.ops.aten.var_mean.correction(add_251, [2], correction = 0, keepdim = True)
        getitem_204: "f32[8, 197, 1]" = var_mean_74[0]
        getitem_205: "f32[8, 197, 1]" = var_mean_74[1];  var_mean_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_219: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_245, memory_format = torch.contiguous_format)
        var_mean_75 = torch.ops.aten.var_mean.correction(clone_219, [2], correction = 0, keepdim = True)
        getitem_206: "f32[1568, 16, 1]" = var_mean_75[0]
        getitem_207: "f32[1568, 16, 1]" = var_mean_75[1];  var_mean_75 = None
        sub_103: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_219, getitem_207);  clone_219 = getitem_207 = None
        add_256: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_75: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        mul_262: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_75);  sub_103 = rsqrt_75 = None
        mul_263: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_262, arg68_1);  mul_262 = arg68_1 = None
        add_257: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_263, arg69_1);  mul_263 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_586: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_257, [25088, 24])
        permute_275: "f32[24, 48]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_56: "f32[25088, 48]" = torch.ops.aten.mm.default(view_586, permute_275);  view_586 = permute_275 = None
        view_587: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_56, [1568, 16, 48]);  mm_56 = None
        view_588: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_587, [1568, 16, 2, 4, 6]);  view_587 = None
        permute_276: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_588, [2, 0, 3, 1, 4]);  view_588 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_28 = torch.ops.aten.unbind.int(permute_276);  permute_276 = None
        getitem_208: "f32[1568, 4, 16, 6]" = unbind_28[0]
        getitem_209: "f32[1568, 4, 16, 6]" = unbind_28[1];  unbind_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_114: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_208, [1568, 4, 16, 6]);  getitem_208 = None
        clone_220: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_592: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_220, [6272, 16, 6]);  clone_220 = None
        permute_279: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_209, [0, 1, 3, 2]);  getitem_209 = None
        expand_115: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_279, [1568, 4, 6, 16]);  permute_279 = None
        clone_221: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_593: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_221, [6272, 6, 16]);  clone_221 = None
        bmm_56: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_592, view_593);  view_592 = view_593 = None
        view_594: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_56, [1568, 4, 16, 16]);  bmm_56 = None
        
        # No stacktrace found for following nodes
        mul_tensor_38: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_594, 1);  view_594 = None
        amax_default_19: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_38, [-1], True)
        sub_tensor_19: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_38, amax_default_19);  mul_tensor_38 = amax_default_19 = None
        mul_tensor_39: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_19, 0.408248290463863);  sub_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_28: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_39);  mul_tensor_39 = None
        sum_29: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_28: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_116: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_28, [1568, 4, 16, 16]);  div_28 = None
        view_595: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_116, [6272, 16, 16]);  expand_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_589: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_257, [25088, 24]);  add_257 = None
        permute_277: "f32[24, 24]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_57: "f32[25088, 24]" = torch.ops.aten.mm.default(view_589, permute_277);  view_589 = permute_277 = None
        view_590: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_57, [1568, 16, 24]);  mm_57 = None
        view_591: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_590, [1568, 16, 4, -1]);  view_590 = None
        permute_278: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_591, [0, 2, 1, 3]);  view_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_117: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_278, [1568, 4, 16, 6]);  permute_278 = None
        clone_222: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_596: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_222, [6272, 16, 6]);  clone_222 = None
        bmm_57: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_595, view_596);  view_595 = view_596 = None
        view_597: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_57, [1568, 4, 16, 6]);  bmm_57 = None
        permute_280: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
        clone_223: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_280, memory_format = torch.contiguous_format);  permute_280 = None
        view_598: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_223, [1568, 16, 24]);  clone_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_599: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_598, [25088, 24]);  view_598 = None
        permute_281: "f32[24, 24]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        
        # No stacktrace found for following nodes
        mm_default_71: "f32[25088, 24]" = torch.ops.aten.mm.default(view_599, permute_281);  view_599 = permute_281 = None
        add_tensor_71: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_71, arg73_1);  mm_default_71 = arg73_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_600: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_71, [1568, 16, 24]);  add_tensor_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_258: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_245, view_600);  add_245 = view_600 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_224: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_258, memory_format = torch.contiguous_format)
        var_mean_76 = torch.ops.aten.var_mean.correction(clone_224, [2], correction = 0, keepdim = True)
        getitem_210: "f32[1568, 16, 1]" = var_mean_76[0]
        getitem_211: "f32[1568, 16, 1]" = var_mean_76[1];  var_mean_76 = None
        sub_105: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_224, getitem_211);  clone_224 = getitem_211 = None
        add_259: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_76: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
        mul_265: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_76);  sub_105 = rsqrt_76 = None
        mul_266: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_265, arg74_1);  mul_265 = arg74_1 = None
        add_260: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_266, arg75_1);  mul_266 = arg75_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_601: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_260, [25088, 24]);  add_260 = None
        permute_282: "f32[24, 96]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        
        # No stacktrace found for following nodes
        mm_default_70: "f32[25088, 96]" = torch.ops.aten.mm.default(view_601, permute_282);  view_601 = permute_282 = None
        add_tensor_70: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_70, arg77_1);  mm_default_70 = arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_602: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_70, [1568, 16, 96]);  add_tensor_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_267: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_602, 0.5)
        mul_268: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_602, 0.7071067811865476);  view_602 = None
        erf_28: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_268);  mul_268 = None
        add_261: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_269: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_267, add_261);  mul_267 = add_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_603: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_269, [25088, 96]);  mul_269 = None
        permute_283: "f32[96, 24]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        
        # No stacktrace found for following nodes
        mm_default_69: "f32[25088, 24]" = torch.ops.aten.mm.default(view_603, permute_283);  view_603 = permute_283 = None
        add_tensor_69: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_69, arg79_1);  mm_default_69 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_604: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_69, [1568, 16, 24]);  add_tensor_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_262: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_258, view_604);  add_258 = view_604 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_227: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_262, memory_format = torch.contiguous_format)
        var_mean_77 = torch.ops.aten.var_mean.correction(clone_227, [2], correction = 0, keepdim = True)
        getitem_212: "f32[1568, 16, 1]" = var_mean_77[0]
        getitem_213: "f32[1568, 16, 1]" = var_mean_77[1];  var_mean_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_102: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_251, getitem_205);  getitem_205 = None
        add_252: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_74: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        mul_257: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_74);  sub_102 = rsqrt_74 = None
        mul_258: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_257, arg62_1);  mul_257 = arg62_1 = None
        add_253: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_258, arg63_1);  mul_258 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_582: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_253, [1576, 384]);  add_253 = None
        permute_273: "f32[384, 1536]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        
        # No stacktrace found for following nodes
        mm_default_68: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_582, permute_273);  view_582 = permute_273 = None
        add_tensor_68: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_68, arg65_1);  mm_default_68 = arg65_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_583: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_68, [8, 197, 1536]);  add_tensor_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_259: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_583, 0.5)
        mul_260: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_583, 0.7071067811865476);  view_583 = None
        erf_27: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_260);  mul_260 = None
        add_254: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_261: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_259, add_254);  mul_259 = add_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_584: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_261, [1576, 1536]);  mul_261 = None
        permute_274: "f32[1536, 384]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        
        # No stacktrace found for following nodes
        mm_default_67: "f32[1576, 384]" = torch.ops.aten.mm.default(view_584, permute_274);  view_584 = permute_274 = None
        add_tensor_67: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_67, arg67_1);  mm_default_67 = arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_585: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_67, [8, 197, 384]);  add_tensor_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_255: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_251, view_585);  add_251 = view_585 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_63: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_255, 1, 0, 1)
        slice_65: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_255, 1, 1, 9223372036854775807);  add_255 = None
        sub_106: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_227, getitem_213);  clone_227 = getitem_213 = None
        add_263: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_77: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        mul_270: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_77);  sub_106 = rsqrt_77 = None
        mul_271: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_270, arg80_1);  mul_270 = arg80_1 = None
        add_264: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_271, arg81_1);  mul_271 = arg81_1 = None
        view_605: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_264, [8, 196, -1]);  add_264 = None
        view_606: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_605, [1568, 384]);  view_605 = None
        permute_284: "f32[384, 384]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        
        # No stacktrace found for following nodes
        mm_default_66: "f32[1568, 384]" = torch.ops.aten.mm.default(view_606, permute_284);  view_606 = permute_284 = None
        add_tensor_66: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_66, arg83_1);  mm_default_66 = arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_607: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_66, [8, 196, 384]);  add_tensor_66 = None
        add_265: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_65, view_607);  slice_65 = view_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_16: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_63, add_265], 1);  slice_63 = add_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_78 = torch.ops.aten.var_mean.correction(cat_16, [2], correction = 0, keepdim = True)
        getitem_214: "f32[8, 197, 1]" = var_mean_78[0]
        getitem_215: "f32[8, 197, 1]" = var_mean_78[1];  var_mean_78 = None
        sub_107: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_16, getitem_215);  getitem_215 = None
        add_266: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-05);  getitem_214 = None
        rsqrt_78: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        mul_272: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_78);  sub_107 = rsqrt_78 = None
        mul_273: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_272, arg84_1);  mul_272 = arg84_1 = None
        add_267: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_273, arg85_1);  mul_273 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_608: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_267, [1576, 384])
        permute_285: "f32[384, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_58: "f32[1576, 768]" = torch.ops.aten.mm.default(view_608, permute_285);  view_608 = permute_285 = None
        view_609: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_58, [8, 197, 768]);  mm_58 = None
        view_610: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_609, [8, 197, 2, 6, 64]);  view_609 = None
        permute_286: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_610, [2, 0, 3, 1, 4]);  view_610 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_29 = torch.ops.aten.unbind.int(permute_286);  permute_286 = None
        getitem_216: "f32[8, 6, 197, 64]" = unbind_29[0]
        getitem_217: "f32[8, 6, 197, 64]" = unbind_29[1];  unbind_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_118: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_216, [8, 6, 197, 64]);  getitem_216 = None
        clone_228: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_118, memory_format = torch.contiguous_format);  expand_118 = None
        view_614: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_228, [48, 197, 64]);  clone_228 = None
        permute_289: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_217, [0, 1, 3, 2]);  getitem_217 = None
        expand_119: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_289, [8, 6, 64, 197]);  permute_289 = None
        clone_229: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
        view_615: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_229, [48, 64, 197]);  clone_229 = None
        bmm_58: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_614, view_615);  view_614 = view_615 = None
        view_616: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_58, [8, 6, 197, 197]);  bmm_58 = None
        
        # No stacktrace found for following nodes
        mul_tensor_36: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_616, 1);  view_616 = None
        amax_default_18: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_36, [-1], True)
        sub_tensor_18: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_36, amax_default_18);  mul_tensor_36 = amax_default_18 = None
        mul_tensor_37: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_18, 0.125);  sub_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_29: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_37);  mul_tensor_37 = None
        sum_30: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_29: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_120: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_29, [8, 6, 197, 197]);  div_29 = None
        view_617: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_120, [48, 197, 197]);  expand_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_611: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_267, [1576, 384]);  add_267 = None
        permute_287: "f32[384, 384]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_59: "f32[1576, 384]" = torch.ops.aten.mm.default(view_611, permute_287);  view_611 = permute_287 = None
        view_612: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_59, [8, 197, 384]);  mm_59 = None
        view_613: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_612, [8, 197, 6, -1]);  view_612 = None
        permute_288: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_121: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_288, [8, 6, 197, 64]);  permute_288 = None
        clone_230: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_618: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_230, [48, 197, 64]);  clone_230 = None
        bmm_59: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_617, view_618);  view_617 = view_618 = None
        view_619: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_59, [8, 6, 197, 64]);  bmm_59 = None
        permute_290: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
        clone_231: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
        view_620: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_231, [8, 197, 384]);  clone_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_621: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_620, [1576, 384]);  view_620 = None
        permute_291: "f32[384, 384]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        
        # No stacktrace found for following nodes
        mm_default_65: "f32[1576, 384]" = torch.ops.aten.mm.default(view_621, permute_291);  view_621 = permute_291 = None
        add_tensor_65: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_65, arg89_1);  mm_default_65 = arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_622: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_65, [8, 197, 384]);  add_tensor_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_268: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_16, view_622);  cat_16 = view_622 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_79 = torch.ops.aten.var_mean.correction(add_268, [2], correction = 0, keepdim = True)
        getitem_218: "f32[8, 197, 1]" = var_mean_79[0]
        getitem_219: "f32[8, 197, 1]" = var_mean_79[1];  var_mean_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_234: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_262, memory_format = torch.contiguous_format)
        var_mean_80 = torch.ops.aten.var_mean.correction(clone_234, [2], correction = 0, keepdim = True)
        getitem_220: "f32[1568, 16, 1]" = var_mean_80[0]
        getitem_221: "f32[1568, 16, 1]" = var_mean_80[1];  var_mean_80 = None
        sub_110: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_234, getitem_221);  clone_234 = getitem_221 = None
        add_273: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_80: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_273);  add_273 = None
        mul_280: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_80);  sub_110 = rsqrt_80 = None
        mul_281: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_280, arg96_1);  mul_280 = arg96_1 = None
        add_274: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_281, arg97_1);  mul_281 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_627: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_274, [25088, 24])
        permute_294: "f32[24, 48]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        mm_60: "f32[25088, 48]" = torch.ops.aten.mm.default(view_627, permute_294);  view_627 = permute_294 = None
        view_628: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_60, [1568, 16, 48]);  mm_60 = None
        view_629: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_628, [1568, 16, 2, 4, 6]);  view_628 = None
        permute_295: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_629, [2, 0, 3, 1, 4]);  view_629 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_30 = torch.ops.aten.unbind.int(permute_295);  permute_295 = None
        getitem_222: "f32[1568, 4, 16, 6]" = unbind_30[0]
        getitem_223: "f32[1568, 4, 16, 6]" = unbind_30[1];  unbind_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_122: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_222, [1568, 4, 16, 6]);  getitem_222 = None
        clone_235: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_633: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_235, [6272, 16, 6]);  clone_235 = None
        permute_298: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_223, [0, 1, 3, 2]);  getitem_223 = None
        expand_123: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_298, [1568, 4, 6, 16]);  permute_298 = None
        clone_236: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_634: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_236, [6272, 6, 16]);  clone_236 = None
        bmm_60: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_633, view_634);  view_633 = view_634 = None
        view_635: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_60, [1568, 4, 16, 16]);  bmm_60 = None
        
        # No stacktrace found for following nodes
        mul_tensor_34: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_635, 1);  view_635 = None
        amax_default_17: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_34, [-1], True)
        sub_tensor_17: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_34, amax_default_17);  mul_tensor_34 = amax_default_17 = None
        mul_tensor_35: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_17, 0.408248290463863);  sub_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_30: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_35);  mul_tensor_35 = None
        sum_31: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_30: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_124: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_30, [1568, 4, 16, 16]);  div_30 = None
        view_636: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_124, [6272, 16, 16]);  expand_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_630: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_274, [25088, 24]);  add_274 = None
        permute_296: "f32[24, 24]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_61: "f32[25088, 24]" = torch.ops.aten.mm.default(view_630, permute_296);  view_630 = permute_296 = None
        view_631: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_61, [1568, 16, 24]);  mm_61 = None
        view_632: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_631, [1568, 16, 4, -1]);  view_631 = None
        permute_297: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_125: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_297, [1568, 4, 16, 6]);  permute_297 = None
        clone_237: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_637: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_237, [6272, 16, 6]);  clone_237 = None
        bmm_61: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_636, view_637);  view_636 = view_637 = None
        view_638: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_61, [1568, 4, 16, 6]);  bmm_61 = None
        permute_299: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
        clone_238: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
        view_639: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_238, [1568, 16, 24]);  clone_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_640: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_639, [25088, 24]);  view_639 = None
        permute_300: "f32[24, 24]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        
        # No stacktrace found for following nodes
        mm_default_64: "f32[25088, 24]" = torch.ops.aten.mm.default(view_640, permute_300);  view_640 = permute_300 = None
        add_tensor_64: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_64, arg101_1);  mm_default_64 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_641: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_64, [1568, 16, 24]);  add_tensor_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_275: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_262, view_641);  add_262 = view_641 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_239: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_275, memory_format = torch.contiguous_format)
        var_mean_81 = torch.ops.aten.var_mean.correction(clone_239, [2], correction = 0, keepdim = True)
        getitem_224: "f32[1568, 16, 1]" = var_mean_81[0]
        getitem_225: "f32[1568, 16, 1]" = var_mean_81[1];  var_mean_81 = None
        sub_112: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_239, getitem_225);  clone_239 = getitem_225 = None
        add_276: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_81: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        mul_283: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_81);  sub_112 = rsqrt_81 = None
        mul_284: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_283, arg102_1);  mul_283 = arg102_1 = None
        add_277: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_284, arg103_1);  mul_284 = arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_642: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_277, [25088, 24]);  add_277 = None
        permute_301: "f32[24, 96]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        
        # No stacktrace found for following nodes
        mm_default_63: "f32[25088, 96]" = torch.ops.aten.mm.default(view_642, permute_301);  view_642 = permute_301 = None
        add_tensor_63: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_63, arg105_1);  mm_default_63 = arg105_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_643: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_63, [1568, 16, 96]);  add_tensor_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_285: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_643, 0.5)
        mul_286: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_643, 0.7071067811865476);  view_643 = None
        erf_30: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_286);  mul_286 = None
        add_278: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_287: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_285, add_278);  mul_285 = add_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_644: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_287, [25088, 96]);  mul_287 = None
        permute_302: "f32[96, 24]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        
        # No stacktrace found for following nodes
        mm_default_62: "f32[25088, 24]" = torch.ops.aten.mm.default(view_644, permute_302);  view_644 = permute_302 = None
        add_tensor_62: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_62, arg107_1);  mm_default_62 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_645: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_62, [1568, 16, 24]);  add_tensor_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_279: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_275, view_645);  add_275 = view_645 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_242: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_279, memory_format = torch.contiguous_format)
        var_mean_82 = torch.ops.aten.var_mean.correction(clone_242, [2], correction = 0, keepdim = True)
        getitem_226: "f32[1568, 16, 1]" = var_mean_82[0]
        getitem_227: "f32[1568, 16, 1]" = var_mean_82[1];  var_mean_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_109: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_268, getitem_219);  getitem_219 = None
        add_269: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_79: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
        mul_275: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_79);  sub_109 = rsqrt_79 = None
        mul_276: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_275, arg90_1);  mul_275 = arg90_1 = None
        add_270: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_276, arg91_1);  mul_276 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_623: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_270, [1576, 384]);  add_270 = None
        permute_292: "f32[384, 1536]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        
        # No stacktrace found for following nodes
        mm_default_61: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_623, permute_292);  view_623 = permute_292 = None
        add_tensor_61: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_61, arg93_1);  mm_default_61 = arg93_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_624: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_61, [8, 197, 1536]);  add_tensor_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_277: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_624, 0.5)
        mul_278: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_624, 0.7071067811865476);  view_624 = None
        erf_29: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_278);  mul_278 = None
        add_271: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_279: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_277, add_271);  mul_277 = add_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_625: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_279, [1576, 1536]);  mul_279 = None
        permute_293: "f32[1536, 384]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        
        # No stacktrace found for following nodes
        mm_default_60: "f32[1576, 384]" = torch.ops.aten.mm.default(view_625, permute_293);  view_625 = permute_293 = None
        add_tensor_60: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_60, arg95_1);  mm_default_60 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_626: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_60, [8, 197, 384]);  add_tensor_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_272: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_268, view_626);  add_268 = view_626 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_67: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_272, 1, 0, 1)
        slice_69: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_272, 1, 1, 9223372036854775807);  add_272 = None
        sub_113: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_242, getitem_227);  clone_242 = getitem_227 = None
        add_280: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_82: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        mul_288: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_82);  sub_113 = rsqrt_82 = None
        mul_289: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_288, arg108_1);  mul_288 = arg108_1 = None
        add_281: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_289, arg109_1);  mul_289 = arg109_1 = None
        view_646: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_281, [8, 196, -1]);  add_281 = None
        view_647: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_646, [1568, 384]);  view_646 = None
        permute_303: "f32[384, 384]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        
        # No stacktrace found for following nodes
        mm_default_59: "f32[1568, 384]" = torch.ops.aten.mm.default(view_647, permute_303);  view_647 = permute_303 = None
        add_tensor_59: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_59, arg111_1);  mm_default_59 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_648: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 196, 384]);  add_tensor_59 = None
        add_282: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_69, view_648);  slice_69 = view_648 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_17: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_67, add_282], 1);  slice_67 = add_282 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_83 = torch.ops.aten.var_mean.correction(cat_17, [2], correction = 0, keepdim = True)
        getitem_228: "f32[8, 197, 1]" = var_mean_83[0]
        getitem_229: "f32[8, 197, 1]" = var_mean_83[1];  var_mean_83 = None
        sub_114: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_17, getitem_229);  getitem_229 = None
        add_283: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-05);  getitem_228 = None
        rsqrt_83: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        mul_290: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_83);  sub_114 = rsqrt_83 = None
        mul_291: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_290, arg112_1);  mul_290 = arg112_1 = None
        add_284: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_291, arg113_1);  mul_291 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_649: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_284, [1576, 384])
        permute_304: "f32[384, 768]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_62: "f32[1576, 768]" = torch.ops.aten.mm.default(view_649, permute_304);  view_649 = permute_304 = None
        view_650: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_62, [8, 197, 768]);  mm_62 = None
        view_651: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_650, [8, 197, 2, 6, 64]);  view_650 = None
        permute_305: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_651, [2, 0, 3, 1, 4]);  view_651 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_31 = torch.ops.aten.unbind.int(permute_305);  permute_305 = None
        getitem_230: "f32[8, 6, 197, 64]" = unbind_31[0]
        getitem_231: "f32[8, 6, 197, 64]" = unbind_31[1];  unbind_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_126: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_230, [8, 6, 197, 64]);  getitem_230 = None
        clone_243: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_126, memory_format = torch.contiguous_format);  expand_126 = None
        view_655: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_243, [48, 197, 64]);  clone_243 = None
        permute_308: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_231, [0, 1, 3, 2]);  getitem_231 = None
        expand_127: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_308, [8, 6, 64, 197]);  permute_308 = None
        clone_244: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_656: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_244, [48, 64, 197]);  clone_244 = None
        bmm_62: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_655, view_656);  view_655 = view_656 = None
        view_657: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_62, [8, 6, 197, 197]);  bmm_62 = None
        
        # No stacktrace found for following nodes
        mul_tensor_32: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_657, 1);  view_657 = None
        amax_default_16: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_32, [-1], True)
        sub_tensor_16: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_32, amax_default_16);  mul_tensor_32 = amax_default_16 = None
        mul_tensor_33: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_16, 0.125);  sub_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_31: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_33);  mul_tensor_33 = None
        sum_32: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_31: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_128: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_31, [8, 6, 197, 197]);  div_31 = None
        view_658: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_128, [48, 197, 197]);  expand_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_652: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_284, [1576, 384]);  add_284 = None
        permute_306: "f32[384, 384]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_63: "f32[1576, 384]" = torch.ops.aten.mm.default(view_652, permute_306);  view_652 = permute_306 = None
        view_653: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_63, [8, 197, 384]);  mm_63 = None
        view_654: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_653, [8, 197, 6, -1]);  view_653 = None
        permute_307: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_129: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_307, [8, 6, 197, 64]);  permute_307 = None
        clone_245: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_659: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_245, [48, 197, 64]);  clone_245 = None
        bmm_63: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_658, view_659);  view_658 = view_659 = None
        view_660: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_63, [8, 6, 197, 64]);  bmm_63 = None
        permute_309: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_660, [0, 2, 1, 3]);  view_660 = None
        clone_246: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
        view_661: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_246, [8, 197, 384]);  clone_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_662: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_661, [1576, 384]);  view_661 = None
        permute_310: "f32[384, 384]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        
        # No stacktrace found for following nodes
        mm_default_58: "f32[1576, 384]" = torch.ops.aten.mm.default(view_662, permute_310);  view_662 = permute_310 = None
        add_tensor_58: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_58, arg117_1);  mm_default_58 = arg117_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_663: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 197, 384]);  add_tensor_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_285: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_17, view_663);  cat_17 = view_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_84 = torch.ops.aten.var_mean.correction(add_285, [2], correction = 0, keepdim = True)
        getitem_232: "f32[8, 197, 1]" = var_mean_84[0]
        getitem_233: "f32[8, 197, 1]" = var_mean_84[1];  var_mean_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_249: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_279, memory_format = torch.contiguous_format)
        var_mean_85 = torch.ops.aten.var_mean.correction(clone_249, [2], correction = 0, keepdim = True)
        getitem_234: "f32[1568, 16, 1]" = var_mean_85[0]
        getitem_235: "f32[1568, 16, 1]" = var_mean_85[1];  var_mean_85 = None
        sub_117: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_249, getitem_235);  clone_249 = getitem_235 = None
        add_290: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-05);  getitem_234 = None
        rsqrt_85: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        mul_298: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_85);  sub_117 = rsqrt_85 = None
        mul_299: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_298, arg124_1);  mul_298 = arg124_1 = None
        add_291: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_299, arg125_1);  mul_299 = arg125_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_668: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_291, [25088, 24])
        permute_313: "f32[24, 48]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_64: "f32[25088, 48]" = torch.ops.aten.mm.default(view_668, permute_313);  view_668 = permute_313 = None
        view_669: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_64, [1568, 16, 48]);  mm_64 = None
        view_670: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_669, [1568, 16, 2, 4, 6]);  view_669 = None
        permute_314: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_670, [2, 0, 3, 1, 4]);  view_670 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_32 = torch.ops.aten.unbind.int(permute_314);  permute_314 = None
        getitem_236: "f32[1568, 4, 16, 6]" = unbind_32[0]
        getitem_237: "f32[1568, 4, 16, 6]" = unbind_32[1];  unbind_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_130: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_236, [1568, 4, 16, 6]);  getitem_236 = None
        clone_250: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_674: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_250, [6272, 16, 6]);  clone_250 = None
        permute_317: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_237, [0, 1, 3, 2]);  getitem_237 = None
        expand_131: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_317, [1568, 4, 6, 16]);  permute_317 = None
        clone_251: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_675: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_251, [6272, 6, 16]);  clone_251 = None
        bmm_64: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_674, view_675);  view_674 = view_675 = None
        view_676: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_64, [1568, 4, 16, 16]);  bmm_64 = None
        
        # No stacktrace found for following nodes
        mul_tensor_30: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_676, 1);  view_676 = None
        amax_default_15: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_30, [-1], True)
        sub_tensor_15: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_30, amax_default_15);  mul_tensor_30 = amax_default_15 = None
        mul_tensor_31: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_15, 0.408248290463863);  sub_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_32: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_31);  mul_tensor_31 = None
        sum_33: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_32: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_132: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_32, [1568, 4, 16, 16]);  div_32 = None
        view_677: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_132, [6272, 16, 16]);  expand_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_671: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_291, [25088, 24]);  add_291 = None
        permute_315: "f32[24, 24]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_65: "f32[25088, 24]" = torch.ops.aten.mm.default(view_671, permute_315);  view_671 = permute_315 = None
        view_672: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_65, [1568, 16, 24]);  mm_65 = None
        view_673: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_672, [1568, 16, 4, -1]);  view_672 = None
        permute_316: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_133: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_316, [1568, 4, 16, 6]);  permute_316 = None
        clone_252: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_678: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_252, [6272, 16, 6]);  clone_252 = None
        bmm_65: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_677, view_678);  view_677 = view_678 = None
        view_679: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_65, [1568, 4, 16, 6]);  bmm_65 = None
        permute_318: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None
        clone_253: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
        view_680: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_253, [1568, 16, 24]);  clone_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_681: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_680, [25088, 24]);  view_680 = None
        permute_319: "f32[24, 24]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        
        # No stacktrace found for following nodes
        mm_default_57: "f32[25088, 24]" = torch.ops.aten.mm.default(view_681, permute_319);  view_681 = permute_319 = None
        add_tensor_57: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_57, arg129_1);  mm_default_57 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_682: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_57, [1568, 16, 24]);  add_tensor_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_292: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_279, view_682);  add_279 = view_682 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_254: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_292, memory_format = torch.contiguous_format)
        var_mean_86 = torch.ops.aten.var_mean.correction(clone_254, [2], correction = 0, keepdim = True)
        getitem_238: "f32[1568, 16, 1]" = var_mean_86[0]
        getitem_239: "f32[1568, 16, 1]" = var_mean_86[1];  var_mean_86 = None
        sub_119: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_254, getitem_239);  clone_254 = getitem_239 = None
        add_293: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_86: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        mul_301: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_86);  sub_119 = rsqrt_86 = None
        mul_302: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_301, arg130_1);  mul_301 = arg130_1 = None
        add_294: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_302, arg131_1);  mul_302 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_683: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_294, [25088, 24]);  add_294 = None
        permute_320: "f32[24, 96]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        
        # No stacktrace found for following nodes
        mm_default_56: "f32[25088, 96]" = torch.ops.aten.mm.default(view_683, permute_320);  view_683 = permute_320 = None
        add_tensor_56: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_56, arg133_1);  mm_default_56 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_684: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_56, [1568, 16, 96]);  add_tensor_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_303: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_684, 0.5)
        mul_304: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_684, 0.7071067811865476);  view_684 = None
        erf_32: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_304);  mul_304 = None
        add_295: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_305: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_303, add_295);  mul_303 = add_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_685: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_305, [25088, 96]);  mul_305 = None
        permute_321: "f32[96, 24]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        
        # No stacktrace found for following nodes
        mm_default_55: "f32[25088, 24]" = torch.ops.aten.mm.default(view_685, permute_321);  view_685 = permute_321 = None
        add_tensor_55: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_55, arg135_1);  mm_default_55 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_686: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_55, [1568, 16, 24]);  add_tensor_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_296: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_292, view_686);  add_292 = view_686 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_257: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_296, memory_format = torch.contiguous_format)
        var_mean_87 = torch.ops.aten.var_mean.correction(clone_257, [2], correction = 0, keepdim = True)
        getitem_240: "f32[1568, 16, 1]" = var_mean_87[0]
        getitem_241: "f32[1568, 16, 1]" = var_mean_87[1];  var_mean_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_116: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_285, getitem_233);  getitem_233 = None
        add_286: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-05);  getitem_232 = None
        rsqrt_84: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        mul_293: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_84);  sub_116 = rsqrt_84 = None
        mul_294: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_293, arg118_1);  mul_293 = arg118_1 = None
        add_287: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_294, arg119_1);  mul_294 = arg119_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_664: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_287, [1576, 384]);  add_287 = None
        permute_311: "f32[384, 1536]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        
        # No stacktrace found for following nodes
        mm_default_54: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_664, permute_311);  view_664 = permute_311 = None
        add_tensor_54: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_54, arg121_1);  mm_default_54 = arg121_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_665: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 197, 1536]);  add_tensor_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_295: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_665, 0.5)
        mul_296: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_665, 0.7071067811865476);  view_665 = None
        erf_31: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_296);  mul_296 = None
        add_288: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_297: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_295, add_288);  mul_295 = add_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_666: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_297, [1576, 1536]);  mul_297 = None
        permute_312: "f32[1536, 384]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        
        # No stacktrace found for following nodes
        mm_default_53: "f32[1576, 384]" = torch.ops.aten.mm.default(view_666, permute_312);  view_666 = permute_312 = None
        add_tensor_53: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_53, arg123_1);  mm_default_53 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_667: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 197, 384]);  add_tensor_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_289: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_285, view_667);  add_285 = view_667 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_71: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_289, 1, 0, 1)
        slice_73: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_289, 1, 1, 9223372036854775807);  add_289 = None
        sub_120: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_257, getitem_241);  clone_257 = getitem_241 = None
        add_297: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_87: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_297);  add_297 = None
        mul_306: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_87);  sub_120 = rsqrt_87 = None
        mul_307: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_306, arg136_1);  mul_306 = arg136_1 = None
        add_298: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_307, arg137_1);  mul_307 = arg137_1 = None
        view_687: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_298, [8, 196, -1]);  add_298 = None
        view_688: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_687, [1568, 384]);  view_687 = None
        permute_322: "f32[384, 384]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        
        # No stacktrace found for following nodes
        mm_default_52: "f32[1568, 384]" = torch.ops.aten.mm.default(view_688, permute_322);  view_688 = permute_322 = None
        add_tensor_52: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_52, arg139_1);  mm_default_52 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_689: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 196, 384]);  add_tensor_52 = None
        add_299: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_73, view_689);  slice_73 = view_689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_18: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_71, add_299], 1);  slice_71 = add_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_88 = torch.ops.aten.var_mean.correction(cat_18, [2], correction = 0, keepdim = True)
        getitem_242: "f32[8, 197, 1]" = var_mean_88[0]
        getitem_243: "f32[8, 197, 1]" = var_mean_88[1];  var_mean_88 = None
        sub_121: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_18, getitem_243);  getitem_243 = None
        add_300: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-05);  getitem_242 = None
        rsqrt_88: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
        mul_308: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_88);  sub_121 = rsqrt_88 = None
        mul_309: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_308, arg140_1);  mul_308 = arg140_1 = None
        add_301: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_309, arg141_1);  mul_309 = arg141_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_690: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_301, [1576, 384])
        permute_323: "f32[384, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_66: "f32[1576, 768]" = torch.ops.aten.mm.default(view_690, permute_323);  view_690 = permute_323 = None
        view_691: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_66, [8, 197, 768]);  mm_66 = None
        view_692: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_691, [8, 197, 2, 6, 64]);  view_691 = None
        permute_324: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_692, [2, 0, 3, 1, 4]);  view_692 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_33 = torch.ops.aten.unbind.int(permute_324);  permute_324 = None
        getitem_244: "f32[8, 6, 197, 64]" = unbind_33[0]
        getitem_245: "f32[8, 6, 197, 64]" = unbind_33[1];  unbind_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_134: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_244, [8, 6, 197, 64]);  getitem_244 = None
        clone_258: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
        view_696: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_258, [48, 197, 64]);  clone_258 = None
        permute_327: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_245, [0, 1, 3, 2]);  getitem_245 = None
        expand_135: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_327, [8, 6, 64, 197]);  permute_327 = None
        clone_259: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
        view_697: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_259, [48, 64, 197]);  clone_259 = None
        bmm_66: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_696, view_697);  view_696 = view_697 = None
        view_698: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_66, [8, 6, 197, 197]);  bmm_66 = None
        
        # No stacktrace found for following nodes
        mul_tensor_28: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_698, 1);  view_698 = None
        amax_default_14: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_28, [-1], True)
        sub_tensor_14: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_28, amax_default_14);  mul_tensor_28 = amax_default_14 = None
        mul_tensor_29: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_14, 0.125);  sub_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_33: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_29);  mul_tensor_29 = None
        sum_34: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_33: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_136: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_33, [8, 6, 197, 197]);  div_33 = None
        view_699: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_136, [48, 197, 197]);  expand_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_693: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_301, [1576, 384]);  add_301 = None
        permute_325: "f32[384, 384]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_67: "f32[1576, 384]" = torch.ops.aten.mm.default(view_693, permute_325);  view_693 = permute_325 = None
        view_694: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_67, [8, 197, 384]);  mm_67 = None
        view_695: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_694, [8, 197, 6, -1]);  view_694 = None
        permute_326: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_137: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_326, [8, 6, 197, 64]);  permute_326 = None
        clone_260: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_700: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_260, [48, 197, 64]);  clone_260 = None
        bmm_67: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_699, view_700);  view_699 = view_700 = None
        view_701: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_67, [8, 6, 197, 64]);  bmm_67 = None
        permute_328: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_701, [0, 2, 1, 3]);  view_701 = None
        clone_261: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        view_702: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_261, [8, 197, 384]);  clone_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_703: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_702, [1576, 384]);  view_702 = None
        permute_329: "f32[384, 384]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        
        # No stacktrace found for following nodes
        mm_default_51: "f32[1576, 384]" = torch.ops.aten.mm.default(view_703, permute_329);  view_703 = permute_329 = None
        add_tensor_51: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_51, arg145_1);  mm_default_51 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_704: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 197, 384]);  add_tensor_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_302: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_18, view_704);  cat_18 = view_704 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_89 = torch.ops.aten.var_mean.correction(add_302, [2], correction = 0, keepdim = True)
        getitem_246: "f32[8, 197, 1]" = var_mean_89[0]
        getitem_247: "f32[8, 197, 1]" = var_mean_89[1];  var_mean_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_264: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_296, memory_format = torch.contiguous_format)
        var_mean_90 = torch.ops.aten.var_mean.correction(clone_264, [2], correction = 0, keepdim = True)
        getitem_248: "f32[1568, 16, 1]" = var_mean_90[0]
        getitem_249: "f32[1568, 16, 1]" = var_mean_90[1];  var_mean_90 = None
        sub_124: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_264, getitem_249);  clone_264 = getitem_249 = None
        add_307: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-05);  getitem_248 = None
        rsqrt_90: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
        mul_316: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_90);  sub_124 = rsqrt_90 = None
        mul_317: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_316, arg152_1);  mul_316 = arg152_1 = None
        add_308: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_317, arg153_1);  mul_317 = arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_709: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_308, [25088, 24])
        permute_332: "f32[24, 48]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_68: "f32[25088, 48]" = torch.ops.aten.mm.default(view_709, permute_332);  view_709 = permute_332 = None
        view_710: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_68, [1568, 16, 48]);  mm_68 = None
        view_711: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_710, [1568, 16, 2, 4, 6]);  view_710 = None
        permute_333: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_711, [2, 0, 3, 1, 4]);  view_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_34 = torch.ops.aten.unbind.int(permute_333);  permute_333 = None
        getitem_250: "f32[1568, 4, 16, 6]" = unbind_34[0]
        getitem_251: "f32[1568, 4, 16, 6]" = unbind_34[1];  unbind_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_138: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_250, [1568, 4, 16, 6]);  getitem_250 = None
        clone_265: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_715: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_265, [6272, 16, 6]);  clone_265 = None
        permute_336: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_251, [0, 1, 3, 2]);  getitem_251 = None
        expand_139: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_336, [1568, 4, 6, 16]);  permute_336 = None
        clone_266: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_716: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_266, [6272, 6, 16]);  clone_266 = None
        bmm_68: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_715, view_716);  view_715 = view_716 = None
        view_717: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_68, [1568, 4, 16, 16]);  bmm_68 = None
        
        # No stacktrace found for following nodes
        mul_tensor_26: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_717, 1);  view_717 = None
        amax_default_13: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_26, [-1], True)
        sub_tensor_13: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_26, amax_default_13);  mul_tensor_26 = amax_default_13 = None
        mul_tensor_27: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_13, 0.408248290463863);  sub_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_34: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_27);  mul_tensor_27 = None
        sum_35: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_34: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_140: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_34, [1568, 4, 16, 16]);  div_34 = None
        view_718: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_140, [6272, 16, 16]);  expand_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_712: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_308, [25088, 24]);  add_308 = None
        permute_334: "f32[24, 24]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_69: "f32[25088, 24]" = torch.ops.aten.mm.default(view_712, permute_334);  view_712 = permute_334 = None
        view_713: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_69, [1568, 16, 24]);  mm_69 = None
        view_714: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_713, [1568, 16, 4, -1]);  view_713 = None
        permute_335: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_714, [0, 2, 1, 3]);  view_714 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_141: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_335, [1568, 4, 16, 6]);  permute_335 = None
        clone_267: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
        view_719: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_267, [6272, 16, 6]);  clone_267 = None
        bmm_69: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_718, view_719);  view_718 = view_719 = None
        view_720: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_69, [1568, 4, 16, 6]);  bmm_69 = None
        permute_337: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
        clone_268: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_721: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_268, [1568, 16, 24]);  clone_268 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_722: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_721, [25088, 24]);  view_721 = None
        permute_338: "f32[24, 24]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        
        # No stacktrace found for following nodes
        mm_default_50: "f32[25088, 24]" = torch.ops.aten.mm.default(view_722, permute_338);  view_722 = permute_338 = None
        add_tensor_50: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_50, arg157_1);  mm_default_50 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_723: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_50, [1568, 16, 24]);  add_tensor_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_309: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_296, view_723);  add_296 = view_723 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_269: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format)
        var_mean_91 = torch.ops.aten.var_mean.correction(clone_269, [2], correction = 0, keepdim = True)
        getitem_252: "f32[1568, 16, 1]" = var_mean_91[0]
        getitem_253: "f32[1568, 16, 1]" = var_mean_91[1];  var_mean_91 = None
        sub_126: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_269, getitem_253);  clone_269 = getitem_253 = None
        add_310: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_91: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
        mul_319: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_91);  sub_126 = rsqrt_91 = None
        mul_320: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_319, arg158_1);  mul_319 = arg158_1 = None
        add_311: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_320, arg159_1);  mul_320 = arg159_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_724: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_311, [25088, 24]);  add_311 = None
        permute_339: "f32[24, 96]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        
        # No stacktrace found for following nodes
        mm_default_49: "f32[25088, 96]" = torch.ops.aten.mm.default(view_724, permute_339);  view_724 = permute_339 = None
        add_tensor_49: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_49, arg161_1);  mm_default_49 = arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_725: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_49, [1568, 16, 96]);  add_tensor_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_321: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_725, 0.5)
        mul_322: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_725, 0.7071067811865476);  view_725 = None
        erf_34: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_322);  mul_322 = None
        add_312: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_323: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_321, add_312);  mul_321 = add_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_726: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_323, [25088, 96]);  mul_323 = None
        permute_340: "f32[96, 24]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        
        # No stacktrace found for following nodes
        mm_default_48: "f32[25088, 24]" = torch.ops.aten.mm.default(view_726, permute_340);  view_726 = permute_340 = None
        add_tensor_48: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_48, arg163_1);  mm_default_48 = arg163_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_727: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_48, [1568, 16, 24]);  add_tensor_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_313: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_309, view_727);  add_309 = view_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_272: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_313, memory_format = torch.contiguous_format)
        var_mean_92 = torch.ops.aten.var_mean.correction(clone_272, [2], correction = 0, keepdim = True)
        getitem_254: "f32[1568, 16, 1]" = var_mean_92[0]
        getitem_255: "f32[1568, 16, 1]" = var_mean_92[1];  var_mean_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_123: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_302, getitem_247);  getitem_247 = None
        add_303: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_89: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        mul_311: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_89);  sub_123 = rsqrt_89 = None
        mul_312: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_311, arg146_1);  mul_311 = arg146_1 = None
        add_304: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_312, arg147_1);  mul_312 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_705: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_304, [1576, 384]);  add_304 = None
        permute_330: "f32[384, 1536]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        
        # No stacktrace found for following nodes
        mm_default_47: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_705, permute_330);  view_705 = permute_330 = None
        add_tensor_47: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_47, arg149_1);  mm_default_47 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_706: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 197, 1536]);  add_tensor_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_313: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_706, 0.5)
        mul_314: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_706, 0.7071067811865476);  view_706 = None
        erf_33: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_314);  mul_314 = None
        add_305: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_315: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_313, add_305);  mul_313 = add_305 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_707: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_315, [1576, 1536]);  mul_315 = None
        permute_331: "f32[1536, 384]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        
        # No stacktrace found for following nodes
        mm_default_46: "f32[1576, 384]" = torch.ops.aten.mm.default(view_707, permute_331);  view_707 = permute_331 = None
        add_tensor_46: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_46, arg151_1);  mm_default_46 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_708: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 197, 384]);  add_tensor_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_306: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_302, view_708);  add_302 = view_708 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_75: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_306, 1, 0, 1)
        slice_77: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_306, 1, 1, 9223372036854775807);  add_306 = None
        sub_127: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_272, getitem_255);  clone_272 = getitem_255 = None
        add_314: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-05);  getitem_254 = None
        rsqrt_92: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
        mul_324: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_92);  sub_127 = rsqrt_92 = None
        mul_325: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_324, arg164_1);  mul_324 = arg164_1 = None
        add_315: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_325, arg165_1);  mul_325 = arg165_1 = None
        view_728: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_315, [8, 196, -1]);  add_315 = None
        view_729: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_728, [1568, 384]);  view_728 = None
        permute_341: "f32[384, 384]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        
        # No stacktrace found for following nodes
        mm_default_45: "f32[1568, 384]" = torch.ops.aten.mm.default(view_729, permute_341);  view_729 = permute_341 = None
        add_tensor_45: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_45, arg167_1);  mm_default_45 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_730: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 196, 384]);  add_tensor_45 = None
        add_316: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_77, view_730);  slice_77 = view_730 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_19: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_75, add_316], 1);  slice_75 = add_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_93 = torch.ops.aten.var_mean.correction(cat_19, [2], correction = 0, keepdim = True)
        getitem_256: "f32[8, 197, 1]" = var_mean_93[0]
        getitem_257: "f32[8, 197, 1]" = var_mean_93[1];  var_mean_93 = None
        sub_128: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_19, getitem_257);  getitem_257 = None
        add_317: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-05);  getitem_256 = None
        rsqrt_93: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
        mul_326: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_93);  sub_128 = rsqrt_93 = None
        mul_327: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_326, arg168_1);  mul_326 = arg168_1 = None
        add_318: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_327, arg169_1);  mul_327 = arg169_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_731: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_318, [1576, 384])
        permute_342: "f32[384, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        mm_70: "f32[1576, 768]" = torch.ops.aten.mm.default(view_731, permute_342);  view_731 = permute_342 = None
        view_732: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_70, [8, 197, 768]);  mm_70 = None
        view_733: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_732, [8, 197, 2, 6, 64]);  view_732 = None
        permute_343: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_733, [2, 0, 3, 1, 4]);  view_733 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_35 = torch.ops.aten.unbind.int(permute_343);  permute_343 = None
        getitem_258: "f32[8, 6, 197, 64]" = unbind_35[0]
        getitem_259: "f32[8, 6, 197, 64]" = unbind_35[1];  unbind_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_142: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_258, [8, 6, 197, 64]);  getitem_258 = None
        clone_273: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_142, memory_format = torch.contiguous_format);  expand_142 = None
        view_737: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_273, [48, 197, 64]);  clone_273 = None
        permute_346: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_259, [0, 1, 3, 2]);  getitem_259 = None
        expand_143: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_346, [8, 6, 64, 197]);  permute_346 = None
        clone_274: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
        view_738: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_274, [48, 64, 197]);  clone_274 = None
        bmm_70: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_737, view_738);  view_737 = view_738 = None
        view_739: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_70, [8, 6, 197, 197]);  bmm_70 = None
        
        # No stacktrace found for following nodes
        mul_tensor_24: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_739, 1);  view_739 = None
        amax_default_12: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_24, [-1], True)
        sub_tensor_12: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_24, amax_default_12);  mul_tensor_24 = amax_default_12 = None
        mul_tensor_25: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_12, 0.125);  sub_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_35: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_25);  mul_tensor_25 = None
        sum_36: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_35: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_144: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_35, [8, 6, 197, 197]);  div_35 = None
        view_740: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_144, [48, 197, 197]);  expand_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_734: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_318, [1576, 384]);  add_318 = None
        permute_344: "f32[384, 384]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_71: "f32[1576, 384]" = torch.ops.aten.mm.default(view_734, permute_344);  view_734 = permute_344 = None
        view_735: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_71, [8, 197, 384]);  mm_71 = None
        view_736: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_735, [8, 197, 6, -1]);  view_735 = None
        permute_345: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_736, [0, 2, 1, 3]);  view_736 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_145: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_345, [8, 6, 197, 64]);  permute_345 = None
        clone_275: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_741: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_275, [48, 197, 64]);  clone_275 = None
        bmm_71: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_740, view_741);  view_740 = view_741 = None
        view_742: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_71, [8, 6, 197, 64]);  bmm_71 = None
        permute_347: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
        clone_276: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        view_743: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_276, [8, 197, 384]);  clone_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_744: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_743, [1576, 384]);  view_743 = None
        permute_348: "f32[384, 384]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        
        # No stacktrace found for following nodes
        mm_default_44: "f32[1576, 384]" = torch.ops.aten.mm.default(view_744, permute_348);  view_744 = permute_348 = None
        add_tensor_44: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_44, arg173_1);  mm_default_44 = arg173_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_745: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 197, 384]);  add_tensor_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_319: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_19, view_745);  cat_19 = view_745 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_94 = torch.ops.aten.var_mean.correction(add_319, [2], correction = 0, keepdim = True)
        getitem_260: "f32[8, 197, 1]" = var_mean_94[0]
        getitem_261: "f32[8, 197, 1]" = var_mean_94[1];  var_mean_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_279: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_313, memory_format = torch.contiguous_format)
        var_mean_95 = torch.ops.aten.var_mean.correction(clone_279, [2], correction = 0, keepdim = True)
        getitem_262: "f32[1568, 16, 1]" = var_mean_95[0]
        getitem_263: "f32[1568, 16, 1]" = var_mean_95[1];  var_mean_95 = None
        sub_131: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_279, getitem_263);  clone_279 = getitem_263 = None
        add_324: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_262, 1e-05);  getitem_262 = None
        rsqrt_95: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        mul_334: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_95);  sub_131 = rsqrt_95 = None
        mul_335: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_334, arg180_1);  mul_334 = arg180_1 = None
        add_325: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_335, arg181_1);  mul_335 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_750: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_325, [25088, 24])
        permute_351: "f32[24, 48]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_72: "f32[25088, 48]" = torch.ops.aten.mm.default(view_750, permute_351);  view_750 = permute_351 = None
        view_751: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_72, [1568, 16, 48]);  mm_72 = None
        view_752: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_751, [1568, 16, 2, 4, 6]);  view_751 = None
        permute_352: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_752, [2, 0, 3, 1, 4]);  view_752 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_36 = torch.ops.aten.unbind.int(permute_352);  permute_352 = None
        getitem_264: "f32[1568, 4, 16, 6]" = unbind_36[0]
        getitem_265: "f32[1568, 4, 16, 6]" = unbind_36[1];  unbind_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_146: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_264, [1568, 4, 16, 6]);  getitem_264 = None
        clone_280: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_756: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_280, [6272, 16, 6]);  clone_280 = None
        permute_355: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_265, [0, 1, 3, 2]);  getitem_265 = None
        expand_147: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_355, [1568, 4, 6, 16]);  permute_355 = None
        clone_281: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_757: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_281, [6272, 6, 16]);  clone_281 = None
        bmm_72: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_756, view_757);  view_756 = view_757 = None
        view_758: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_72, [1568, 4, 16, 16]);  bmm_72 = None
        
        # No stacktrace found for following nodes
        mul_tensor_22: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_758, 1);  view_758 = None
        amax_default_11: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_22, [-1], True)
        sub_tensor_11: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_22, amax_default_11);  mul_tensor_22 = amax_default_11 = None
        mul_tensor_23: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_11, 0.408248290463863);  sub_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_36: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_23);  mul_tensor_23 = None
        sum_37: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_36: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_148: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_36, [1568, 4, 16, 16]);  div_36 = None
        view_759: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_148, [6272, 16, 16]);  expand_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_753: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_325, [25088, 24]);  add_325 = None
        permute_353: "f32[24, 24]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_73: "f32[25088, 24]" = torch.ops.aten.mm.default(view_753, permute_353);  view_753 = permute_353 = None
        view_754: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_73, [1568, 16, 24]);  mm_73 = None
        view_755: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_754, [1568, 16, 4, -1]);  view_754 = None
        permute_354: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_755, [0, 2, 1, 3]);  view_755 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_149: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_354, [1568, 4, 16, 6]);  permute_354 = None
        clone_282: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_760: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_282, [6272, 16, 6]);  clone_282 = None
        bmm_73: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_759, view_760);  view_759 = view_760 = None
        view_761: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_73, [1568, 4, 16, 6]);  bmm_73 = None
        permute_356: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_761, [0, 2, 1, 3]);  view_761 = None
        clone_283: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
        view_762: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_283, [1568, 16, 24]);  clone_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_763: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_762, [25088, 24]);  view_762 = None
        permute_357: "f32[24, 24]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        
        # No stacktrace found for following nodes
        mm_default_43: "f32[25088, 24]" = torch.ops.aten.mm.default(view_763, permute_357);  view_763 = permute_357 = None
        add_tensor_43: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_43, arg185_1);  mm_default_43 = arg185_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_764: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_43, [1568, 16, 24]);  add_tensor_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_326: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_313, view_764);  add_313 = view_764 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_284: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_326, memory_format = torch.contiguous_format)
        var_mean_96 = torch.ops.aten.var_mean.correction(clone_284, [2], correction = 0, keepdim = True)
        getitem_266: "f32[1568, 16, 1]" = var_mean_96[0]
        getitem_267: "f32[1568, 16, 1]" = var_mean_96[1];  var_mean_96 = None
        sub_133: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_284, getitem_267);  clone_284 = getitem_267 = None
        add_327: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-05);  getitem_266 = None
        rsqrt_96: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
        mul_337: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_96);  sub_133 = rsqrt_96 = None
        mul_338: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_337, arg186_1);  mul_337 = arg186_1 = None
        add_328: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_338, arg187_1);  mul_338 = arg187_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_765: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_328, [25088, 24]);  add_328 = None
        permute_358: "f32[24, 96]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        
        # No stacktrace found for following nodes
        mm_default_42: "f32[25088, 96]" = torch.ops.aten.mm.default(view_765, permute_358);  view_765 = permute_358 = None
        add_tensor_42: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_42, arg189_1);  mm_default_42 = arg189_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_766: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_42, [1568, 16, 96]);  add_tensor_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_339: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_766, 0.5)
        mul_340: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_766, 0.7071067811865476);  view_766 = None
        erf_36: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_340);  mul_340 = None
        add_329: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_341: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_339, add_329);  mul_339 = add_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_767: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_341, [25088, 96]);  mul_341 = None
        permute_359: "f32[96, 24]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        
        # No stacktrace found for following nodes
        mm_default_41: "f32[25088, 24]" = torch.ops.aten.mm.default(view_767, permute_359);  view_767 = permute_359 = None
        add_tensor_41: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_41, arg191_1);  mm_default_41 = arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_768: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_41, [1568, 16, 24]);  add_tensor_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_330: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_326, view_768);  add_326 = view_768 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_287: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
        var_mean_97 = torch.ops.aten.var_mean.correction(clone_287, [2], correction = 0, keepdim = True)
        getitem_268: "f32[1568, 16, 1]" = var_mean_97[0]
        getitem_269: "f32[1568, 16, 1]" = var_mean_97[1];  var_mean_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_130: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_319, getitem_261);  getitem_261 = None
        add_320: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-05);  getitem_260 = None
        rsqrt_94: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
        mul_329: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_94);  sub_130 = rsqrt_94 = None
        mul_330: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_329, arg174_1);  mul_329 = arg174_1 = None
        add_321: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_330, arg175_1);  mul_330 = arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_746: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_321, [1576, 384]);  add_321 = None
        permute_349: "f32[384, 1536]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        
        # No stacktrace found for following nodes
        mm_default_40: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_746, permute_349);  view_746 = permute_349 = None
        add_tensor_40: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_40, arg177_1);  mm_default_40 = arg177_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_747: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 197, 1536]);  add_tensor_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_331: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_747, 0.5)
        mul_332: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_747, 0.7071067811865476);  view_747 = None
        erf_35: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_332);  mul_332 = None
        add_322: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_333: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_331, add_322);  mul_331 = add_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_748: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_333, [1576, 1536]);  mul_333 = None
        permute_350: "f32[1536, 384]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        
        # No stacktrace found for following nodes
        mm_default_39: "f32[1576, 384]" = torch.ops.aten.mm.default(view_748, permute_350);  view_748 = permute_350 = None
        add_tensor_39: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_39, arg179_1);  mm_default_39 = arg179_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_749: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 197, 384]);  add_tensor_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_323: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_319, view_749);  add_319 = view_749 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_79: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_323, 1, 0, 1)
        slice_81: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_323, 1, 1, 9223372036854775807);  add_323 = None
        sub_134: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_287, getitem_269);  clone_287 = getitem_269 = None
        add_331: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
        rsqrt_97: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        mul_342: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_97);  sub_134 = rsqrt_97 = None
        mul_343: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_342, arg192_1);  mul_342 = arg192_1 = None
        add_332: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_343, arg193_1);  mul_343 = arg193_1 = None
        view_769: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_332, [8, 196, -1]);  add_332 = None
        view_770: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_769, [1568, 384]);  view_769 = None
        permute_360: "f32[384, 384]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        
        # No stacktrace found for following nodes
        mm_default_38: "f32[1568, 384]" = torch.ops.aten.mm.default(view_770, permute_360);  view_770 = permute_360 = None
        add_tensor_38: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_38, arg195_1);  mm_default_38 = arg195_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_771: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 196, 384]);  add_tensor_38 = None
        add_333: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_81, view_771);  slice_81 = view_771 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_20: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_79, add_333], 1);  slice_79 = add_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_98 = torch.ops.aten.var_mean.correction(cat_20, [2], correction = 0, keepdim = True)
        getitem_270: "f32[8, 197, 1]" = var_mean_98[0]
        getitem_271: "f32[8, 197, 1]" = var_mean_98[1];  var_mean_98 = None
        sub_135: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_20, getitem_271);  getitem_271 = None
        add_334: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-05);  getitem_270 = None
        rsqrt_98: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        mul_344: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_98);  sub_135 = rsqrt_98 = None
        mul_345: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_344, arg196_1);  mul_344 = arg196_1 = None
        add_335: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_345, arg197_1);  mul_345 = arg197_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_772: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_335, [1576, 384])
        permute_361: "f32[384, 768]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        mm_74: "f32[1576, 768]" = torch.ops.aten.mm.default(view_772, permute_361);  view_772 = permute_361 = None
        view_773: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_74, [8, 197, 768]);  mm_74 = None
        view_774: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_773, [8, 197, 2, 6, 64]);  view_773 = None
        permute_362: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_774, [2, 0, 3, 1, 4]);  view_774 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_37 = torch.ops.aten.unbind.int(permute_362);  permute_362 = None
        getitem_272: "f32[8, 6, 197, 64]" = unbind_37[0]
        getitem_273: "f32[8, 6, 197, 64]" = unbind_37[1];  unbind_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_150: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_272, [8, 6, 197, 64]);  getitem_272 = None
        clone_288: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_150, memory_format = torch.contiguous_format);  expand_150 = None
        view_778: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_288, [48, 197, 64]);  clone_288 = None
        permute_365: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_273, [0, 1, 3, 2]);  getitem_273 = None
        expand_151: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_365, [8, 6, 64, 197]);  permute_365 = None
        clone_289: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_779: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_289, [48, 64, 197]);  clone_289 = None
        bmm_74: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_778, view_779);  view_778 = view_779 = None
        view_780: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_74, [8, 6, 197, 197]);  bmm_74 = None
        
        # No stacktrace found for following nodes
        mul_tensor_20: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_780, 1);  view_780 = None
        amax_default_10: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_20, [-1], True)
        sub_tensor_10: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_20, amax_default_10);  mul_tensor_20 = amax_default_10 = None
        mul_tensor_21: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.125);  sub_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_37: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_21);  mul_tensor_21 = None
        sum_38: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_37: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_37, sum_38);  exp_37 = sum_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_152: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_37, [8, 6, 197, 197]);  div_37 = None
        view_781: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_152, [48, 197, 197]);  expand_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_775: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_335, [1576, 384]);  add_335 = None
        permute_363: "f32[384, 384]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        mm_75: "f32[1576, 384]" = torch.ops.aten.mm.default(view_775, permute_363);  view_775 = permute_363 = None
        view_776: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_75, [8, 197, 384]);  mm_75 = None
        view_777: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_776, [8, 197, 6, -1]);  view_776 = None
        permute_364: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_777, [0, 2, 1, 3]);  view_777 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_153: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_364, [8, 6, 197, 64]);  permute_364 = None
        clone_290: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_782: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_290, [48, 197, 64]);  clone_290 = None
        bmm_75: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_781, view_782);  view_781 = view_782 = None
        view_783: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_75, [8, 6, 197, 64]);  bmm_75 = None
        permute_366: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_783, [0, 2, 1, 3]);  view_783 = None
        clone_291: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
        view_784: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_291, [8, 197, 384]);  clone_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_785: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_784, [1576, 384]);  view_784 = None
        permute_367: "f32[384, 384]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[1576, 384]" = torch.ops.aten.mm.default(view_785, permute_367);  view_785 = permute_367 = None
        add_tensor_37: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_37, arg201_1);  mm_default_37 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_786: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 197, 384]);  add_tensor_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_336: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_20, view_786);  cat_20 = view_786 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_99 = torch.ops.aten.var_mean.correction(add_336, [2], correction = 0, keepdim = True)
        getitem_274: "f32[8, 197, 1]" = var_mean_99[0]
        getitem_275: "f32[8, 197, 1]" = var_mean_99[1];  var_mean_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_294: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
        var_mean_100 = torch.ops.aten.var_mean.correction(clone_294, [2], correction = 0, keepdim = True)
        getitem_276: "f32[1568, 16, 1]" = var_mean_100[0]
        getitem_277: "f32[1568, 16, 1]" = var_mean_100[1];  var_mean_100 = None
        sub_138: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_294, getitem_277);  clone_294 = getitem_277 = None
        add_341: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_276, 1e-05);  getitem_276 = None
        rsqrt_100: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        mul_352: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_100);  sub_138 = rsqrt_100 = None
        mul_353: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_352, arg208_1);  mul_352 = arg208_1 = None
        add_342: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_353, arg209_1);  mul_353 = arg209_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_791: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_342, [25088, 24])
        permute_370: "f32[24, 48]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        mm_76: "f32[25088, 48]" = torch.ops.aten.mm.default(view_791, permute_370);  view_791 = permute_370 = None
        view_792: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_76, [1568, 16, 48]);  mm_76 = None
        view_793: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_792, [1568, 16, 2, 4, 6]);  view_792 = None
        permute_371: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_793, [2, 0, 3, 1, 4]);  view_793 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_38 = torch.ops.aten.unbind.int(permute_371);  permute_371 = None
        getitem_278: "f32[1568, 4, 16, 6]" = unbind_38[0]
        getitem_279: "f32[1568, 4, 16, 6]" = unbind_38[1];  unbind_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_154: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_278, [1568, 4, 16, 6]);  getitem_278 = None
        clone_295: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_797: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_295, [6272, 16, 6]);  clone_295 = None
        permute_374: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_279, [0, 1, 3, 2]);  getitem_279 = None
        expand_155: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_374, [1568, 4, 6, 16]);  permute_374 = None
        clone_296: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_798: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_296, [6272, 6, 16]);  clone_296 = None
        bmm_76: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_797, view_798);  view_797 = view_798 = None
        view_799: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_76, [1568, 4, 16, 16]);  bmm_76 = None
        
        # No stacktrace found for following nodes
        mul_tensor_18: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_799, 1);  view_799 = None
        amax_default_9: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_18, [-1], True)
        sub_tensor_9: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_18, amax_default_9);  mul_tensor_18 = amax_default_9 = None
        mul_tensor_19: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_9, 0.408248290463863);  sub_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_38: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_19);  mul_tensor_19 = None
        sum_39: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_38: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_156: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_38, [1568, 4, 16, 16]);  div_38 = None
        view_800: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_156, [6272, 16, 16]);  expand_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_794: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_342, [25088, 24]);  add_342 = None
        permute_372: "f32[24, 24]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        mm_77: "f32[25088, 24]" = torch.ops.aten.mm.default(view_794, permute_372);  view_794 = permute_372 = None
        view_795: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_77, [1568, 16, 24]);  mm_77 = None
        view_796: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_795, [1568, 16, 4, -1]);  view_795 = None
        permute_373: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_796, [0, 2, 1, 3]);  view_796 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_157: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_373, [1568, 4, 16, 6]);  permute_373 = None
        clone_297: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_801: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_297, [6272, 16, 6]);  clone_297 = None
        bmm_77: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_800, view_801);  view_800 = view_801 = None
        view_802: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_77, [1568, 4, 16, 6]);  bmm_77 = None
        permute_375: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None
        clone_298: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
        view_803: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_298, [1568, 16, 24]);  clone_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_804: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_803, [25088, 24]);  view_803 = None
        permute_376: "f32[24, 24]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[25088, 24]" = torch.ops.aten.mm.default(view_804, permute_376);  view_804 = permute_376 = None
        add_tensor_36: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_36, arg213_1);  mm_default_36 = arg213_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_805: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_36, [1568, 16, 24]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_343: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_330, view_805);  add_330 = view_805 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_299: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_343, memory_format = torch.contiguous_format)
        var_mean_101 = torch.ops.aten.var_mean.correction(clone_299, [2], correction = 0, keepdim = True)
        getitem_280: "f32[1568, 16, 1]" = var_mean_101[0]
        getitem_281: "f32[1568, 16, 1]" = var_mean_101[1];  var_mean_101 = None
        sub_140: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_299, getitem_281);  clone_299 = getitem_281 = None
        add_344: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_280, 1e-05);  getitem_280 = None
        rsqrt_101: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_344);  add_344 = None
        mul_355: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_101);  sub_140 = rsqrt_101 = None
        mul_356: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_355, arg214_1);  mul_355 = arg214_1 = None
        add_345: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_356, arg215_1);  mul_356 = arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_806: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_345, [25088, 24]);  add_345 = None
        permute_377: "f32[24, 96]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[25088, 96]" = torch.ops.aten.mm.default(view_806, permute_377);  view_806 = permute_377 = None
        add_tensor_35: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_35, arg217_1);  mm_default_35 = arg217_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_807: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_35, [1568, 16, 96]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_357: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_807, 0.5)
        mul_358: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_807, 0.7071067811865476);  view_807 = None
        erf_38: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_358);  mul_358 = None
        add_346: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_359: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_357, add_346);  mul_357 = add_346 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_808: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_359, [25088, 96]);  mul_359 = None
        permute_378: "f32[96, 24]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[25088, 24]" = torch.ops.aten.mm.default(view_808, permute_378);  view_808 = permute_378 = None
        add_tensor_34: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_34, arg219_1);  mm_default_34 = arg219_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_809: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_34, [1568, 16, 24]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_347: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_343, view_809);  add_343 = view_809 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_302: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_347, memory_format = torch.contiguous_format)
        var_mean_102 = torch.ops.aten.var_mean.correction(clone_302, [2], correction = 0, keepdim = True)
        getitem_282: "f32[1568, 16, 1]" = var_mean_102[0]
        getitem_283: "f32[1568, 16, 1]" = var_mean_102[1];  var_mean_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_137: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_336, getitem_275);  getitem_275 = None
        add_337: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_274, 1e-05);  getitem_274 = None
        rsqrt_99: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_337);  add_337 = None
        mul_347: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_99);  sub_137 = rsqrt_99 = None
        mul_348: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_347, arg202_1);  mul_347 = arg202_1 = None
        add_338: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_348, arg203_1);  mul_348 = arg203_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_787: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_338, [1576, 384]);  add_338 = None
        permute_368: "f32[384, 1536]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_787, permute_368);  view_787 = permute_368 = None
        add_tensor_33: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_33, arg205_1);  mm_default_33 = arg205_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_788: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 197, 1536]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_349: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_788, 0.5)
        mul_350: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_788, 0.7071067811865476);  view_788 = None
        erf_37: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_339: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_351: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_349, add_339);  mul_349 = add_339 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_789: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_351, [1576, 1536]);  mul_351 = None
        permute_369: "f32[1536, 384]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[1576, 384]" = torch.ops.aten.mm.default(view_789, permute_369);  view_789 = permute_369 = None
        add_tensor_32: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_32, arg207_1);  mm_default_32 = arg207_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_790: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 197, 384]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_340: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_336, view_790);  add_336 = view_790 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_83: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_340, 1, 0, 1)
        slice_85: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_340, 1, 1, 9223372036854775807);  add_340 = None
        sub_141: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_302, getitem_283);  clone_302 = getitem_283 = None
        add_348: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-05);  getitem_282 = None
        rsqrt_102: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
        mul_360: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_102);  sub_141 = rsqrt_102 = None
        mul_361: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_360, arg220_1);  mul_360 = arg220_1 = None
        add_349: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_361, arg221_1);  mul_361 = arg221_1 = None
        view_810: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_349, [8, 196, -1]);  add_349 = None
        view_811: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_810, [1568, 384]);  view_810 = None
        permute_379: "f32[384, 384]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[1568, 384]" = torch.ops.aten.mm.default(view_811, permute_379);  view_811 = permute_379 = None
        add_tensor_31: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_31, arg223_1);  mm_default_31 = arg223_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_812: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 196, 384]);  add_tensor_31 = None
        add_350: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_85, view_812);  slice_85 = view_812 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_21: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_83, add_350], 1);  slice_83 = add_350 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_103 = torch.ops.aten.var_mean.correction(cat_21, [2], correction = 0, keepdim = True)
        getitem_284: "f32[8, 197, 1]" = var_mean_103[0]
        getitem_285: "f32[8, 197, 1]" = var_mean_103[1];  var_mean_103 = None
        sub_142: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_21, getitem_285);  getitem_285 = None
        add_351: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-05);  getitem_284 = None
        rsqrt_103: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        mul_362: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_103);  sub_142 = rsqrt_103 = None
        mul_363: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_362, arg224_1);  mul_362 = arg224_1 = None
        add_352: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_363, arg225_1);  mul_363 = arg225_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_813: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_352, [1576, 384])
        permute_380: "f32[384, 768]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        mm_78: "f32[1576, 768]" = torch.ops.aten.mm.default(view_813, permute_380);  view_813 = permute_380 = None
        view_814: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_78, [8, 197, 768]);  mm_78 = None
        view_815: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_814, [8, 197, 2, 6, 64]);  view_814 = None
        permute_381: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_815, [2, 0, 3, 1, 4]);  view_815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_39 = torch.ops.aten.unbind.int(permute_381);  permute_381 = None
        getitem_286: "f32[8, 6, 197, 64]" = unbind_39[0]
        getitem_287: "f32[8, 6, 197, 64]" = unbind_39[1];  unbind_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_158: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_286, [8, 6, 197, 64]);  getitem_286 = None
        clone_303: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_158, memory_format = torch.contiguous_format);  expand_158 = None
        view_819: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_303, [48, 197, 64]);  clone_303 = None
        permute_384: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_287, [0, 1, 3, 2]);  getitem_287 = None
        expand_159: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_384, [8, 6, 64, 197]);  permute_384 = None
        clone_304: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_820: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_304, [48, 64, 197]);  clone_304 = None
        bmm_78: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_819, view_820);  view_819 = view_820 = None
        view_821: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_78, [8, 6, 197, 197]);  bmm_78 = None
        
        # No stacktrace found for following nodes
        mul_tensor_16: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_821, 1);  view_821 = None
        amax_default_8: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_16, [-1], True)
        sub_tensor_8: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_16, amax_default_8);  mul_tensor_16 = amax_default_8 = None
        mul_tensor_17: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.125);  sub_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_39: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_17);  mul_tensor_17 = None
        sum_40: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_39: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_39, sum_40);  exp_39 = sum_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_160: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_39, [8, 6, 197, 197]);  div_39 = None
        view_822: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_160, [48, 197, 197]);  expand_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_816: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_352, [1576, 384]);  add_352 = None
        permute_382: "f32[384, 384]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        mm_79: "f32[1576, 384]" = torch.ops.aten.mm.default(view_816, permute_382);  view_816 = permute_382 = None
        view_817: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_79, [8, 197, 384]);  mm_79 = None
        view_818: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_817, [8, 197, 6, -1]);  view_817 = None
        permute_383: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_818, [0, 2, 1, 3]);  view_818 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_161: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_383, [8, 6, 197, 64]);  permute_383 = None
        clone_305: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_823: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_305, [48, 197, 64]);  clone_305 = None
        bmm_79: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_822, view_823);  view_822 = view_823 = None
        view_824: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_79, [8, 6, 197, 64]);  bmm_79 = None
        permute_385: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_824, [0, 2, 1, 3]);  view_824 = None
        clone_306: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_825: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_306, [8, 197, 384]);  clone_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_826: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_825, [1576, 384]);  view_825 = None
        permute_386: "f32[384, 384]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[1576, 384]" = torch.ops.aten.mm.default(view_826, permute_386);  view_826 = permute_386 = None
        add_tensor_30: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_30, arg229_1);  mm_default_30 = arg229_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_827: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 197, 384]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_353: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_21, view_827);  cat_21 = view_827 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_104 = torch.ops.aten.var_mean.correction(add_353, [2], correction = 0, keepdim = True)
        getitem_288: "f32[8, 197, 1]" = var_mean_104[0]
        getitem_289: "f32[8, 197, 1]" = var_mean_104[1];  var_mean_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_309: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_347, memory_format = torch.contiguous_format)
        var_mean_105 = torch.ops.aten.var_mean.correction(clone_309, [2], correction = 0, keepdim = True)
        getitem_290: "f32[1568, 16, 1]" = var_mean_105[0]
        getitem_291: "f32[1568, 16, 1]" = var_mean_105[1];  var_mean_105 = None
        sub_145: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_309, getitem_291);  clone_309 = getitem_291 = None
        add_358: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-05);  getitem_290 = None
        rsqrt_105: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_358);  add_358 = None
        mul_370: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_105);  sub_145 = rsqrt_105 = None
        mul_371: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_370, arg236_1);  mul_370 = arg236_1 = None
        add_359: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_371, arg237_1);  mul_371 = arg237_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_832: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_359, [25088, 24])
        permute_389: "f32[24, 48]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        mm_80: "f32[25088, 48]" = torch.ops.aten.mm.default(view_832, permute_389);  view_832 = permute_389 = None
        view_833: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_80, [1568, 16, 48]);  mm_80 = None
        view_834: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_833, [1568, 16, 2, 4, 6]);  view_833 = None
        permute_390: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_834, [2, 0, 3, 1, 4]);  view_834 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_40 = torch.ops.aten.unbind.int(permute_390);  permute_390 = None
        getitem_292: "f32[1568, 4, 16, 6]" = unbind_40[0]
        getitem_293: "f32[1568, 4, 16, 6]" = unbind_40[1];  unbind_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_162: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_292, [1568, 4, 16, 6]);  getitem_292 = None
        clone_310: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_838: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_310, [6272, 16, 6]);  clone_310 = None
        permute_393: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_293, [0, 1, 3, 2]);  getitem_293 = None
        expand_163: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_393, [1568, 4, 6, 16]);  permute_393 = None
        clone_311: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_839: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_311, [6272, 6, 16]);  clone_311 = None
        bmm_80: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_838, view_839);  view_838 = view_839 = None
        view_840: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_80, [1568, 4, 16, 16]);  bmm_80 = None
        
        # No stacktrace found for following nodes
        mul_tensor_14: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_840, 1);  view_840 = None
        amax_default_7: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_14, [-1], True)
        sub_tensor_7: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = amax_default_7 = None
        mul_tensor_15: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_7, 0.408248290463863);  sub_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_40: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_15);  mul_tensor_15 = None
        sum_41: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_40: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_164: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_40, [1568, 4, 16, 16]);  div_40 = None
        view_841: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_164, [6272, 16, 16]);  expand_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_835: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_359, [25088, 24]);  add_359 = None
        permute_391: "f32[24, 24]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_81: "f32[25088, 24]" = torch.ops.aten.mm.default(view_835, permute_391);  view_835 = permute_391 = None
        view_836: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_81, [1568, 16, 24]);  mm_81 = None
        view_837: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_836, [1568, 16, 4, -1]);  view_836 = None
        permute_392: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_837, [0, 2, 1, 3]);  view_837 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_165: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_392, [1568, 4, 16, 6]);  permute_392 = None
        clone_312: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_842: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_312, [6272, 16, 6]);  clone_312 = None
        bmm_81: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_841, view_842);  view_841 = view_842 = None
        view_843: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_81, [1568, 4, 16, 6]);  bmm_81 = None
        permute_394: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_843, [0, 2, 1, 3]);  view_843 = None
        clone_313: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_394, memory_format = torch.contiguous_format);  permute_394 = None
        view_844: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_313, [1568, 16, 24]);  clone_313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_845: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_844, [25088, 24]);  view_844 = None
        permute_395: "f32[24, 24]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[25088, 24]" = torch.ops.aten.mm.default(view_845, permute_395);  view_845 = permute_395 = None
        add_tensor_29: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_29, arg241_1);  mm_default_29 = arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_846: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_29, [1568, 16, 24]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_360: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_347, view_846);  add_347 = view_846 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_314: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_360, memory_format = torch.contiguous_format)
        var_mean_106 = torch.ops.aten.var_mean.correction(clone_314, [2], correction = 0, keepdim = True)
        getitem_294: "f32[1568, 16, 1]" = var_mean_106[0]
        getitem_295: "f32[1568, 16, 1]" = var_mean_106[1];  var_mean_106 = None
        sub_147: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_314, getitem_295);  clone_314 = getitem_295 = None
        add_361: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_294, 1e-05);  getitem_294 = None
        rsqrt_106: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
        mul_373: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_106);  sub_147 = rsqrt_106 = None
        mul_374: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_373, arg242_1);  mul_373 = arg242_1 = None
        add_362: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_374, arg243_1);  mul_374 = arg243_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_847: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_362, [25088, 24]);  add_362 = None
        permute_396: "f32[24, 96]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[25088, 96]" = torch.ops.aten.mm.default(view_847, permute_396);  view_847 = permute_396 = None
        add_tensor_28: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_28, arg245_1);  mm_default_28 = arg245_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_848: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_28, [1568, 16, 96]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_375: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_848, 0.5)
        mul_376: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_848, 0.7071067811865476);  view_848 = None
        erf_40: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_376);  mul_376 = None
        add_363: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_377: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_375, add_363);  mul_375 = add_363 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_849: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_377, [25088, 96]);  mul_377 = None
        permute_397: "f32[96, 24]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[25088, 24]" = torch.ops.aten.mm.default(view_849, permute_397);  view_849 = permute_397 = None
        add_tensor_27: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_27, arg247_1);  mm_default_27 = arg247_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_850: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_27, [1568, 16, 24]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_364: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_360, view_850);  add_360 = view_850 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_317: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_364, memory_format = torch.contiguous_format)
        var_mean_107 = torch.ops.aten.var_mean.correction(clone_317, [2], correction = 0, keepdim = True)
        getitem_296: "f32[1568, 16, 1]" = var_mean_107[0]
        getitem_297: "f32[1568, 16, 1]" = var_mean_107[1];  var_mean_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_144: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_353, getitem_289);  getitem_289 = None
        add_354: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_104: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_354);  add_354 = None
        mul_365: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_144, rsqrt_104);  sub_144 = rsqrt_104 = None
        mul_366: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_365, arg230_1);  mul_365 = arg230_1 = None
        add_355: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_366, arg231_1);  mul_366 = arg231_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_828: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_355, [1576, 384]);  add_355 = None
        permute_387: "f32[384, 1536]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_828, permute_387);  view_828 = permute_387 = None
        add_tensor_26: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_26, arg233_1);  mm_default_26 = arg233_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_829: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 197, 1536]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_367: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_829, 0.5)
        mul_368: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_829, 0.7071067811865476);  view_829 = None
        erf_39: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_368);  mul_368 = None
        add_356: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_369: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_367, add_356);  mul_367 = add_356 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_830: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_369, [1576, 1536]);  mul_369 = None
        permute_388: "f32[1536, 384]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[1576, 384]" = torch.ops.aten.mm.default(view_830, permute_388);  view_830 = permute_388 = None
        add_tensor_25: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_25, arg235_1);  mm_default_25 = arg235_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_831: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 197, 384]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_357: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_353, view_831);  add_353 = view_831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_87: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_357, 1, 0, 1)
        slice_89: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_357, 1, 1, 9223372036854775807);  add_357 = None
        sub_148: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_317, getitem_297);  clone_317 = getitem_297 = None
        add_365: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-05);  getitem_296 = None
        rsqrt_107: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_365);  add_365 = None
        mul_378: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_107);  sub_148 = rsqrt_107 = None
        mul_379: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_378, arg248_1);  mul_378 = arg248_1 = None
        add_366: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_379, arg249_1);  mul_379 = arg249_1 = None
        view_851: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_366, [8, 196, -1]);  add_366 = None
        view_852: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_851, [1568, 384]);  view_851 = None
        permute_398: "f32[384, 384]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[1568, 384]" = torch.ops.aten.mm.default(view_852, permute_398);  view_852 = permute_398 = None
        add_tensor_24: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_24, arg251_1);  mm_default_24 = arg251_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_853: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 196, 384]);  add_tensor_24 = None
        add_367: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_89, view_853);  slice_89 = view_853 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_22: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_87, add_367], 1);  slice_87 = add_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_108 = torch.ops.aten.var_mean.correction(cat_22, [2], correction = 0, keepdim = True)
        getitem_298: "f32[8, 197, 1]" = var_mean_108[0]
        getitem_299: "f32[8, 197, 1]" = var_mean_108[1];  var_mean_108 = None
        sub_149: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_22, getitem_299);  getitem_299 = None
        add_368: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_298, 1e-05);  getitem_298 = None
        rsqrt_108: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
        mul_380: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_108);  sub_149 = rsqrt_108 = None
        mul_381: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_380, arg252_1);  mul_380 = arg252_1 = None
        add_369: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_381, arg253_1);  mul_381 = arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_854: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_369, [1576, 384])
        permute_399: "f32[384, 768]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
        mm_82: "f32[1576, 768]" = torch.ops.aten.mm.default(view_854, permute_399);  view_854 = permute_399 = None
        view_855: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_82, [8, 197, 768]);  mm_82 = None
        view_856: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_855, [8, 197, 2, 6, 64]);  view_855 = None
        permute_400: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_856, [2, 0, 3, 1, 4]);  view_856 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_41 = torch.ops.aten.unbind.int(permute_400);  permute_400 = None
        getitem_300: "f32[8, 6, 197, 64]" = unbind_41[0]
        getitem_301: "f32[8, 6, 197, 64]" = unbind_41[1];  unbind_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_166: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_300, [8, 6, 197, 64]);  getitem_300 = None
        clone_318: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_166, memory_format = torch.contiguous_format);  expand_166 = None
        view_860: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_318, [48, 197, 64]);  clone_318 = None
        permute_403: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_301, [0, 1, 3, 2]);  getitem_301 = None
        expand_167: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_403, [8, 6, 64, 197]);  permute_403 = None
        clone_319: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_167, memory_format = torch.contiguous_format);  expand_167 = None
        view_861: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_319, [48, 64, 197]);  clone_319 = None
        bmm_82: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_860, view_861);  view_860 = view_861 = None
        view_862: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_82, [8, 6, 197, 197]);  bmm_82 = None
        
        # No stacktrace found for following nodes
        mul_tensor_12: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_862, 1);  view_862 = None
        amax_default_6: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_12, [-1], True)
        sub_tensor_6: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = amax_default_6 = None
        mul_tensor_13: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.125);  sub_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_41: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_13);  mul_tensor_13 = None
        sum_42: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_41: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_41, sum_42);  exp_41 = sum_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_168: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_41, [8, 6, 197, 197]);  div_41 = None
        view_863: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_168, [48, 197, 197]);  expand_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_857: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_369, [1576, 384]);  add_369 = None
        permute_401: "f32[384, 384]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        mm_83: "f32[1576, 384]" = torch.ops.aten.mm.default(view_857, permute_401);  view_857 = permute_401 = None
        view_858: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_83, [8, 197, 384]);  mm_83 = None
        view_859: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_858, [8, 197, 6, -1]);  view_858 = None
        permute_402: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_859, [0, 2, 1, 3]);  view_859 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_169: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_402, [8, 6, 197, 64]);  permute_402 = None
        clone_320: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_864: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_320, [48, 197, 64]);  clone_320 = None
        bmm_83: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_863, view_864);  view_863 = view_864 = None
        view_865: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_83, [8, 6, 197, 64]);  bmm_83 = None
        permute_404: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_865, [0, 2, 1, 3]);  view_865 = None
        clone_321: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_404, memory_format = torch.contiguous_format);  permute_404 = None
        view_866: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_321, [8, 197, 384]);  clone_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_867: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_866, [1576, 384]);  view_866 = None
        permute_405: "f32[384, 384]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[1576, 384]" = torch.ops.aten.mm.default(view_867, permute_405);  view_867 = permute_405 = None
        add_tensor_23: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_23, arg257_1);  mm_default_23 = arg257_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_868: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 197, 384]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_370: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_22, view_868);  cat_22 = view_868 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_109 = torch.ops.aten.var_mean.correction(add_370, [2], correction = 0, keepdim = True)
        getitem_302: "f32[8, 197, 1]" = var_mean_109[0]
        getitem_303: "f32[8, 197, 1]" = var_mean_109[1];  var_mean_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_324: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_364, memory_format = torch.contiguous_format)
        var_mean_110 = torch.ops.aten.var_mean.correction(clone_324, [2], correction = 0, keepdim = True)
        getitem_304: "f32[1568, 16, 1]" = var_mean_110[0]
        getitem_305: "f32[1568, 16, 1]" = var_mean_110[1];  var_mean_110 = None
        sub_152: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_324, getitem_305);  clone_324 = getitem_305 = None
        add_375: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_304, 1e-05);  getitem_304 = None
        rsqrt_110: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_375);  add_375 = None
        mul_388: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_110);  sub_152 = rsqrt_110 = None
        mul_389: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_388, arg264_1);  mul_388 = arg264_1 = None
        add_376: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_389, arg265_1);  mul_389 = arg265_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_873: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_376, [25088, 24])
        permute_408: "f32[24, 48]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        mm_84: "f32[25088, 48]" = torch.ops.aten.mm.default(view_873, permute_408);  view_873 = permute_408 = None
        view_874: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_84, [1568, 16, 48]);  mm_84 = None
        view_875: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_874, [1568, 16, 2, 4, 6]);  view_874 = None
        permute_409: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_875, [2, 0, 3, 1, 4]);  view_875 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_42 = torch.ops.aten.unbind.int(permute_409);  permute_409 = None
        getitem_306: "f32[1568, 4, 16, 6]" = unbind_42[0]
        getitem_307: "f32[1568, 4, 16, 6]" = unbind_42[1];  unbind_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_170: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_306, [1568, 4, 16, 6]);  getitem_306 = None
        clone_325: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_879: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_325, [6272, 16, 6]);  clone_325 = None
        permute_412: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_307, [0, 1, 3, 2]);  getitem_307 = None
        expand_171: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_412, [1568, 4, 6, 16]);  permute_412 = None
        clone_326: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_880: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_326, [6272, 6, 16]);  clone_326 = None
        bmm_84: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_879, view_880);  view_879 = view_880 = None
        view_881: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_84, [1568, 4, 16, 16]);  bmm_84 = None
        
        # No stacktrace found for following nodes
        mul_tensor_10: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_881, 1);  view_881 = None
        amax_default_5: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_10, [-1], True)
        sub_tensor_5: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_5, 0.408248290463863);  sub_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_42: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        sum_43: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_42: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_172: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_42, [1568, 4, 16, 16]);  div_42 = None
        view_882: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_172, [6272, 16, 16]);  expand_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_876: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_376, [25088, 24]);  add_376 = None
        permute_410: "f32[24, 24]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        mm_85: "f32[25088, 24]" = torch.ops.aten.mm.default(view_876, permute_410);  view_876 = permute_410 = None
        view_877: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_85, [1568, 16, 24]);  mm_85 = None
        view_878: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_877, [1568, 16, 4, -1]);  view_877 = None
        permute_411: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_878, [0, 2, 1, 3]);  view_878 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_173: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_411, [1568, 4, 16, 6]);  permute_411 = None
        clone_327: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_883: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_327, [6272, 16, 6]);  clone_327 = None
        bmm_85: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_882, view_883);  view_882 = view_883 = None
        view_884: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_85, [1568, 4, 16, 6]);  bmm_85 = None
        permute_413: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_884, [0, 2, 1, 3]);  view_884 = None
        clone_328: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_885: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_328, [1568, 16, 24]);  clone_328 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_886: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_885, [25088, 24]);  view_885 = None
        permute_414: "f32[24, 24]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[25088, 24]" = torch.ops.aten.mm.default(view_886, permute_414);  view_886 = permute_414 = None
        add_tensor_22: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_22, arg269_1);  mm_default_22 = arg269_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_887: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_22, [1568, 16, 24]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_377: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_364, view_887);  add_364 = view_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_329: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_377, memory_format = torch.contiguous_format)
        var_mean_111 = torch.ops.aten.var_mean.correction(clone_329, [2], correction = 0, keepdim = True)
        getitem_308: "f32[1568, 16, 1]" = var_mean_111[0]
        getitem_309: "f32[1568, 16, 1]" = var_mean_111[1];  var_mean_111 = None
        sub_154: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_329, getitem_309);  clone_329 = getitem_309 = None
        add_378: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-05);  getitem_308 = None
        rsqrt_111: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_378);  add_378 = None
        mul_391: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_111);  sub_154 = rsqrt_111 = None
        mul_392: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_391, arg270_1);  mul_391 = arg270_1 = None
        add_379: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_392, arg271_1);  mul_392 = arg271_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_888: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_379, [25088, 24]);  add_379 = None
        permute_415: "f32[24, 96]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[25088, 96]" = torch.ops.aten.mm.default(view_888, permute_415);  view_888 = permute_415 = None
        add_tensor_21: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_21, arg273_1);  mm_default_21 = arg273_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_889: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_21, [1568, 16, 96]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_393: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_889, 0.5)
        mul_394: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_889, 0.7071067811865476);  view_889 = None
        erf_42: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_394);  mul_394 = None
        add_380: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_395: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_393, add_380);  mul_393 = add_380 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_890: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_395, [25088, 96]);  mul_395 = None
        permute_416: "f32[96, 24]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[25088, 24]" = torch.ops.aten.mm.default(view_890, permute_416);  view_890 = permute_416 = None
        add_tensor_20: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_20, arg275_1);  mm_default_20 = arg275_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_891: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_20, [1568, 16, 24]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_381: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_377, view_891);  add_377 = view_891 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_332: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_381, memory_format = torch.contiguous_format)
        var_mean_112 = torch.ops.aten.var_mean.correction(clone_332, [2], correction = 0, keepdim = True)
        getitem_310: "f32[1568, 16, 1]" = var_mean_112[0]
        getitem_311: "f32[1568, 16, 1]" = var_mean_112[1];  var_mean_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_151: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_370, getitem_303);  getitem_303 = None
        add_371: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-05);  getitem_302 = None
        rsqrt_109: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_371);  add_371 = None
        mul_383: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_109);  sub_151 = rsqrt_109 = None
        mul_384: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_383, arg258_1);  mul_383 = arg258_1 = None
        add_372: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_384, arg259_1);  mul_384 = arg259_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_869: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_372, [1576, 384]);  add_372 = None
        permute_406: "f32[384, 1536]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_869, permute_406);  view_869 = permute_406 = None
        add_tensor_19: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_19, arg261_1);  mm_default_19 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_870: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 197, 1536]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_385: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_870, 0.5)
        mul_386: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_870, 0.7071067811865476);  view_870 = None
        erf_41: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_386);  mul_386 = None
        add_373: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_387: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_385, add_373);  mul_385 = add_373 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_871: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_387, [1576, 1536]);  mul_387 = None
        permute_407: "f32[1536, 384]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[1576, 384]" = torch.ops.aten.mm.default(view_871, permute_407);  view_871 = permute_407 = None
        add_tensor_18: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_18, arg263_1);  mm_default_18 = arg263_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_872: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 197, 384]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_374: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_370, view_872);  add_370 = view_872 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_91: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_374, 1, 0, 1)
        slice_93: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_374, 1, 1, 9223372036854775807);  add_374 = None
        sub_155: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_332, getitem_311);  clone_332 = getitem_311 = None
        add_382: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_112: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_382);  add_382 = None
        mul_396: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_112);  sub_155 = rsqrt_112 = None
        mul_397: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_396, arg276_1);  mul_396 = arg276_1 = None
        add_383: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_397, arg277_1);  mul_397 = arg277_1 = None
        view_892: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_383, [8, 196, -1]);  add_383 = None
        view_893: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_892, [1568, 384]);  view_892 = None
        permute_417: "f32[384, 384]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[1568, 384]" = torch.ops.aten.mm.default(view_893, permute_417);  view_893 = permute_417 = None
        add_tensor_17: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_17, arg279_1);  mm_default_17 = arg279_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_894: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 196, 384]);  add_tensor_17 = None
        add_384: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_93, view_894);  slice_93 = view_894 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_23: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_91, add_384], 1);  slice_91 = add_384 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_113 = torch.ops.aten.var_mean.correction(cat_23, [2], correction = 0, keepdim = True)
        getitem_312: "f32[8, 197, 1]" = var_mean_113[0]
        getitem_313: "f32[8, 197, 1]" = var_mean_113[1];  var_mean_113 = None
        sub_156: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_23, getitem_313);  getitem_313 = None
        add_385: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_312, 1e-05);  getitem_312 = None
        rsqrt_113: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_385);  add_385 = None
        mul_398: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_156, rsqrt_113);  sub_156 = rsqrt_113 = None
        mul_399: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_398, arg280_1);  mul_398 = arg280_1 = None
        add_386: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_399, arg281_1);  mul_399 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_895: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_386, [1576, 384])
        permute_418: "f32[384, 768]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        mm_86: "f32[1576, 768]" = torch.ops.aten.mm.default(view_895, permute_418);  view_895 = permute_418 = None
        view_896: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_86, [8, 197, 768]);  mm_86 = None
        view_897: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_896, [8, 197, 2, 6, 64]);  view_896 = None
        permute_419: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_897, [2, 0, 3, 1, 4]);  view_897 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_43 = torch.ops.aten.unbind.int(permute_419);  permute_419 = None
        getitem_314: "f32[8, 6, 197, 64]" = unbind_43[0]
        getitem_315: "f32[8, 6, 197, 64]" = unbind_43[1];  unbind_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_174: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_314, [8, 6, 197, 64]);  getitem_314 = None
        clone_333: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_174, memory_format = torch.contiguous_format);  expand_174 = None
        view_901: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_333, [48, 197, 64]);  clone_333 = None
        permute_422: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_315, [0, 1, 3, 2]);  getitem_315 = None
        expand_175: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_422, [8, 6, 64, 197]);  permute_422 = None
        clone_334: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_175, memory_format = torch.contiguous_format);  expand_175 = None
        view_902: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_334, [48, 64, 197]);  clone_334 = None
        bmm_86: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_901, view_902);  view_901 = view_902 = None
        view_903: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_86, [8, 6, 197, 197]);  bmm_86 = None
        
        # No stacktrace found for following nodes
        mul_tensor_8: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_903, 1);  view_903 = None
        amax_default_4: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_8, [-1], True)
        sub_tensor_4: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.125);  sub_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_43: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        sum_44: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_43: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_43, sum_44);  exp_43 = sum_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_176: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_43, [8, 6, 197, 197]);  div_43 = None
        view_904: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_176, [48, 197, 197]);  expand_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_898: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_386, [1576, 384]);  add_386 = None
        permute_420: "f32[384, 384]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        mm_87: "f32[1576, 384]" = torch.ops.aten.mm.default(view_898, permute_420);  view_898 = permute_420 = None
        view_899: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_87, [8, 197, 384]);  mm_87 = None
        view_900: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_899, [8, 197, 6, -1]);  view_899 = None
        permute_421: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_900, [0, 2, 1, 3]);  view_900 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_177: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_421, [8, 6, 197, 64]);  permute_421 = None
        clone_335: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_905: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_335, [48, 197, 64]);  clone_335 = None
        bmm_87: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_904, view_905);  view_904 = view_905 = None
        view_906: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_87, [8, 6, 197, 64]);  bmm_87 = None
        permute_423: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_906, [0, 2, 1, 3]);  view_906 = None
        clone_336: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
        view_907: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_336, [8, 197, 384]);  clone_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_908: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_907, [1576, 384]);  view_907 = None
        permute_424: "f32[384, 384]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[1576, 384]" = torch.ops.aten.mm.default(view_908, permute_424);  view_908 = permute_424 = None
        add_tensor_16: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_16, arg285_1);  mm_default_16 = arg285_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_909: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 197, 384]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_387: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_23, view_909);  cat_23 = view_909 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_114 = torch.ops.aten.var_mean.correction(add_387, [2], correction = 0, keepdim = True)
        getitem_316: "f32[8, 197, 1]" = var_mean_114[0]
        getitem_317: "f32[8, 197, 1]" = var_mean_114[1];  var_mean_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_339: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_381, memory_format = torch.contiguous_format)
        var_mean_115 = torch.ops.aten.var_mean.correction(clone_339, [2], correction = 0, keepdim = True)
        getitem_318: "f32[1568, 16, 1]" = var_mean_115[0]
        getitem_319: "f32[1568, 16, 1]" = var_mean_115[1];  var_mean_115 = None
        sub_159: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_339, getitem_319);  clone_339 = getitem_319 = None
        add_392: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_318, 1e-05);  getitem_318 = None
        rsqrt_115: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_392);  add_392 = None
        mul_406: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_159, rsqrt_115);  sub_159 = rsqrt_115 = None
        mul_407: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_406, arg292_1);  mul_406 = arg292_1 = None
        add_393: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_407, arg293_1);  mul_407 = arg293_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_914: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_393, [25088, 24])
        permute_427: "f32[24, 48]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        mm_88: "f32[25088, 48]" = torch.ops.aten.mm.default(view_914, permute_427);  view_914 = permute_427 = None
        view_915: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_88, [1568, 16, 48]);  mm_88 = None
        view_916: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_915, [1568, 16, 2, 4, 6]);  view_915 = None
        permute_428: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_916, [2, 0, 3, 1, 4]);  view_916 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_44 = torch.ops.aten.unbind.int(permute_428);  permute_428 = None
        getitem_320: "f32[1568, 4, 16, 6]" = unbind_44[0]
        getitem_321: "f32[1568, 4, 16, 6]" = unbind_44[1];  unbind_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_178: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_320, [1568, 4, 16, 6]);  getitem_320 = None
        clone_340: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_920: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_340, [6272, 16, 6]);  clone_340 = None
        permute_431: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_321, [0, 1, 3, 2]);  getitem_321 = None
        expand_179: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_431, [1568, 4, 6, 16]);  permute_431 = None
        clone_341: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_921: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_341, [6272, 6, 16]);  clone_341 = None
        bmm_88: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_920, view_921);  view_920 = view_921 = None
        view_922: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_88, [1568, 4, 16, 16]);  bmm_88 = None
        
        # No stacktrace found for following nodes
        mul_tensor_6: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_922, 1);  view_922 = None
        amax_default_3: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_6, [-1], True)
        sub_tensor_3: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_3, 0.408248290463863);  sub_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_44: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        sum_45: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_44: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_180: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_44, [1568, 4, 16, 16]);  div_44 = None
        view_923: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_180, [6272, 16, 16]);  expand_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_917: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_393, [25088, 24]);  add_393 = None
        permute_429: "f32[24, 24]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        mm_89: "f32[25088, 24]" = torch.ops.aten.mm.default(view_917, permute_429);  view_917 = permute_429 = None
        view_918: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_89, [1568, 16, 24]);  mm_89 = None
        view_919: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_918, [1568, 16, 4, -1]);  view_918 = None
        permute_430: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_919, [0, 2, 1, 3]);  view_919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_181: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_430, [1568, 4, 16, 6]);  permute_430 = None
        clone_342: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_924: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_342, [6272, 16, 6]);  clone_342 = None
        bmm_89: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_923, view_924);  view_923 = view_924 = None
        view_925: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_89, [1568, 4, 16, 6]);  bmm_89 = None
        permute_432: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_925, [0, 2, 1, 3]);  view_925 = None
        clone_343: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_432, memory_format = torch.contiguous_format);  permute_432 = None
        view_926: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_343, [1568, 16, 24]);  clone_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_927: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_926, [25088, 24]);  view_926 = None
        permute_433: "f32[24, 24]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[25088, 24]" = torch.ops.aten.mm.default(view_927, permute_433);  view_927 = permute_433 = None
        add_tensor_15: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_15, arg297_1);  mm_default_15 = arg297_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_928: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_15, [1568, 16, 24]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_394: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_381, view_928);  add_381 = view_928 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_344: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_394, memory_format = torch.contiguous_format)
        var_mean_116 = torch.ops.aten.var_mean.correction(clone_344, [2], correction = 0, keepdim = True)
        getitem_322: "f32[1568, 16, 1]" = var_mean_116[0]
        getitem_323: "f32[1568, 16, 1]" = var_mean_116[1];  var_mean_116 = None
        sub_161: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_344, getitem_323);  clone_344 = getitem_323 = None
        add_395: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_322, 1e-05);  getitem_322 = None
        rsqrt_116: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_395);  add_395 = None
        mul_409: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_116);  sub_161 = rsqrt_116 = None
        mul_410: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_409, arg298_1);  mul_409 = arg298_1 = None
        add_396: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_410, arg299_1);  mul_410 = arg299_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_929: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_396, [25088, 24]);  add_396 = None
        permute_434: "f32[24, 96]" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[25088, 96]" = torch.ops.aten.mm.default(view_929, permute_434);  view_929 = permute_434 = None
        add_tensor_14: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_14, arg301_1);  mm_default_14 = arg301_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_930: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_14, [1568, 16, 96]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_411: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_930, 0.5)
        mul_412: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_930, 0.7071067811865476);  view_930 = None
        erf_44: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_412);  mul_412 = None
        add_397: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_413: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_411, add_397);  mul_411 = add_397 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_931: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_413, [25088, 96]);  mul_413 = None
        permute_435: "f32[96, 24]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[25088, 24]" = torch.ops.aten.mm.default(view_931, permute_435);  view_931 = permute_435 = None
        add_tensor_13: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_13, arg303_1);  mm_default_13 = arg303_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_932: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_13, [1568, 16, 24]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_398: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_394, view_932);  add_394 = view_932 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_347: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_398, memory_format = torch.contiguous_format)
        var_mean_117 = torch.ops.aten.var_mean.correction(clone_347, [2], correction = 0, keepdim = True)
        getitem_324: "f32[1568, 16, 1]" = var_mean_117[0]
        getitem_325: "f32[1568, 16, 1]" = var_mean_117[1];  var_mean_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_158: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_387, getitem_317);  getitem_317 = None
        add_388: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_316, 1e-05);  getitem_316 = None
        rsqrt_114: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_388);  add_388 = None
        mul_401: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_114);  sub_158 = rsqrt_114 = None
        mul_402: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_401, arg286_1);  mul_401 = arg286_1 = None
        add_389: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_402, arg287_1);  mul_402 = arg287_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_910: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_389, [1576, 384]);  add_389 = None
        permute_425: "f32[384, 1536]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_910, permute_425);  view_910 = permute_425 = None
        add_tensor_12: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_12, arg289_1);  mm_default_12 = arg289_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_911: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 197, 1536]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_403: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_911, 0.5)
        mul_404: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_911, 0.7071067811865476);  view_911 = None
        erf_43: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_404);  mul_404 = None
        add_390: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_405: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_403, add_390);  mul_403 = add_390 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_912: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_405, [1576, 1536]);  mul_405 = None
        permute_426: "f32[1536, 384]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[1576, 384]" = torch.ops.aten.mm.default(view_912, permute_426);  view_912 = permute_426 = None
        add_tensor_11: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_11, arg291_1);  mm_default_11 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_913: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 197, 384]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_391: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_387, view_913);  add_387 = view_913 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_95: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_391, 1, 0, 1)
        slice_97: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_391, 1, 1, 9223372036854775807);  add_391 = None
        sub_162: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_347, getitem_325);  clone_347 = getitem_325 = None
        add_399: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_324, 1e-05);  getitem_324 = None
        rsqrt_117: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_399);  add_399 = None
        mul_414: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_117);  sub_162 = rsqrt_117 = None
        mul_415: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_414, arg304_1);  mul_414 = arg304_1 = None
        add_400: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_415, arg305_1);  mul_415 = arg305_1 = None
        view_933: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_400, [8, 196, -1]);  add_400 = None
        view_934: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_933, [1568, 384]);  view_933 = None
        permute_436: "f32[384, 384]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[1568, 384]" = torch.ops.aten.mm.default(view_934, permute_436);  view_934 = permute_436 = None
        add_tensor_10: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_10, arg307_1);  mm_default_10 = arg307_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_935: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 384]);  add_tensor_10 = None
        add_401: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_97, view_935);  slice_97 = view_935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_24: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_95, add_401], 1);  slice_95 = add_401 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_118 = torch.ops.aten.var_mean.correction(cat_24, [2], correction = 0, keepdim = True)
        getitem_326: "f32[8, 197, 1]" = var_mean_118[0]
        getitem_327: "f32[8, 197, 1]" = var_mean_118[1];  var_mean_118 = None
        sub_163: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_24, getitem_327);  getitem_327 = None
        add_402: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_326, 1e-05);  getitem_326 = None
        rsqrt_118: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_402);  add_402 = None
        mul_416: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_163, rsqrt_118);  sub_163 = rsqrt_118 = None
        mul_417: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_416, arg308_1);  mul_416 = arg308_1 = None
        add_403: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_417, arg309_1);  mul_417 = arg309_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_936: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_403, [1576, 384])
        permute_437: "f32[384, 768]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        mm_90: "f32[1576, 768]" = torch.ops.aten.mm.default(view_936, permute_437);  view_936 = permute_437 = None
        view_937: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_90, [8, 197, 768]);  mm_90 = None
        view_938: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_937, [8, 197, 2, 6, 64]);  view_937 = None
        permute_438: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_938, [2, 0, 3, 1, 4]);  view_938 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_45 = torch.ops.aten.unbind.int(permute_438);  permute_438 = None
        getitem_328: "f32[8, 6, 197, 64]" = unbind_45[0]
        getitem_329: "f32[8, 6, 197, 64]" = unbind_45[1];  unbind_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_182: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_328, [8, 6, 197, 64]);  getitem_328 = None
        clone_348: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_182, memory_format = torch.contiguous_format);  expand_182 = None
        view_942: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_348, [48, 197, 64]);  clone_348 = None
        permute_441: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_329, [0, 1, 3, 2]);  getitem_329 = None
        expand_183: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_441, [8, 6, 64, 197]);  permute_441 = None
        clone_349: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_943: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_349, [48, 64, 197]);  clone_349 = None
        bmm_90: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_942, view_943);  view_942 = view_943 = None
        view_944: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_90, [8, 6, 197, 197]);  bmm_90 = None
        
        # No stacktrace found for following nodes
        mul_tensor_4: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_944, 1);  view_944 = None
        amax_default_2: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor_4, [-1], True)
        sub_tensor_2: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.125);  sub_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_45: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        sum_46: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_45: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_45, sum_46);  exp_45 = sum_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_184: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_45, [8, 6, 197, 197]);  div_45 = None
        view_945: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_184, [48, 197, 197]);  expand_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_939: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_403, [1576, 384]);  add_403 = None
        permute_439: "f32[384, 384]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        mm_91: "f32[1576, 384]" = torch.ops.aten.mm.default(view_939, permute_439);  view_939 = permute_439 = None
        view_940: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_91, [8, 197, 384]);  mm_91 = None
        view_941: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_940, [8, 197, 6, -1]);  view_940 = None
        permute_440: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_941, [0, 2, 1, 3]);  view_941 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_185: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_440, [8, 6, 197, 64]);  permute_440 = None
        clone_350: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_946: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_350, [48, 197, 64]);  clone_350 = None
        bmm_91: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_945, view_946);  view_945 = view_946 = None
        view_947: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_91, [8, 6, 197, 64]);  bmm_91 = None
        permute_442: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_947, [0, 2, 1, 3]);  view_947 = None
        clone_351: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None
        view_948: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_351, [8, 197, 384]);  clone_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_949: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_948, [1576, 384]);  view_948 = None
        permute_443: "f32[384, 384]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[1576, 384]" = torch.ops.aten.mm.default(view_949, permute_443);  view_949 = permute_443 = None
        add_tensor_9: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_9, arg313_1);  mm_default_9 = arg313_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_950: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 197, 384]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_404: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_24, view_950);  cat_24 = view_950 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_119 = torch.ops.aten.var_mean.correction(add_404, [2], correction = 0, keepdim = True)
        getitem_330: "f32[8, 197, 1]" = var_mean_119[0]
        getitem_331: "f32[8, 197, 1]" = var_mean_119[1];  var_mean_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        clone_354: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_398, memory_format = torch.contiguous_format)
        var_mean_120 = torch.ops.aten.var_mean.correction(clone_354, [2], correction = 0, keepdim = True)
        getitem_332: "f32[1568, 16, 1]" = var_mean_120[0]
        getitem_333: "f32[1568, 16, 1]" = var_mean_120[1];  var_mean_120 = None
        sub_166: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_354, getitem_333);  clone_354 = getitem_333 = None
        add_409: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_332, 1e-05);  getitem_332 = None
        rsqrt_120: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_409);  add_409 = None
        mul_424: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_120);  sub_166 = rsqrt_120 = None
        mul_425: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_424, arg320_1);  mul_424 = arg320_1 = None
        add_410: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_425, arg321_1);  mul_425 = arg321_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_955: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_410, [25088, 24])
        permute_446: "f32[24, 48]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        mm_92: "f32[25088, 48]" = torch.ops.aten.mm.default(view_955, permute_446);  view_955 = permute_446 = None
        view_956: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_92, [1568, 16, 48]);  mm_92 = None
        view_957: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_956, [1568, 16, 2, 4, 6]);  view_956 = None
        permute_447: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_957, [2, 0, 3, 1, 4]);  view_957 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_46 = torch.ops.aten.unbind.int(permute_447);  permute_447 = None
        getitem_334: "f32[1568, 4, 16, 6]" = unbind_46[0]
        getitem_335: "f32[1568, 4, 16, 6]" = unbind_46[1];  unbind_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_186: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_334, [1568, 4, 16, 6]);  getitem_334 = None
        clone_355: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_961: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_355, [6272, 16, 6]);  clone_355 = None
        permute_450: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_335, [0, 1, 3, 2]);  getitem_335 = None
        expand_187: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_450, [1568, 4, 6, 16]);  permute_450 = None
        clone_356: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_962: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_356, [6272, 6, 16]);  clone_356 = None
        bmm_92: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_961, view_962);  view_961 = view_962 = None
        view_963: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_92, [1568, 4, 16, 16]);  bmm_92 = None
        
        # No stacktrace found for following nodes
        mul_tensor_2: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_963, 1);  view_963 = None
        amax_default_1: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_2, [-1], True)
        sub_tensor_1: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_1, 0.408248290463863);  sub_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_46: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        sum_47: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_46: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_188: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_46, [1568, 4, 16, 16]);  div_46 = None
        view_964: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_188, [6272, 16, 16]);  expand_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_958: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_410, [25088, 24]);  add_410 = None
        permute_448: "f32[24, 24]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        mm_93: "f32[25088, 24]" = torch.ops.aten.mm.default(view_958, permute_448);  view_958 = permute_448 = None
        view_959: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_93, [1568, 16, 24]);  mm_93 = None
        view_960: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_959, [1568, 16, 4, -1]);  view_959 = None
        permute_449: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_960, [0, 2, 1, 3]);  view_960 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_189: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_449, [1568, 4, 16, 6]);  permute_449 = None
        clone_357: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_965: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_357, [6272, 16, 6]);  clone_357 = None
        bmm_93: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_964, view_965);  view_964 = view_965 = None
        view_966: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_93, [1568, 4, 16, 6]);  bmm_93 = None
        permute_451: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_966, [0, 2, 1, 3]);  view_966 = None
        clone_358: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_967: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_358, [1568, 16, 24]);  clone_358 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_968: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_967, [25088, 24]);  view_967 = None
        permute_452: "f32[24, 24]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[25088, 24]" = torch.ops.aten.mm.default(view_968, permute_452);  view_968 = permute_452 = None
        add_tensor_8: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_8, arg325_1);  mm_default_8 = arg325_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_969: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_8, [1568, 16, 24]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:145 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        add_411: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_398, view_969);  add_398 = view_969 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        clone_359: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_411, memory_format = torch.contiguous_format)
        var_mean_121 = torch.ops.aten.var_mean.correction(clone_359, [2], correction = 0, keepdim = True)
        getitem_336: "f32[1568, 16, 1]" = var_mean_121[0]
        getitem_337: "f32[1568, 16, 1]" = var_mean_121[1];  var_mean_121 = None
        sub_168: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_359, getitem_337);  clone_359 = getitem_337 = None
        add_412: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_121: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_412);  add_412 = None
        mul_427: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_168, rsqrt_121);  sub_168 = rsqrt_121 = None
        mul_428: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_427, arg326_1);  mul_427 = arg326_1 = None
        add_413: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_428, arg327_1);  mul_428 = arg327_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_970: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_413, [25088, 24]);  add_413 = None
        permute_453: "f32[24, 96]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[25088, 96]" = torch.ops.aten.mm.default(view_970, permute_453);  view_970 = permute_453 = None
        add_tensor_7: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_7, arg329_1);  mm_default_7 = arg329_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_971: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_7, [1568, 16, 96]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_429: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_971, 0.5)
        mul_430: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_971, 0.7071067811865476);  view_971 = None
        erf_46: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_430);  mul_430 = None
        add_414: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_431: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_429, add_414);  mul_429 = add_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_972: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_431, [25088, 96]);  mul_431 = None
        permute_454: "f32[96, 24]" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[25088, 24]" = torch.ops.aten.mm.default(view_972, permute_454);  view_972 = permute_454 = None
        add_tensor_6: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_6, arg331_1);  mm_default_6 = arg331_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_973: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_6, [1568, 16, 24]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:146 in forward, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        add_415: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_411, view_973);  add_411 = view_973 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        clone_362: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_415, memory_format = torch.contiguous_format);  add_415 = None
        var_mean_122 = torch.ops.aten.var_mean.correction(clone_362, [2], correction = 0, keepdim = True)
        getitem_338: "f32[1568, 16, 1]" = var_mean_122[0]
        getitem_339: "f32[1568, 16, 1]" = var_mean_122[1];  var_mean_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        sub_165: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_404, getitem_331);  getitem_331 = None
        add_405: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_330, 1e-05);  getitem_330 = None
        rsqrt_119: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
        mul_419: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_119);  sub_165 = rsqrt_119 = None
        mul_420: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_419, arg314_1);  mul_419 = arg314_1 = None
        add_406: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_420, arg315_1);  mul_420 = arg315_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_951: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_406, [1576, 384]);  add_406 = None
        permute_444: "f32[384, 1536]" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_951, permute_444);  view_951 = permute_444 = None
        add_tensor_5: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_5, arg317_1);  mm_default_5 = arg317_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_952: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 197, 1536]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_421: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_952, 0.5)
        mul_422: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_952, 0.7071067811865476);  view_952 = None
        erf_45: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_422);  mul_422 = None
        add_407: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_423: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_421, add_407);  mul_421 = add_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_953: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_423, [1576, 1536]);  mul_423 = None
        permute_445: "f32[1536, 384]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[1576, 384]" = torch.ops.aten.mm.default(view_953, permute_445);  view_953 = permute_445 = None
        add_tensor_4: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_4, arg319_1);  mm_default_4 = arg319_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_954: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 197, 384]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_408: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_404, view_954);  add_404 = view_954 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        slice_99: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_408, 1, 0, 1)
        slice_101: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_408, 1, 1, 9223372036854775807);  add_408 = None
        sub_169: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_362, getitem_339);  clone_362 = getitem_339 = None
        add_416: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-05);  getitem_338 = None
        rsqrt_122: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_416);  add_416 = None
        mul_432: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_122);  sub_169 = rsqrt_122 = None
        mul_433: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_432, arg332_1);  mul_432 = arg332_1 = None
        add_417: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_433, arg333_1);  mul_433 = arg333_1 = None
        view_974: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_417, [8, 196, -1]);  add_417 = None
        view_975: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_974, [1568, 384]);  view_974 = None
        permute_455: "f32[384, 384]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[1568, 384]" = torch.ops.aten.mm.default(view_975, permute_455);  view_975 = permute_455 = None
        add_tensor_3: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_3, arg335_1);  mm_default_3 = arg335_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:150 in forward, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
        view_976: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 196, 384]);  add_tensor_3 = None
        add_418: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_101, view_976);  slice_101 = view_976 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:149 in forward, code: patch_embed = torch.cat(
        cat_25: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_99, add_418], 1);  slice_99 = add_418 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        var_mean_123 = torch.ops.aten.var_mean.correction(cat_25, [2], correction = 0, keepdim = True)
        getitem_340: "f32[8, 197, 1]" = var_mean_123[0]
        getitem_341: "f32[8, 197, 1]" = var_mean_123[1];  var_mean_123 = None
        sub_170: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_25, getitem_341);  getitem_341 = None
        add_419: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_340, 1e-05);  getitem_340 = None
        rsqrt_123: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_419);  add_419 = None
        mul_434: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_170, rsqrt_123);  sub_170 = rsqrt_123 = None
        mul_435: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_434, arg336_1);  mul_434 = arg336_1 = None
        add_420: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_435, arg337_1);  mul_435 = arg337_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:66 in forward, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_977: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_420, [1576, 384])
        permute_456: "f32[384, 768]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        mm_94: "f32[1576, 768]" = torch.ops.aten.mm.default(view_977, permute_456);  view_977 = permute_456 = None
        view_978: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_94, [8, 197, 768]);  mm_94 = None
        view_979: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_978, [8, 197, 2, 6, 64]);  view_978 = None
        permute_457: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_979, [2, 0, 3, 1, 4]);  view_979 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:67 in forward, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        unbind_47 = torch.ops.aten.unbind.int(permute_457);  permute_457 = None
        getitem_342: "f32[8, 6, 197, 64]" = unbind_47[0]
        getitem_343: "f32[8, 6, 197, 64]" = unbind_47[1];  unbind_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:70 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        expand_190: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_342, [8, 6, 197, 64]);  getitem_342 = None
        clone_363: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_190, memory_format = torch.contiguous_format);  expand_190 = None
        view_983: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_363, [48, 197, 64]);  clone_363 = None
        permute_460: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_343, [0, 1, 3, 2]);  getitem_343 = None
        expand_191: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_460, [8, 6, 64, 197]);  permute_460 = None
        clone_364: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_191, memory_format = torch.contiguous_format);  expand_191 = None
        view_984: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_364, [48, 64, 197]);  clone_364 = None
        bmm_94: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_983, view_984);  view_983 = view_984 = None
        view_985: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_94, [8, 6, 197, 197]);  bmm_94 = None
        
        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_985, 1);  view_985 = None
        amax_default: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:71 in forward, code: attn = attn.softmax(dim=-1)
        exp_47: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_48: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_47: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_47, sum_48);  exp_47 = sum_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_192: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_47, [8, 6, 197, 197]);  div_47 = None
        view_986: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_192, [48, 197, 197]);  expand_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:68 in forward, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        view_980: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_420, [1576, 384]);  add_420 = None
        permute_458: "f32[384, 384]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        mm_95: "f32[1576, 384]" = torch.ops.aten.mm.default(view_980, permute_458);  view_980 = permute_458 = None
        view_981: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_95, [8, 197, 384]);  mm_95 = None
        view_982: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_981, [8, 197, 6, -1]);  view_981 = None
        permute_459: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_982, [0, 2, 1, 3]);  view_982 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:74 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        expand_193: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_459, [8, 6, 197, 64]);  permute_459 = None
        clone_365: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_987: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_365, [48, 197, 64]);  clone_365 = None
        bmm_95: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_986, view_987);  view_986 = view_987 = None
        view_988: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_95, [8, 6, 197, 64]);  bmm_95 = None
        permute_461: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_988, [0, 2, 1, 3]);  view_988 = None
        clone_366: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
        view_989: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_366, [8, 197, 384]);  clone_366 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_990: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_989, [1576, 384]);  view_989 = None
        permute_462: "f32[384, 384]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[1576, 384]" = torch.ops.aten.mm.default(view_990, permute_462);  view_990 = permute_462 = None
        add_tensor_2: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_2, arg341_1);  mm_default_2 = arg341_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:75 in forward, code: x = self.proj(x)
        view_991: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 197, 384]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:152 in forward, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        add_421: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_25, view_991);  cat_25 = view_991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        var_mean_124 = torch.ops.aten.var_mean.correction(add_421, [2], correction = 0, keepdim = True)
        getitem_344: "f32[8, 197, 1]" = var_mean_124[0]
        getitem_345: "f32[8, 197, 1]" = var_mean_124[1];  var_mean_124 = None
        sub_172: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_421, getitem_345);  getitem_345 = None
        add_422: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_344, 1e-05);  getitem_344 = None
        rsqrt_124: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
        mul_437: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_172, rsqrt_124);  sub_172 = rsqrt_124 = None
        mul_438: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_437, arg342_1);  mul_437 = arg342_1 = None
        add_423: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_438, arg343_1);  mul_438 = arg343_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_992: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_423, [1576, 384]);  add_423 = None
        permute_463: "f32[384, 1536]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_992, permute_463);  view_992 = permute_463 = None
        add_tensor_1: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_1, arg345_1);  mm_default_1 = arg345_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_993: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 197, 1536]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_439: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_993, 0.5)
        mul_440: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_993, 0.7071067811865476);  view_993 = None
        erf_47: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_440);  mul_440 = None
        add_424: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_441: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_439, add_424);  mul_439 = add_424 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_994: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_441, [1576, 1536]);  mul_441 = None
        permute_464: "f32[1536, 384]" = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[1576, 384]" = torch.ops.aten.mm.default(view_994, permute_464);  view_994 = permute_464 = None
        add_tensor: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default, arg347_1);  mm_default = arg347_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_995: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor, [8, 197, 384]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:153 in forward, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        add_425: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_421, view_995);  add_421 = view_995 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:325 in forward_features, code: patch_embed = self.norm(patch_embed)
        var_mean_125 = torch.ops.aten.var_mean.correction(add_425, [2], correction = 0, keepdim = True)
        getitem_346: "f32[8, 197, 1]" = var_mean_125[0]
        getitem_347: "f32[8, 197, 1]" = var_mean_125[1];  var_mean_125 = None
        sub_173: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_425, getitem_347);  add_425 = getitem_347 = None
        add_426: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_346, 1e-05);  getitem_346 = None
        rsqrt_125: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_426);  add_426 = None
        mul_442: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_173, rsqrt_125);  sub_173 = rsqrt_125 = None
        mul_443: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_442, arg348_1);  mul_442 = arg348_1 = None
        add_427: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_443, arg349_1);  mul_443 = arg349_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:330 in forward_head, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        select_1: "f32[8, 384]" = torch.ops.aten.select.int(add_427, 1, 0);  add_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:331 in forward_head, code: x = self.head_drop(x)
        clone_369: "f32[8, 384]" = torch.ops.aten.clone.default(select_1);  select_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/tnt.py:332 in forward_head, code: return x if pre_logits else self.head(x)
        permute_465: "f32[384, 1000]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
        addmm_171: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg351_1, clone_369, permute_465);  arg351_1 = clone_369 = permute_465 = None
        return (addmm_171,)
        