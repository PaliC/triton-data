class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 224, 224]", arg1_1: "f32[128, 3, 4, 4]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[128]", arg7_1: "f32[384, 128]", arg8_1: "f32[384]", arg9_1: "f32[169, 4]", arg10_1: "i64[49, 49]", arg11_1: "f32[128, 128]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[512, 128]", arg16_1: "f32[512]", arg17_1: "f32[128, 512]", arg18_1: "f32[128]", arg19_1: "f32[128]", arg20_1: "f32[128]", arg21_1: "f32[64, 49, 49]", arg22_1: "f32[384, 128]", arg23_1: "f32[384]", arg24_1: "f32[169, 4]", arg25_1: "i64[49, 49]", arg26_1: "f32[128, 128]", arg27_1: "f32[128]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[512, 128]", arg31_1: "f32[512]", arg32_1: "f32[128, 512]", arg33_1: "f32[128]", arg34_1: "f32[512]", arg35_1: "f32[512]", arg36_1: "f32[256, 512]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[768, 256]", arg40_1: "f32[768]", arg41_1: "f32[169, 8]", arg42_1: "i64[49, 49]", arg43_1: "f32[256, 256]", arg44_1: "f32[256]", arg45_1: "f32[256]", arg46_1: "f32[256]", arg47_1: "f32[1024, 256]", arg48_1: "f32[1024]", arg49_1: "f32[256, 1024]", arg50_1: "f32[256]", arg51_1: "f32[256]", arg52_1: "f32[256]", arg53_1: "f32[16, 49, 49]", arg54_1: "f32[768, 256]", arg55_1: "f32[768]", arg56_1: "f32[169, 8]", arg57_1: "i64[49, 49]", arg58_1: "f32[256, 256]", arg59_1: "f32[256]", arg60_1: "f32[256]", arg61_1: "f32[256]", arg62_1: "f32[1024, 256]", arg63_1: "f32[1024]", arg64_1: "f32[256, 1024]", arg65_1: "f32[256]", arg66_1: "f32[1024]", arg67_1: "f32[1024]", arg68_1: "f32[512, 1024]", arg69_1: "f32[512]", arg70_1: "f32[512]", arg71_1: "f32[1536, 512]", arg72_1: "f32[1536]", arg73_1: "f32[169, 16]", arg74_1: "i64[49, 49]", arg75_1: "f32[512, 512]", arg76_1: "f32[512]", arg77_1: "f32[512]", arg78_1: "f32[512]", arg79_1: "f32[2048, 512]", arg80_1: "f32[2048]", arg81_1: "f32[512, 2048]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[4, 49, 49]", arg86_1: "f32[1536, 512]", arg87_1: "f32[1536]", arg88_1: "f32[169, 16]", arg89_1: "i64[49, 49]", arg90_1: "f32[512, 512]", arg91_1: "f32[512]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[2048, 512]", arg95_1: "f32[2048]", arg96_1: "f32[512, 2048]", arg97_1: "f32[512]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[1536, 512]", arg101_1: "f32[1536]", arg102_1: "f32[169, 16]", arg103_1: "i64[49, 49]", arg104_1: "f32[512, 512]", arg105_1: "f32[512]", arg106_1: "f32[512]", arg107_1: "f32[512]", arg108_1: "f32[2048, 512]", arg109_1: "f32[2048]", arg110_1: "f32[512, 2048]", arg111_1: "f32[512]", arg112_1: "f32[512]", arg113_1: "f32[512]", arg114_1: "f32[4, 49, 49]", arg115_1: "f32[1536, 512]", arg116_1: "f32[1536]", arg117_1: "f32[169, 16]", arg118_1: "i64[49, 49]", arg119_1: "f32[512, 512]", arg120_1: "f32[512]", arg121_1: "f32[512]", arg122_1: "f32[512]", arg123_1: "f32[2048, 512]", arg124_1: "f32[2048]", arg125_1: "f32[512, 2048]", arg126_1: "f32[512]", arg127_1: "f32[512]", arg128_1: "f32[512]", arg129_1: "f32[1536, 512]", arg130_1: "f32[1536]", arg131_1: "f32[169, 16]", arg132_1: "i64[49, 49]", arg133_1: "f32[512, 512]", arg134_1: "f32[512]", arg135_1: "f32[512]", arg136_1: "f32[512]", arg137_1: "f32[2048, 512]", arg138_1: "f32[2048]", arg139_1: "f32[512, 2048]", arg140_1: "f32[512]", arg141_1: "f32[512]", arg142_1: "f32[512]", arg143_1: "f32[4, 49, 49]", arg144_1: "f32[1536, 512]", arg145_1: "f32[1536]", arg146_1: "f32[169, 16]", arg147_1: "i64[49, 49]", arg148_1: "f32[512, 512]", arg149_1: "f32[512]", arg150_1: "f32[512]", arg151_1: "f32[512]", arg152_1: "f32[2048, 512]", arg153_1: "f32[2048]", arg154_1: "f32[512, 2048]", arg155_1: "f32[512]", arg156_1: "f32[512]", arg157_1: "f32[512]", arg158_1: "f32[1536, 512]", arg159_1: "f32[1536]", arg160_1: "f32[169, 16]", arg161_1: "i64[49, 49]", arg162_1: "f32[512, 512]", arg163_1: "f32[512]", arg164_1: "f32[512]", arg165_1: "f32[512]", arg166_1: "f32[2048, 512]", arg167_1: "f32[2048]", arg168_1: "f32[512, 2048]", arg169_1: "f32[512]", arg170_1: "f32[512]", arg171_1: "f32[512]", arg172_1: "f32[4, 49, 49]", arg173_1: "f32[1536, 512]", arg174_1: "f32[1536]", arg175_1: "f32[169, 16]", arg176_1: "i64[49, 49]", arg177_1: "f32[512, 512]", arg178_1: "f32[512]", arg179_1: "f32[512]", arg180_1: "f32[512]", arg181_1: "f32[2048, 512]", arg182_1: "f32[2048]", arg183_1: "f32[512, 2048]", arg184_1: "f32[512]", arg185_1: "f32[512]", arg186_1: "f32[512]", arg187_1: "f32[1536, 512]", arg188_1: "f32[1536]", arg189_1: "f32[169, 16]", arg190_1: "i64[49, 49]", arg191_1: "f32[512, 512]", arg192_1: "f32[512]", arg193_1: "f32[512]", arg194_1: "f32[512]", arg195_1: "f32[2048, 512]", arg196_1: "f32[2048]", arg197_1: "f32[512, 2048]", arg198_1: "f32[512]", arg199_1: "f32[512]", arg200_1: "f32[512]", arg201_1: "f32[4, 49, 49]", arg202_1: "f32[1536, 512]", arg203_1: "f32[1536]", arg204_1: "f32[169, 16]", arg205_1: "i64[49, 49]", arg206_1: "f32[512, 512]", arg207_1: "f32[512]", arg208_1: "f32[512]", arg209_1: "f32[512]", arg210_1: "f32[2048, 512]", arg211_1: "f32[2048]", arg212_1: "f32[512, 2048]", arg213_1: "f32[512]", arg214_1: "f32[512]", arg215_1: "f32[512]", arg216_1: "f32[1536, 512]", arg217_1: "f32[1536]", arg218_1: "f32[169, 16]", arg219_1: "i64[49, 49]", arg220_1: "f32[512, 512]", arg221_1: "f32[512]", arg222_1: "f32[512]", arg223_1: "f32[512]", arg224_1: "f32[2048, 512]", arg225_1: "f32[2048]", arg226_1: "f32[512, 2048]", arg227_1: "f32[512]", arg228_1: "f32[512]", arg229_1: "f32[512]", arg230_1: "f32[4, 49, 49]", arg231_1: "f32[1536, 512]", arg232_1: "f32[1536]", arg233_1: "f32[169, 16]", arg234_1: "i64[49, 49]", arg235_1: "f32[512, 512]", arg236_1: "f32[512]", arg237_1: "f32[512]", arg238_1: "f32[512]", arg239_1: "f32[2048, 512]", arg240_1: "f32[2048]", arg241_1: "f32[512, 2048]", arg242_1: "f32[512]", arg243_1: "f32[512]", arg244_1: "f32[512]", arg245_1: "f32[1536, 512]", arg246_1: "f32[1536]", arg247_1: "f32[169, 16]", arg248_1: "i64[49, 49]", arg249_1: "f32[512, 512]", arg250_1: "f32[512]", arg251_1: "f32[512]", arg252_1: "f32[512]", arg253_1: "f32[2048, 512]", arg254_1: "f32[2048]", arg255_1: "f32[512, 2048]", arg256_1: "f32[512]", arg257_1: "f32[512]", arg258_1: "f32[512]", arg259_1: "f32[4, 49, 49]", arg260_1: "f32[1536, 512]", arg261_1: "f32[1536]", arg262_1: "f32[169, 16]", arg263_1: "i64[49, 49]", arg264_1: "f32[512, 512]", arg265_1: "f32[512]", arg266_1: "f32[512]", arg267_1: "f32[512]", arg268_1: "f32[2048, 512]", arg269_1: "f32[2048]", arg270_1: "f32[512, 2048]", arg271_1: "f32[512]", arg272_1: "f32[512]", arg273_1: "f32[512]", arg274_1: "f32[1536, 512]", arg275_1: "f32[1536]", arg276_1: "f32[169, 16]", arg277_1: "i64[49, 49]", arg278_1: "f32[512, 512]", arg279_1: "f32[512]", arg280_1: "f32[512]", arg281_1: "f32[512]", arg282_1: "f32[2048, 512]", arg283_1: "f32[2048]", arg284_1: "f32[512, 2048]", arg285_1: "f32[512]", arg286_1: "f32[512]", arg287_1: "f32[512]", arg288_1: "f32[4, 49, 49]", arg289_1: "f32[1536, 512]", arg290_1: "f32[1536]", arg291_1: "f32[169, 16]", arg292_1: "i64[49, 49]", arg293_1: "f32[512, 512]", arg294_1: "f32[512]", arg295_1: "f32[512]", arg296_1: "f32[512]", arg297_1: "f32[2048, 512]", arg298_1: "f32[2048]", arg299_1: "f32[512, 2048]", arg300_1: "f32[512]", arg301_1: "f32[512]", arg302_1: "f32[512]", arg303_1: "f32[1536, 512]", arg304_1: "f32[1536]", arg305_1: "f32[169, 16]", arg306_1: "i64[49, 49]", arg307_1: "f32[512, 512]", arg308_1: "f32[512]", arg309_1: "f32[512]", arg310_1: "f32[512]", arg311_1: "f32[2048, 512]", arg312_1: "f32[2048]", arg313_1: "f32[512, 2048]", arg314_1: "f32[512]", arg315_1: "f32[512]", arg316_1: "f32[512]", arg317_1: "f32[4, 49, 49]", arg318_1: "f32[1536, 512]", arg319_1: "f32[1536]", arg320_1: "f32[169, 16]", arg321_1: "i64[49, 49]", arg322_1: "f32[512, 512]", arg323_1: "f32[512]", arg324_1: "f32[512]", arg325_1: "f32[512]", arg326_1: "f32[2048, 512]", arg327_1: "f32[2048]", arg328_1: "f32[512, 2048]", arg329_1: "f32[512]", arg330_1: "f32[2048]", arg331_1: "f32[2048]", arg332_1: "f32[1024, 2048]", arg333_1: "f32[1024]", arg334_1: "f32[1024]", arg335_1: "f32[3072, 1024]", arg336_1: "f32[3072]", arg337_1: "f32[169, 32]", arg338_1: "i64[49, 49]", arg339_1: "f32[1024, 1024]", arg340_1: "f32[1024]", arg341_1: "f32[1024]", arg342_1: "f32[1024]", arg343_1: "f32[4096, 1024]", arg344_1: "f32[4096]", arg345_1: "f32[1024, 4096]", arg346_1: "f32[1024]", arg347_1: "f32[1024]", arg348_1: "f32[1024]", arg349_1: "f32[3072, 1024]", arg350_1: "f32[3072]", arg351_1: "f32[169, 32]", arg352_1: "i64[49, 49]", arg353_1: "f32[1024, 1024]", arg354_1: "f32[1024]", arg355_1: "f32[1024]", arg356_1: "f32[1024]", arg357_1: "f32[4096, 1024]", arg358_1: "f32[4096]", arg359_1: "f32[1024, 4096]", arg360_1: "f32[1024]", arg361_1: "f32[1024]", arg362_1: "f32[1024]", arg363_1: "f32[1000, 1024]", arg364_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:131 in forward, code: x = self.proj(x)
        convolution_1: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/format.py:43 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute_248: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.norm(x)
        clone_265: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        var_mean_53 = torch.ops.aten.var_mean.correction(clone_265, [3], correction = 0, keepdim = True)
        getitem_178: "f32[8, 56, 56, 1]" = var_mean_53[0]
        getitem_179: "f32[8, 56, 56, 1]" = var_mean_53[1];  var_mean_53 = None
        add_257: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_53: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        sub_77: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_265, getitem_179);  clone_265 = getitem_179 = None
        mul_202: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_53);  sub_77 = rsqrt_53 = None
        mul_203: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_202, arg3_1);  mul_202 = arg3_1 = None
        add_258: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_203, arg4_1);  mul_203 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_54 = torch.ops.aten.var_mean.correction(add_258, [3], correction = 0, keepdim = True)
        getitem_180: "f32[8, 56, 56, 1]" = var_mean_54[0]
        getitem_181: "f32[8, 56, 56, 1]" = var_mean_54[1];  var_mean_54 = None
        add_259: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_54: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
        sub_78: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add_258, getitem_181);  getitem_181 = None
        mul_204: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_54);  sub_78 = rsqrt_54 = None
        mul_205: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_204, arg5_1);  mul_204 = arg5_1 = None
        add_260: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_205, arg6_1);  mul_205 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_658: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(add_260, [8, 8, 7, 8, 7, 128]);  add_260 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_249: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_658, [0, 1, 3, 2, 4, 5]);  view_658 = None
        clone_266: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_659: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(clone_266, [-1, 7, 7, 128]);  clone_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_660: "f32[512, 49, 128]" = torch.ops.aten.view.default(view_659, [-1, 49, 128]);  view_659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_661: "f32[25088, 128]" = torch.ops.aten.view.default(view_660, [25088, 128]);  view_660 = None
        permute_250: "f32[128, 384]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_97: "f32[25088, 384]" = torch.ops.aten.addmm.default(arg8_1, view_661, permute_250);  arg8_1 = view_661 = permute_250 = None
        view_662: "f32[512, 49, 384]" = torch.ops.aten.view.default(addmm_97, [512, 49, 384]);  addmm_97 = None
        view_663: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.view.default(view_662, [512, 49, 3, 4, -1]);  view_662 = None
        permute_251: "f32[3, 512, 4, 49, 32]" = torch.ops.aten.permute.default(view_663, [2, 0, 3, 1, 4]);  view_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_24 = torch.ops.aten.unbind.int(permute_251);  permute_251 = None
        getitem_182: "f32[512, 4, 49, 32]" = unbind_24[0]
        getitem_183: "f32[512, 4, 49, 32]" = unbind_24[1]
        getitem_184: "f32[512, 4, 49, 32]" = unbind_24[2];  unbind_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_206: "f32[512, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_182, 0.1767766952966369);  getitem_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_252: "f32[512, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_183, [0, 1, 3, 2]);  getitem_183 = None
        expand_96: "f32[512, 4, 49, 32]" = torch.ops.aten.expand.default(mul_206, [512, 4, 49, 32]);  mul_206 = None
        clone_267: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_664: "f32[2048, 49, 32]" = torch.ops.aten.view.default(clone_267, [2048, 49, 32]);  clone_267 = None
        expand_97: "f32[512, 4, 32, 49]" = torch.ops.aten.expand.default(permute_252, [512, 4, 32, 49]);  permute_252 = None
        clone_268: "f32[512, 4, 32, 49]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_665: "f32[2048, 32, 49]" = torch.ops.aten.view.default(clone_268, [2048, 32, 49]);  clone_268 = None
        bmm_48: "f32[2048, 49, 49]" = torch.ops.aten.bmm.default(view_664, view_665);  view_664 = view_665 = None
        view_666: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(bmm_48, [512, 4, 49, 49]);  bmm_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_667: "i64[2401]" = torch.ops.aten.view.default(arg10_1, [-1]);  arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_68: "f32[2401, 4]" = torch.ops.aten.index.Tensor(arg9_1, [view_667]);  arg9_1 = view_667 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_668: "f32[49, 49, 4]" = torch.ops.aten.view.default(index_68, [49, 49, -1]);  index_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_253: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_668, [2, 0, 1]);  view_668 = None
        clone_269: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_46: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_269, 0);  clone_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_261: "f32[512, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_666, unsqueeze_46);  view_666 = unsqueeze_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_24: "f32[512, 4, 49, 1]" = torch.ops.aten.amax.default(add_261, [-1], True)
        sub_79: "f32[512, 4, 49, 49]" = torch.ops.aten.sub.Tensor(add_261, amax_24);  add_261 = amax_24 = None
        exp_24: "f32[512, 4, 49, 49]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
        sum_25: "f32[512, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24: "f32[512, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_98: "f32[512, 4, 49, 49]" = torch.ops.aten.expand.default(div_24, [512, 4, 49, 49]);  div_24 = None
        view_669: "f32[2048, 49, 49]" = torch.ops.aten.view.default(expand_98, [2048, 49, 49]);  expand_98 = None
        expand_99: "f32[512, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_184, [512, 4, 49, 32]);  getitem_184 = None
        clone_271: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_670: "f32[2048, 49, 32]" = torch.ops.aten.view.default(clone_271, [2048, 49, 32]);  clone_271 = None
        bmm_49: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(view_669, view_670);  view_669 = view_670 = None
        view_671: "f32[512, 4, 49, 32]" = torch.ops.aten.view.default(bmm_49, [512, 4, 49, 32]);  bmm_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_254: "f32[512, 49, 4, 32]" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None
        clone_272: "f32[512, 49, 4, 32]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_672: "f32[512, 49, 128]" = torch.ops.aten.view.default(clone_272, [512, 49, 128]);  clone_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_673: "f32[25088, 128]" = torch.ops.aten.view.default(view_672, [25088, 128]);  view_672 = None
        permute_255: "f32[128, 128]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_98: "f32[25088, 128]" = torch.ops.aten.addmm.default(arg12_1, view_673, permute_255);  arg12_1 = view_673 = permute_255 = None
        view_674: "f32[512, 49, 128]" = torch.ops.aten.view.default(addmm_98, [512, 49, 128]);  addmm_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_675: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(view_674, [-1, 7, 7, 128]);  view_674 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_676: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_675, [-1, 8, 8, 7, 7, 128]);  view_675 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_256: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_676, [0, 1, 3, 2, 4, 5]);  view_676 = None
        clone_274: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_677: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(clone_274, [-1, 56, 56, 128]);  clone_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_262: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(add_258, view_677);  add_258 = view_677 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_678: "f32[8, 3136, 128]" = torch.ops.aten.view.default(add_262, [8, -1, 128]);  add_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_55 = torch.ops.aten.var_mean.correction(view_678, [2], correction = 0, keepdim = True)
        getitem_185: "f32[8, 3136, 1]" = var_mean_55[0]
        getitem_186: "f32[8, 3136, 1]" = var_mean_55[1];  var_mean_55 = None
        add_263: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_185, 1e-05);  getitem_185 = None
        rsqrt_55: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        sub_80: "f32[8, 3136, 128]" = torch.ops.aten.sub.Tensor(view_678, getitem_186);  getitem_186 = None
        mul_207: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_55);  sub_80 = rsqrt_55 = None
        mul_208: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_207, arg13_1);  mul_207 = arg13_1 = None
        add_264: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(mul_208, arg14_1);  mul_208 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_679: "f32[25088, 128]" = torch.ops.aten.view.default(add_264, [25088, 128]);  add_264 = None
        permute_257: "f32[128, 512]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_99: "f32[25088, 512]" = torch.ops.aten.addmm.default(arg16_1, view_679, permute_257);  arg16_1 = view_679 = permute_257 = None
        view_680: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_99, [8, 3136, 512]);  addmm_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_209: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_680, 0.5)
        mul_210: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_680, 0.7071067811865476);  view_680 = None
        erf_24: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_210);  mul_210 = None
        add_265: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_211: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_209, add_265);  mul_209 = add_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_681: "f32[25088, 512]" = torch.ops.aten.view.default(mul_211, [25088, 512]);  mul_211 = None
        permute_258: "f32[512, 128]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_100: "f32[25088, 128]" = torch.ops.aten.addmm.default(arg18_1, view_681, permute_258);  arg18_1 = view_681 = permute_258 = None
        view_682: "f32[8, 3136, 128]" = torch.ops.aten.view.default(addmm_100, [8, 3136, 128]);  addmm_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_266: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(view_678, view_682);  view_678 = view_682 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_683: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(add_266, [8, 56, 56, 128]);  add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_56 = torch.ops.aten.var_mean.correction(view_683, [3], correction = 0, keepdim = True)
        getitem_187: "f32[8, 56, 56, 1]" = var_mean_56[0]
        getitem_188: "f32[8, 56, 56, 1]" = var_mean_56[1];  var_mean_56 = None
        add_267: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_187, 1e-05);  getitem_187 = None
        rsqrt_56: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
        sub_81: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(view_683, getitem_188);  getitem_188 = None
        mul_212: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_56);  sub_81 = rsqrt_56 = None
        mul_213: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_212, arg19_1);  mul_212 = arg19_1 = None
        add_268: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_213, arg20_1);  mul_213 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_44: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_269: "i64[56]" = torch.ops.aten.add.Tensor(iota_44, 3);  iota_44 = None
        fmod_44: "i64[56]" = torch.ops.aten.fmod.Scalar(add_269, 56);  add_269 = None
        index_69: "f32[8, 56, 56, 128]" = torch.ops.aten.index.Tensor(add_268, [None, fmod_44]);  add_268 = fmod_44 = None
        iota_45: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_270: "i64[56]" = torch.ops.aten.add.Tensor(iota_45, 3);  iota_45 = None
        fmod_45: "i64[56]" = torch.ops.aten.fmod.Scalar(add_270, 56);  add_270 = None
        index_70: "f32[8, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_69, [None, None, fmod_45]);  index_69 = fmod_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_684: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(index_70, [8, 8, 7, 8, 7, 128]);  index_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_259: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_684, [0, 1, 3, 2, 4, 5]);  view_684 = None
        clone_277: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_685: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(clone_277, [-1, 7, 7, 128]);  clone_277 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_686: "f32[512, 49, 128]" = torch.ops.aten.view.default(view_685, [-1, 49, 128]);  view_685 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_687: "f32[25088, 128]" = torch.ops.aten.view.default(view_686, [25088, 128]);  view_686 = None
        permute_260: "f32[128, 384]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_101: "f32[25088, 384]" = torch.ops.aten.addmm.default(arg23_1, view_687, permute_260);  arg23_1 = view_687 = permute_260 = None
        view_688: "f32[512, 49, 384]" = torch.ops.aten.view.default(addmm_101, [512, 49, 384]);  addmm_101 = None
        view_689: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.view.default(view_688, [512, 49, 3, 4, -1]);  view_688 = None
        permute_261: "f32[3, 512, 4, 49, 32]" = torch.ops.aten.permute.default(view_689, [2, 0, 3, 1, 4]);  view_689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_25 = torch.ops.aten.unbind.int(permute_261);  permute_261 = None
        getitem_189: "f32[512, 4, 49, 32]" = unbind_25[0]
        getitem_190: "f32[512, 4, 49, 32]" = unbind_25[1]
        getitem_191: "f32[512, 4, 49, 32]" = unbind_25[2];  unbind_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_214: "f32[512, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_189, 0.1767766952966369);  getitem_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_262: "f32[512, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_190, [0, 1, 3, 2]);  getitem_190 = None
        expand_100: "f32[512, 4, 49, 32]" = torch.ops.aten.expand.default(mul_214, [512, 4, 49, 32]);  mul_214 = None
        clone_278: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_690: "f32[2048, 49, 32]" = torch.ops.aten.view.default(clone_278, [2048, 49, 32]);  clone_278 = None
        expand_101: "f32[512, 4, 32, 49]" = torch.ops.aten.expand.default(permute_262, [512, 4, 32, 49]);  permute_262 = None
        clone_279: "f32[512, 4, 32, 49]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_691: "f32[2048, 32, 49]" = torch.ops.aten.view.default(clone_279, [2048, 32, 49]);  clone_279 = None
        bmm_50: "f32[2048, 49, 49]" = torch.ops.aten.bmm.default(view_690, view_691);  view_690 = view_691 = None
        view_692: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(bmm_50, [512, 4, 49, 49]);  bmm_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_693: "i64[2401]" = torch.ops.aten.view.default(arg25_1, [-1]);  arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_71: "f32[2401, 4]" = torch.ops.aten.index.Tensor(arg24_1, [view_693]);  arg24_1 = view_693 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_694: "f32[49, 49, 4]" = torch.ops.aten.view.default(index_71, [49, 49, -1]);  index_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_263: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_694, [2, 0, 1]);  view_694 = None
        clone_280: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_47: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_280, 0);  clone_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_271: "f32[512, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_692, unsqueeze_47);  view_692 = unsqueeze_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_695: "f32[8, 64, 4, 49, 49]" = torch.ops.aten.view.default(add_271, [-1, 64, 4, 49, 49]);  add_271 = None
        unsqueeze_48: "f32[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg21_1, 1);  arg21_1 = None
        unsqueeze_49: "f32[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 0);  unsqueeze_48 = None
        add_272: "f32[8, 64, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_695, unsqueeze_49);  view_695 = unsqueeze_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_696: "f32[512, 4, 49, 49]" = torch.ops.aten.view.default(add_272, [-1, 4, 49, 49]);  add_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_25: "f32[512, 4, 49, 1]" = torch.ops.aten.amax.default(view_696, [-1], True)
        sub_82: "f32[512, 4, 49, 49]" = torch.ops.aten.sub.Tensor(view_696, amax_25);  view_696 = amax_25 = None
        exp_25: "f32[512, 4, 49, 49]" = torch.ops.aten.exp.default(sub_82);  sub_82 = None
        sum_26: "f32[512, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_25: "f32[512, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_102: "f32[512, 4, 49, 49]" = torch.ops.aten.expand.default(div_25, [512, 4, 49, 49]);  div_25 = None
        view_697: "f32[2048, 49, 49]" = torch.ops.aten.view.default(expand_102, [2048, 49, 49]);  expand_102 = None
        expand_103: "f32[512, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_191, [512, 4, 49, 32]);  getitem_191 = None
        clone_282: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_698: "f32[2048, 49, 32]" = torch.ops.aten.view.default(clone_282, [2048, 49, 32]);  clone_282 = None
        bmm_51: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(view_697, view_698);  view_697 = view_698 = None
        view_699: "f32[512, 4, 49, 32]" = torch.ops.aten.view.default(bmm_51, [512, 4, 49, 32]);  bmm_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_264: "f32[512, 49, 4, 32]" = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None
        clone_283: "f32[512, 49, 4, 32]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
        view_700: "f32[512, 49, 128]" = torch.ops.aten.view.default(clone_283, [512, 49, 128]);  clone_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_701: "f32[25088, 128]" = torch.ops.aten.view.default(view_700, [25088, 128]);  view_700 = None
        permute_265: "f32[128, 128]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_102: "f32[25088, 128]" = torch.ops.aten.addmm.default(arg27_1, view_701, permute_265);  arg27_1 = view_701 = permute_265 = None
        view_702: "f32[512, 49, 128]" = torch.ops.aten.view.default(addmm_102, [512, 49, 128]);  addmm_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_703: "f32[512, 7, 7, 128]" = torch.ops.aten.view.default(view_702, [-1, 7, 7, 128]);  view_702 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_704: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_703, [-1, 8, 8, 7, 7, 128]);  view_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_266: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_704, [0, 1, 3, 2, 4, 5]);  view_704 = None
        clone_285: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
        view_705: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(clone_285, [-1, 56, 56, 128]);  clone_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_46: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_273: "i64[56]" = torch.ops.aten.add.Tensor(iota_46, 53);  iota_46 = None
        fmod_46: "i64[56]" = torch.ops.aten.fmod.Scalar(add_273, 56);  add_273 = None
        index_72: "f32[8, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_705, [None, fmod_46]);  view_705 = fmod_46 = None
        iota_47: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_274: "i64[56]" = torch.ops.aten.add.Tensor(iota_47, 53);  iota_47 = None
        fmod_47: "i64[56]" = torch.ops.aten.fmod.Scalar(add_274, 56);  add_274 = None
        index_73: "f32[8, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_72, [None, None, fmod_47]);  index_72 = fmod_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_275: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_683, index_73);  view_683 = index_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_706: "f32[8, 3136, 128]" = torch.ops.aten.view.default(add_275, [8, -1, 128]);  add_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_57 = torch.ops.aten.var_mean.correction(view_706, [2], correction = 0, keepdim = True)
        getitem_192: "f32[8, 3136, 1]" = var_mean_57[0]
        getitem_193: "f32[8, 3136, 1]" = var_mean_57[1];  var_mean_57 = None
        add_276: "f32[8, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_57: "f32[8, 3136, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_83: "f32[8, 3136, 128]" = torch.ops.aten.sub.Tensor(view_706, getitem_193);  getitem_193 = None
        mul_215: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_57);  sub_83 = rsqrt_57 = None
        mul_216: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_215, arg28_1);  mul_215 = arg28_1 = None
        add_277: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(mul_216, arg29_1);  mul_216 = arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_707: "f32[25088, 128]" = torch.ops.aten.view.default(add_277, [25088, 128]);  add_277 = None
        permute_267: "f32[128, 512]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_103: "f32[25088, 512]" = torch.ops.aten.addmm.default(arg31_1, view_707, permute_267);  arg31_1 = view_707 = permute_267 = None
        view_708: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_103, [8, 3136, 512]);  addmm_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_217: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_708, 0.5)
        mul_218: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_708, 0.7071067811865476);  view_708 = None
        erf_25: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
        add_278: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_219: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_217, add_278);  mul_217 = add_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_709: "f32[25088, 512]" = torch.ops.aten.view.default(mul_219, [25088, 512]);  mul_219 = None
        permute_268: "f32[512, 128]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_104: "f32[25088, 128]" = torch.ops.aten.addmm.default(arg33_1, view_709, permute_268);  arg33_1 = view_709 = permute_268 = None
        view_710: "f32[8, 3136, 128]" = torch.ops.aten.view.default(addmm_104, [8, 3136, 128]);  addmm_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_279: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(view_706, view_710);  view_706 = view_710 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_711: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(add_279, [8, 56, 56, 128]);  add_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:442 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_712: "f32[8, 28, 2, 28, 2, 128]" = torch.ops.aten.view.default(view_711, [8, 28, 2, 28, 2, 128]);  view_711 = None
        permute_269: "f32[8, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(view_712, [0, 1, 3, 4, 2, 5]);  view_712 = None
        clone_288: "f32[8, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_713: "f32[8, 28, 28, 512]" = torch.ops.aten.view.default(clone_288, [8, 28, 28, 512]);  clone_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in forward, code: x = self.norm(x)
        var_mean_58 = torch.ops.aten.var_mean.correction(view_713, [3], correction = 0, keepdim = True)
        getitem_194: "f32[8, 28, 28, 1]" = var_mean_58[0]
        getitem_195: "f32[8, 28, 28, 1]" = var_mean_58[1];  var_mean_58 = None
        add_280: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_58: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        sub_84: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(view_713, getitem_195);  view_713 = getitem_195 = None
        mul_220: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_58);  sub_84 = rsqrt_58 = None
        mul_221: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_220, arg34_1);  mul_220 = arg34_1 = None
        add_281: "f32[8, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_221, arg35_1);  mul_221 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:444 in forward, code: x = self.reduction(x)
        permute_270: "f32[512, 256]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        view_714: "f32[6272, 512]" = torch.ops.aten.view.default(add_281, [6272, 512]);  add_281 = None
        mm_3: "f32[6272, 256]" = torch.ops.aten.mm.default(view_714, permute_270);  view_714 = permute_270 = None
        view_715: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(mm_3, [8, 28, 28, 256]);  mm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_59 = torch.ops.aten.var_mean.correction(view_715, [3], correction = 0, keepdim = True)
        getitem_196: "f32[8, 28, 28, 1]" = var_mean_59[0]
        getitem_197: "f32[8, 28, 28, 1]" = var_mean_59[1];  var_mean_59 = None
        add_282: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_59: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_85: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(view_715, getitem_197);  getitem_197 = None
        mul_222: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_59);  sub_85 = rsqrt_59 = None
        mul_223: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_222, arg37_1);  mul_222 = arg37_1 = None
        add_283: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_223, arg38_1);  mul_223 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_716: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(add_283, [8, 4, 7, 4, 7, 256]);  add_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_271: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_716, [0, 1, 3, 2, 4, 5]);  view_716 = None
        clone_289: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_717: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(clone_289, [-1, 7, 7, 256]);  clone_289 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_718: "f32[128, 49, 256]" = torch.ops.aten.view.default(view_717, [-1, 49, 256]);  view_717 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_719: "f32[6272, 256]" = torch.ops.aten.view.default(view_718, [6272, 256]);  view_718 = None
        permute_272: "f32[256, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_105: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg40_1, view_719, permute_272);  arg40_1 = view_719 = permute_272 = None
        view_720: "f32[128, 49, 768]" = torch.ops.aten.view.default(addmm_105, [128, 49, 768]);  addmm_105 = None
        view_721: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.view.default(view_720, [128, 49, 3, 8, -1]);  view_720 = None
        permute_273: "f32[3, 128, 8, 49, 32]" = torch.ops.aten.permute.default(view_721, [2, 0, 3, 1, 4]);  view_721 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_26 = torch.ops.aten.unbind.int(permute_273);  permute_273 = None
        getitem_198: "f32[128, 8, 49, 32]" = unbind_26[0]
        getitem_199: "f32[128, 8, 49, 32]" = unbind_26[1]
        getitem_200: "f32[128, 8, 49, 32]" = unbind_26[2];  unbind_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_224: "f32[128, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_198, 0.1767766952966369);  getitem_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_274: "f32[128, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_199, [0, 1, 3, 2]);  getitem_199 = None
        expand_104: "f32[128, 8, 49, 32]" = torch.ops.aten.expand.default(mul_224, [128, 8, 49, 32]);  mul_224 = None
        clone_290: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_722: "f32[1024, 49, 32]" = torch.ops.aten.view.default(clone_290, [1024, 49, 32]);  clone_290 = None
        expand_105: "f32[128, 8, 32, 49]" = torch.ops.aten.expand.default(permute_274, [128, 8, 32, 49]);  permute_274 = None
        clone_291: "f32[128, 8, 32, 49]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_723: "f32[1024, 32, 49]" = torch.ops.aten.view.default(clone_291, [1024, 32, 49]);  clone_291 = None
        bmm_52: "f32[1024, 49, 49]" = torch.ops.aten.bmm.default(view_722, view_723);  view_722 = view_723 = None
        view_724: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(bmm_52, [128, 8, 49, 49]);  bmm_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_725: "i64[2401]" = torch.ops.aten.view.default(arg42_1, [-1]);  arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_74: "f32[2401, 8]" = torch.ops.aten.index.Tensor(arg41_1, [view_725]);  arg41_1 = view_725 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_726: "f32[49, 49, 8]" = torch.ops.aten.view.default(index_74, [49, 49, -1]);  index_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_275: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_726, [2, 0, 1]);  view_726 = None
        clone_292: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_50: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_292, 0);  clone_292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_284: "f32[128, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_724, unsqueeze_50);  view_724 = unsqueeze_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_26: "f32[128, 8, 49, 1]" = torch.ops.aten.amax.default(add_284, [-1], True)
        sub_86: "f32[128, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_284, amax_26);  add_284 = amax_26 = None
        exp_26: "f32[128, 8, 49, 49]" = torch.ops.aten.exp.default(sub_86);  sub_86 = None
        sum_27: "f32[128, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_26: "f32[128, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_106: "f32[128, 8, 49, 49]" = torch.ops.aten.expand.default(div_26, [128, 8, 49, 49]);  div_26 = None
        view_727: "f32[1024, 49, 49]" = torch.ops.aten.view.default(expand_106, [1024, 49, 49]);  expand_106 = None
        expand_107: "f32[128, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_200, [128, 8, 49, 32]);  getitem_200 = None
        clone_294: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_728: "f32[1024, 49, 32]" = torch.ops.aten.view.default(clone_294, [1024, 49, 32]);  clone_294 = None
        bmm_53: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(view_727, view_728);  view_727 = view_728 = None
        view_729: "f32[128, 8, 49, 32]" = torch.ops.aten.view.default(bmm_53, [128, 8, 49, 32]);  bmm_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_276: "f32[128, 49, 8, 32]" = torch.ops.aten.permute.default(view_729, [0, 2, 1, 3]);  view_729 = None
        clone_295: "f32[128, 49, 8, 32]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
        view_730: "f32[128, 49, 256]" = torch.ops.aten.view.default(clone_295, [128, 49, 256]);  clone_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_731: "f32[6272, 256]" = torch.ops.aten.view.default(view_730, [6272, 256]);  view_730 = None
        permute_277: "f32[256, 256]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_106: "f32[6272, 256]" = torch.ops.aten.addmm.default(arg44_1, view_731, permute_277);  arg44_1 = view_731 = permute_277 = None
        view_732: "f32[128, 49, 256]" = torch.ops.aten.view.default(addmm_106, [128, 49, 256]);  addmm_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_733: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(view_732, [-1, 7, 7, 256]);  view_732 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_734: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_733, [-1, 4, 4, 7, 7, 256]);  view_733 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_278: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_734, [0, 1, 3, 2, 4, 5]);  view_734 = None
        clone_297: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_278, memory_format = torch.contiguous_format);  permute_278 = None
        view_735: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(clone_297, [-1, 28, 28, 256]);  clone_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_285: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_715, view_735);  view_715 = view_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_736: "f32[8, 784, 256]" = torch.ops.aten.view.default(add_285, [8, -1, 256]);  add_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_60 = torch.ops.aten.var_mean.correction(view_736, [2], correction = 0, keepdim = True)
        getitem_201: "f32[8, 784, 1]" = var_mean_60[0]
        getitem_202: "f32[8, 784, 1]" = var_mean_60[1];  var_mean_60 = None
        add_286: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_201, 1e-05);  getitem_201 = None
        rsqrt_60: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_87: "f32[8, 784, 256]" = torch.ops.aten.sub.Tensor(view_736, getitem_202);  getitem_202 = None
        mul_225: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_60);  sub_87 = rsqrt_60 = None
        mul_226: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(mul_225, arg45_1);  mul_225 = arg45_1 = None
        add_287: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(mul_226, arg46_1);  mul_226 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_737: "f32[6272, 256]" = torch.ops.aten.view.default(add_287, [6272, 256]);  add_287 = None
        permute_279: "f32[256, 1024]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_107: "f32[6272, 1024]" = torch.ops.aten.addmm.default(arg48_1, view_737, permute_279);  arg48_1 = view_737 = permute_279 = None
        view_738: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_107, [8, 784, 1024]);  addmm_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_227: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_738, 0.5)
        mul_228: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_738, 0.7071067811865476);  view_738 = None
        erf_26: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_228);  mul_228 = None
        add_288: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_229: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_227, add_288);  mul_227 = add_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_739: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_229, [6272, 1024]);  mul_229 = None
        permute_280: "f32[1024, 256]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_108: "f32[6272, 256]" = torch.ops.aten.addmm.default(arg50_1, view_739, permute_280);  arg50_1 = view_739 = permute_280 = None
        view_740: "f32[8, 784, 256]" = torch.ops.aten.view.default(addmm_108, [8, 784, 256]);  addmm_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_289: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(view_736, view_740);  view_736 = view_740 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_741: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(add_289, [8, 28, 28, 256]);  add_289 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_61 = torch.ops.aten.var_mean.correction(view_741, [3], correction = 0, keepdim = True)
        getitem_203: "f32[8, 28, 28, 1]" = var_mean_61[0]
        getitem_204: "f32[8, 28, 28, 1]" = var_mean_61[1];  var_mean_61 = None
        add_290: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_203, 1e-05);  getitem_203 = None
        rsqrt_61: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        sub_88: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(view_741, getitem_204);  getitem_204 = None
        mul_230: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_61);  sub_88 = rsqrt_61 = None
        mul_231: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_230, arg51_1);  mul_230 = arg51_1 = None
        add_291: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_231, arg52_1);  mul_231 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_48: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_292: "i64[28]" = torch.ops.aten.add.Tensor(iota_48, 3);  iota_48 = None
        fmod_48: "i64[28]" = torch.ops.aten.fmod.Scalar(add_292, 28);  add_292 = None
        index_75: "f32[8, 28, 28, 256]" = torch.ops.aten.index.Tensor(add_291, [None, fmod_48]);  add_291 = fmod_48 = None
        iota_49: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_293: "i64[28]" = torch.ops.aten.add.Tensor(iota_49, 3);  iota_49 = None
        fmod_49: "i64[28]" = torch.ops.aten.fmod.Scalar(add_293, 28);  add_293 = None
        index_76: "f32[8, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_75, [None, None, fmod_49]);  index_75 = fmod_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_742: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(index_76, [8, 4, 7, 4, 7, 256]);  index_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_281: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_742, [0, 1, 3, 2, 4, 5]);  view_742 = None
        clone_300: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_743: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(clone_300, [-1, 7, 7, 256]);  clone_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_744: "f32[128, 49, 256]" = torch.ops.aten.view.default(view_743, [-1, 49, 256]);  view_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_745: "f32[6272, 256]" = torch.ops.aten.view.default(view_744, [6272, 256]);  view_744 = None
        permute_282: "f32[256, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_109: "f32[6272, 768]" = torch.ops.aten.addmm.default(arg55_1, view_745, permute_282);  arg55_1 = view_745 = permute_282 = None
        view_746: "f32[128, 49, 768]" = torch.ops.aten.view.default(addmm_109, [128, 49, 768]);  addmm_109 = None
        view_747: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.view.default(view_746, [128, 49, 3, 8, -1]);  view_746 = None
        permute_283: "f32[3, 128, 8, 49, 32]" = torch.ops.aten.permute.default(view_747, [2, 0, 3, 1, 4]);  view_747 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_27 = torch.ops.aten.unbind.int(permute_283);  permute_283 = None
        getitem_205: "f32[128, 8, 49, 32]" = unbind_27[0]
        getitem_206: "f32[128, 8, 49, 32]" = unbind_27[1]
        getitem_207: "f32[128, 8, 49, 32]" = unbind_27[2];  unbind_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_232: "f32[128, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_205, 0.1767766952966369);  getitem_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_284: "f32[128, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_206, [0, 1, 3, 2]);  getitem_206 = None
        expand_108: "f32[128, 8, 49, 32]" = torch.ops.aten.expand.default(mul_232, [128, 8, 49, 32]);  mul_232 = None
        clone_301: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_748: "f32[1024, 49, 32]" = torch.ops.aten.view.default(clone_301, [1024, 49, 32]);  clone_301 = None
        expand_109: "f32[128, 8, 32, 49]" = torch.ops.aten.expand.default(permute_284, [128, 8, 32, 49]);  permute_284 = None
        clone_302: "f32[128, 8, 32, 49]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_749: "f32[1024, 32, 49]" = torch.ops.aten.view.default(clone_302, [1024, 32, 49]);  clone_302 = None
        bmm_54: "f32[1024, 49, 49]" = torch.ops.aten.bmm.default(view_748, view_749);  view_748 = view_749 = None
        view_750: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(bmm_54, [128, 8, 49, 49]);  bmm_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_751: "i64[2401]" = torch.ops.aten.view.default(arg57_1, [-1]);  arg57_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_77: "f32[2401, 8]" = torch.ops.aten.index.Tensor(arg56_1, [view_751]);  arg56_1 = view_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_752: "f32[49, 49, 8]" = torch.ops.aten.view.default(index_77, [49, 49, -1]);  index_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_285: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_752, [2, 0, 1]);  view_752 = None
        clone_303: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_51: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_303, 0);  clone_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_294: "f32[128, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_750, unsqueeze_51);  view_750 = unsqueeze_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_753: "f32[8, 16, 8, 49, 49]" = torch.ops.aten.view.default(add_294, [-1, 16, 8, 49, 49]);  add_294 = None
        unsqueeze_52: "f32[16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg53_1, 1);  arg53_1 = None
        unsqueeze_53: "f32[1, 16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 0);  unsqueeze_52 = None
        add_295: "f32[8, 16, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_753, unsqueeze_53);  view_753 = unsqueeze_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_754: "f32[128, 8, 49, 49]" = torch.ops.aten.view.default(add_295, [-1, 8, 49, 49]);  add_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_27: "f32[128, 8, 49, 1]" = torch.ops.aten.amax.default(view_754, [-1], True)
        sub_89: "f32[128, 8, 49, 49]" = torch.ops.aten.sub.Tensor(view_754, amax_27);  view_754 = amax_27 = None
        exp_27: "f32[128, 8, 49, 49]" = torch.ops.aten.exp.default(sub_89);  sub_89 = None
        sum_28: "f32[128, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_27: "f32[128, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_110: "f32[128, 8, 49, 49]" = torch.ops.aten.expand.default(div_27, [128, 8, 49, 49]);  div_27 = None
        view_755: "f32[1024, 49, 49]" = torch.ops.aten.view.default(expand_110, [1024, 49, 49]);  expand_110 = None
        expand_111: "f32[128, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_207, [128, 8, 49, 32]);  getitem_207 = None
        clone_305: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_756: "f32[1024, 49, 32]" = torch.ops.aten.view.default(clone_305, [1024, 49, 32]);  clone_305 = None
        bmm_55: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(view_755, view_756);  view_755 = view_756 = None
        view_757: "f32[128, 8, 49, 32]" = torch.ops.aten.view.default(bmm_55, [128, 8, 49, 32]);  bmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_286: "f32[128, 49, 8, 32]" = torch.ops.aten.permute.default(view_757, [0, 2, 1, 3]);  view_757 = None
        clone_306: "f32[128, 49, 8, 32]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_758: "f32[128, 49, 256]" = torch.ops.aten.view.default(clone_306, [128, 49, 256]);  clone_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_759: "f32[6272, 256]" = torch.ops.aten.view.default(view_758, [6272, 256]);  view_758 = None
        permute_287: "f32[256, 256]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_110: "f32[6272, 256]" = torch.ops.aten.addmm.default(arg59_1, view_759, permute_287);  arg59_1 = view_759 = permute_287 = None
        view_760: "f32[128, 49, 256]" = torch.ops.aten.view.default(addmm_110, [128, 49, 256]);  addmm_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_761: "f32[128, 7, 7, 256]" = torch.ops.aten.view.default(view_760, [-1, 7, 7, 256]);  view_760 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_762: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_761, [-1, 4, 4, 7, 7, 256]);  view_761 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_288: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_762, [0, 1, 3, 2, 4, 5]);  view_762 = None
        clone_308: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_763: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(clone_308, [-1, 28, 28, 256]);  clone_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_50: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_296: "i64[28]" = torch.ops.aten.add.Tensor(iota_50, 25);  iota_50 = None
        fmod_50: "i64[28]" = torch.ops.aten.fmod.Scalar(add_296, 28);  add_296 = None
        index_78: "f32[8, 28, 28, 256]" = torch.ops.aten.index.Tensor(view_763, [None, fmod_50]);  view_763 = fmod_50 = None
        iota_51: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_297: "i64[28]" = torch.ops.aten.add.Tensor(iota_51, 25);  iota_51 = None
        fmod_51: "i64[28]" = torch.ops.aten.fmod.Scalar(add_297, 28);  add_297 = None
        index_79: "f32[8, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_78, [None, None, fmod_51]);  index_78 = fmod_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_298: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_741, index_79);  view_741 = index_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_764: "f32[8, 784, 256]" = torch.ops.aten.view.default(add_298, [8, -1, 256]);  add_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_62 = torch.ops.aten.var_mean.correction(view_764, [2], correction = 0, keepdim = True)
        getitem_208: "f32[8, 784, 1]" = var_mean_62[0]
        getitem_209: "f32[8, 784, 1]" = var_mean_62[1];  var_mean_62 = None
        add_299: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_62: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
        sub_90: "f32[8, 784, 256]" = torch.ops.aten.sub.Tensor(view_764, getitem_209);  getitem_209 = None
        mul_233: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_62);  sub_90 = rsqrt_62 = None
        mul_234: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(mul_233, arg60_1);  mul_233 = arg60_1 = None
        add_300: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(mul_234, arg61_1);  mul_234 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_765: "f32[6272, 256]" = torch.ops.aten.view.default(add_300, [6272, 256]);  add_300 = None
        permute_289: "f32[256, 1024]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_111: "f32[6272, 1024]" = torch.ops.aten.addmm.default(arg63_1, view_765, permute_289);  arg63_1 = view_765 = permute_289 = None
        view_766: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_111, [8, 784, 1024]);  addmm_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_235: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_766, 0.5)
        mul_236: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_766, 0.7071067811865476);  view_766 = None
        erf_27: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_236);  mul_236 = None
        add_301: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_237: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_235, add_301);  mul_235 = add_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_767: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_237, [6272, 1024]);  mul_237 = None
        permute_290: "f32[1024, 256]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_112: "f32[6272, 256]" = torch.ops.aten.addmm.default(arg65_1, view_767, permute_290);  arg65_1 = view_767 = permute_290 = None
        view_768: "f32[8, 784, 256]" = torch.ops.aten.view.default(addmm_112, [8, 784, 256]);  addmm_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_302: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(view_764, view_768);  view_764 = view_768 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_769: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(add_302, [8, 28, 28, 256]);  add_302 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:442 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_770: "f32[8, 14, 2, 14, 2, 256]" = torch.ops.aten.view.default(view_769, [8, 14, 2, 14, 2, 256]);  view_769 = None
        permute_291: "f32[8, 14, 14, 2, 2, 256]" = torch.ops.aten.permute.default(view_770, [0, 1, 3, 4, 2, 5]);  view_770 = None
        clone_311: "f32[8, 14, 14, 2, 2, 256]" = torch.ops.aten.clone.default(permute_291, memory_format = torch.contiguous_format);  permute_291 = None
        view_771: "f32[8, 14, 14, 1024]" = torch.ops.aten.view.default(clone_311, [8, 14, 14, 1024]);  clone_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in forward, code: x = self.norm(x)
        var_mean_63 = torch.ops.aten.var_mean.correction(view_771, [3], correction = 0, keepdim = True)
        getitem_210: "f32[8, 14, 14, 1]" = var_mean_63[0]
        getitem_211: "f32[8, 14, 14, 1]" = var_mean_63[1];  var_mean_63 = None
        add_303: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_63: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_91: "f32[8, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(view_771, getitem_211);  view_771 = getitem_211 = None
        mul_238: "f32[8, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_63);  sub_91 = rsqrt_63 = None
        mul_239: "f32[8, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_238, arg66_1);  mul_238 = arg66_1 = None
        add_304: "f32[8, 14, 14, 1024]" = torch.ops.aten.add.Tensor(mul_239, arg67_1);  mul_239 = arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:444 in forward, code: x = self.reduction(x)
        permute_292: "f32[1024, 512]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        view_772: "f32[1568, 1024]" = torch.ops.aten.view.default(add_304, [1568, 1024]);  add_304 = None
        mm_4: "f32[1568, 512]" = torch.ops.aten.mm.default(view_772, permute_292);  view_772 = permute_292 = None
        view_773: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_4, [8, 14, 14, 512]);  mm_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_64 = torch.ops.aten.var_mean.correction(view_773, [3], correction = 0, keepdim = True)
        getitem_212: "f32[8, 14, 14, 1]" = var_mean_64[0]
        getitem_213: "f32[8, 14, 14, 1]" = var_mean_64[1];  var_mean_64 = None
        add_305: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_64: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_305);  add_305 = None
        sub_92: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_773, getitem_213);  getitem_213 = None
        mul_240: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_64);  sub_92 = rsqrt_64 = None
        mul_241: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_240, arg69_1);  mul_240 = arg69_1 = None
        add_306: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_241, arg70_1);  mul_241 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_774: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_306, [8, 2, 7, 2, 7, 512]);  add_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_293: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_774, [0, 1, 3, 2, 4, 5]);  view_774 = None
        clone_312: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_775: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_312, [-1, 7, 7, 512]);  clone_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_776: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_775, [-1, 49, 512]);  view_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_777: "f32[1568, 512]" = torch.ops.aten.view.default(view_776, [1568, 512]);  view_776 = None
        permute_294: "f32[512, 1536]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_113: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg72_1, view_777, permute_294);  arg72_1 = view_777 = permute_294 = None
        view_778: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_113, [32, 49, 1536]);  addmm_113 = None
        view_779: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_778, [32, 49, 3, 16, -1]);  view_778 = None
        permute_295: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_779, [2, 0, 3, 1, 4]);  view_779 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_28 = torch.ops.aten.unbind.int(permute_295);  permute_295 = None
        getitem_214: "f32[32, 16, 49, 32]" = unbind_28[0]
        getitem_215: "f32[32, 16, 49, 32]" = unbind_28[1]
        getitem_216: "f32[32, 16, 49, 32]" = unbind_28[2];  unbind_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_242: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_214, 0.1767766952966369);  getitem_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_296: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_215, [0, 1, 3, 2]);  getitem_215 = None
        expand_112: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_242, [32, 16, 49, 32]);  mul_242 = None
        clone_313: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
        view_780: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_313, [512, 49, 32]);  clone_313 = None
        expand_113: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_296, [32, 16, 32, 49]);  permute_296 = None
        clone_314: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_781: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_314, [512, 32, 49]);  clone_314 = None
        bmm_56: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_56, [32, 16, 49, 49]);  bmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_783: "i64[2401]" = torch.ops.aten.view.default(arg74_1, [-1]);  arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_80: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg73_1, [view_783]);  arg73_1 = view_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_784: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_80, [49, 49, -1]);  index_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_297: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_784, [2, 0, 1]);  view_784 = None
        clone_315: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_297, memory_format = torch.contiguous_format);  permute_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_54: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_315, 0);  clone_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_307: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_782, unsqueeze_54);  view_782 = unsqueeze_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_28: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_307, [-1], True)
        sub_93: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_307, amax_28);  add_307 = amax_28 = None
        exp_28: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_29: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_28: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_114: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_28, [32, 16, 49, 49]);  div_28 = None
        view_785: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_114, [512, 49, 49]);  expand_114 = None
        expand_115: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_216, [32, 16, 49, 32]);  getitem_216 = None
        clone_317: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_786: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_317, [512, 49, 32]);  clone_317 = None
        bmm_57: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_785, view_786);  view_785 = view_786 = None
        view_787: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_57, [32, 16, 49, 32]);  bmm_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_298: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_787, [0, 2, 1, 3]);  view_787 = None
        clone_318: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
        view_788: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_318, [32, 49, 512]);  clone_318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_789: "f32[1568, 512]" = torch.ops.aten.view.default(view_788, [1568, 512]);  view_788 = None
        permute_299: "f32[512, 512]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_114: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg76_1, view_789, permute_299);  arg76_1 = view_789 = permute_299 = None
        view_790: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_114, [32, 49, 512]);  addmm_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_791: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_790, [-1, 7, 7, 512]);  view_790 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_792: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_791, [-1, 2, 2, 7, 7, 512]);  view_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_300: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_792, [0, 1, 3, 2, 4, 5]);  view_792 = None
        clone_320: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
        view_793: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_320, [-1, 14, 14, 512]);  clone_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_308: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_773, view_793);  view_773 = view_793 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_794: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_308, [8, -1, 512]);  add_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_65 = torch.ops.aten.var_mean.correction(view_794, [2], correction = 0, keepdim = True)
        getitem_217: "f32[8, 196, 1]" = var_mean_65[0]
        getitem_218: "f32[8, 196, 1]" = var_mean_65[1];  var_mean_65 = None
        add_309: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_217, 1e-05);  getitem_217 = None
        rsqrt_65: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
        sub_94: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_794, getitem_218);  getitem_218 = None
        mul_243: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_65);  sub_94 = rsqrt_65 = None
        mul_244: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_243, arg77_1);  mul_243 = arg77_1 = None
        add_310: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_244, arg78_1);  mul_244 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_795: "f32[1568, 512]" = torch.ops.aten.view.default(add_310, [1568, 512]);  add_310 = None
        permute_301: "f32[512, 2048]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_115: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg80_1, view_795, permute_301);  arg80_1 = view_795 = permute_301 = None
        view_796: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_115, [8, 196, 2048]);  addmm_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_245: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_796, 0.5)
        mul_246: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_796, 0.7071067811865476);  view_796 = None
        erf_28: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_246);  mul_246 = None
        add_311: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_247: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_245, add_311);  mul_245 = add_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_797: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_247, [1568, 2048]);  mul_247 = None
        permute_302: "f32[2048, 512]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_116: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg82_1, view_797, permute_302);  arg82_1 = view_797 = permute_302 = None
        view_798: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_116, [8, 196, 512]);  addmm_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_312: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_794, view_798);  view_794 = view_798 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_799: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_312, [8, 14, 14, 512]);  add_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_66 = torch.ops.aten.var_mean.correction(view_799, [3], correction = 0, keepdim = True)
        getitem_219: "f32[8, 14, 14, 1]" = var_mean_66[0]
        getitem_220: "f32[8, 14, 14, 1]" = var_mean_66[1];  var_mean_66 = None
        add_313: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_219, 1e-05);  getitem_219 = None
        rsqrt_66: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
        sub_95: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_799, getitem_220);  getitem_220 = None
        mul_248: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_66);  sub_95 = rsqrt_66 = None
        mul_249: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_248, arg83_1);  mul_248 = arg83_1 = None
        add_314: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_249, arg84_1);  mul_249 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_52: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_315: "i64[14]" = torch.ops.aten.add.Tensor(iota_52, 3);  iota_52 = None
        fmod_52: "i64[14]" = torch.ops.aten.fmod.Scalar(add_315, 14);  add_315 = None
        index_81: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_314, [None, fmod_52]);  add_314 = fmod_52 = None
        iota_53: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_316: "i64[14]" = torch.ops.aten.add.Tensor(iota_53, 3);  iota_53 = None
        fmod_53: "i64[14]" = torch.ops.aten.fmod.Scalar(add_316, 14);  add_316 = None
        index_82: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_81, [None, None, fmod_53]);  index_81 = fmod_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_800: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_82, [8, 2, 7, 2, 7, 512]);  index_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_303: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_800, [0, 1, 3, 2, 4, 5]);  view_800 = None
        clone_323: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_801: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_323, [-1, 7, 7, 512]);  clone_323 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_802: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_801, [-1, 49, 512]);  view_801 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_803: "f32[1568, 512]" = torch.ops.aten.view.default(view_802, [1568, 512]);  view_802 = None
        permute_304: "f32[512, 1536]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_117: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg87_1, view_803, permute_304);  arg87_1 = view_803 = permute_304 = None
        view_804: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_117, [32, 49, 1536]);  addmm_117 = None
        view_805: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_804, [32, 49, 3, 16, -1]);  view_804 = None
        permute_305: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_805, [2, 0, 3, 1, 4]);  view_805 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_29 = torch.ops.aten.unbind.int(permute_305);  permute_305 = None
        getitem_221: "f32[32, 16, 49, 32]" = unbind_29[0]
        getitem_222: "f32[32, 16, 49, 32]" = unbind_29[1]
        getitem_223: "f32[32, 16, 49, 32]" = unbind_29[2];  unbind_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_250: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_221, 0.1767766952966369);  getitem_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_306: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_222, [0, 1, 3, 2]);  getitem_222 = None
        expand_116: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_250, [32, 16, 49, 32]);  mul_250 = None
        clone_324: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_806: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_324, [512, 49, 32]);  clone_324 = None
        expand_117: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_306, [32, 16, 32, 49]);  permute_306 = None
        clone_325: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_807: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_325, [512, 32, 49]);  clone_325 = None
        bmm_58: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_806, view_807);  view_806 = view_807 = None
        view_808: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_58, [32, 16, 49, 49]);  bmm_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_809: "i64[2401]" = torch.ops.aten.view.default(arg89_1, [-1]);  arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_83: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg88_1, [view_809]);  arg88_1 = view_809 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_810: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_83, [49, 49, -1]);  index_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_307: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_810, [2, 0, 1]);  view_810 = None
        clone_326: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_55: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_326, 0);  clone_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_317: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_808, unsqueeze_55);  view_808 = unsqueeze_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_811: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_317, [-1, 4, 16, 49, 49]);  add_317 = None
        unsqueeze_56: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg85_1, 1);  arg85_1 = None
        unsqueeze_57: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, 0);  unsqueeze_56 = None
        add_318: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_811, unsqueeze_57);  view_811 = unsqueeze_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_812: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_318, [-1, 16, 49, 49]);  add_318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_29: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_812, [-1], True)
        sub_96: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_812, amax_29);  view_812 = amax_29 = None
        exp_29: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_96);  sub_96 = None
        sum_30: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_29: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_118: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_29, [32, 16, 49, 49]);  div_29 = None
        view_813: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_118, [512, 49, 49]);  expand_118 = None
        expand_119: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_223, [32, 16, 49, 32]);  getitem_223 = None
        clone_328: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
        view_814: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_328, [512, 49, 32]);  clone_328 = None
        bmm_59: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_813, view_814);  view_813 = view_814 = None
        view_815: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_59, [32, 16, 49, 32]);  bmm_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_308: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_815, [0, 2, 1, 3]);  view_815 = None
        clone_329: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
        view_816: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_329, [32, 49, 512]);  clone_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_817: "f32[1568, 512]" = torch.ops.aten.view.default(view_816, [1568, 512]);  view_816 = None
        permute_309: "f32[512, 512]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_118: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg91_1, view_817, permute_309);  arg91_1 = view_817 = permute_309 = None
        view_818: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_118, [32, 49, 512]);  addmm_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_819: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_818, [-1, 7, 7, 512]);  view_818 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_820: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_819, [-1, 2, 2, 7, 7, 512]);  view_819 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_310: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_820, [0, 1, 3, 2, 4, 5]);  view_820 = None
        clone_331: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        view_821: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_331, [-1, 14, 14, 512]);  clone_331 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_54: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_319: "i64[14]" = torch.ops.aten.add.Tensor(iota_54, 11);  iota_54 = None
        fmod_54: "i64[14]" = torch.ops.aten.fmod.Scalar(add_319, 14);  add_319 = None
        index_84: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_821, [None, fmod_54]);  view_821 = fmod_54 = None
        iota_55: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_320: "i64[14]" = torch.ops.aten.add.Tensor(iota_55, 11);  iota_55 = None
        fmod_55: "i64[14]" = torch.ops.aten.fmod.Scalar(add_320, 14);  add_320 = None
        index_85: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_84, [None, None, fmod_55]);  index_84 = fmod_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_321: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_799, index_85);  view_799 = index_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_822: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_321, [8, -1, 512]);  add_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_67 = torch.ops.aten.var_mean.correction(view_822, [2], correction = 0, keepdim = True)
        getitem_224: "f32[8, 196, 1]" = var_mean_67[0]
        getitem_225: "f32[8, 196, 1]" = var_mean_67[1];  var_mean_67 = None
        add_322: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_67: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_322);  add_322 = None
        sub_97: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_822, getitem_225);  getitem_225 = None
        mul_251: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_67);  sub_97 = rsqrt_67 = None
        mul_252: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_251, arg92_1);  mul_251 = arg92_1 = None
        add_323: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_252, arg93_1);  mul_252 = arg93_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_823: "f32[1568, 512]" = torch.ops.aten.view.default(add_323, [1568, 512]);  add_323 = None
        permute_311: "f32[512, 2048]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_119: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg95_1, view_823, permute_311);  arg95_1 = view_823 = permute_311 = None
        view_824: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_119, [8, 196, 2048]);  addmm_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_253: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_824, 0.5)
        mul_254: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_824, 0.7071067811865476);  view_824 = None
        erf_29: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_254);  mul_254 = None
        add_324: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_255: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_253, add_324);  mul_253 = add_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_825: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_255, [1568, 2048]);  mul_255 = None
        permute_312: "f32[2048, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_120: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg97_1, view_825, permute_312);  arg97_1 = view_825 = permute_312 = None
        view_826: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_120, [8, 196, 512]);  addmm_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_325: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_822, view_826);  view_822 = view_826 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_827: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_325, [8, 14, 14, 512]);  add_325 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_68 = torch.ops.aten.var_mean.correction(view_827, [3], correction = 0, keepdim = True)
        getitem_226: "f32[8, 14, 14, 1]" = var_mean_68[0]
        getitem_227: "f32[8, 14, 14, 1]" = var_mean_68[1];  var_mean_68 = None
        add_326: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_68: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
        sub_98: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_827, getitem_227);  getitem_227 = None
        mul_256: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_68);  sub_98 = rsqrt_68 = None
        mul_257: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_256, arg98_1);  mul_256 = arg98_1 = None
        add_327: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_257, arg99_1);  mul_257 = arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_828: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_327, [8, 2, 7, 2, 7, 512]);  add_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_313: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_828, [0, 1, 3, 2, 4, 5]);  view_828 = None
        clone_334: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
        view_829: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_334, [-1, 7, 7, 512]);  clone_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_830: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_829, [-1, 49, 512]);  view_829 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_831: "f32[1568, 512]" = torch.ops.aten.view.default(view_830, [1568, 512]);  view_830 = None
        permute_314: "f32[512, 1536]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_121: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg101_1, view_831, permute_314);  arg101_1 = view_831 = permute_314 = None
        view_832: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_121, [32, 49, 1536]);  addmm_121 = None
        view_833: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_832, [32, 49, 3, 16, -1]);  view_832 = None
        permute_315: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_833, [2, 0, 3, 1, 4]);  view_833 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_30 = torch.ops.aten.unbind.int(permute_315);  permute_315 = None
        getitem_228: "f32[32, 16, 49, 32]" = unbind_30[0]
        getitem_229: "f32[32, 16, 49, 32]" = unbind_30[1]
        getitem_230: "f32[32, 16, 49, 32]" = unbind_30[2];  unbind_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_258: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_228, 0.1767766952966369);  getitem_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_316: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_229, [0, 1, 3, 2]);  getitem_229 = None
        expand_120: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_258, [32, 16, 49, 32]);  mul_258 = None
        clone_335: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
        view_834: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_335, [512, 49, 32]);  clone_335 = None
        expand_121: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_316, [32, 16, 32, 49]);  permute_316 = None
        clone_336: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_835: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_336, [512, 32, 49]);  clone_336 = None
        bmm_60: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_834, view_835);  view_834 = view_835 = None
        view_836: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_60, [32, 16, 49, 49]);  bmm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_837: "i64[2401]" = torch.ops.aten.view.default(arg103_1, [-1]);  arg103_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_86: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg102_1, [view_837]);  arg102_1 = view_837 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_838: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_86, [49, 49, -1]);  index_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_317: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_838, [2, 0, 1]);  view_838 = None
        clone_337: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_58: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_337, 0);  clone_337 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_328: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_836, unsqueeze_58);  view_836 = unsqueeze_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_30: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_328, [-1], True)
        sub_99: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_328, amax_30);  add_328 = amax_30 = None
        exp_30: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_99);  sub_99 = None
        sum_31: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_30: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_122: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_30, [32, 16, 49, 49]);  div_30 = None
        view_839: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_122, [512, 49, 49]);  expand_122 = None
        expand_123: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_230, [32, 16, 49, 32]);  getitem_230 = None
        clone_339: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_840: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_339, [512, 49, 32]);  clone_339 = None
        bmm_61: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_839, view_840);  view_839 = view_840 = None
        view_841: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_61, [32, 16, 49, 32]);  bmm_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_318: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_841, [0, 2, 1, 3]);  view_841 = None
        clone_340: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
        view_842: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_340, [32, 49, 512]);  clone_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_843: "f32[1568, 512]" = torch.ops.aten.view.default(view_842, [1568, 512]);  view_842 = None
        permute_319: "f32[512, 512]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_122: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg105_1, view_843, permute_319);  arg105_1 = view_843 = permute_319 = None
        view_844: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_122, [32, 49, 512]);  addmm_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_845: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_844, [-1, 7, 7, 512]);  view_844 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_846: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_845, [-1, 2, 2, 7, 7, 512]);  view_845 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_320: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_846, [0, 1, 3, 2, 4, 5]);  view_846 = None
        clone_342: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_847: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_342, [-1, 14, 14, 512]);  clone_342 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_329: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_827, view_847);  view_827 = view_847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_848: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_329, [8, -1, 512]);  add_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_69 = torch.ops.aten.var_mean.correction(view_848, [2], correction = 0, keepdim = True)
        getitem_231: "f32[8, 196, 1]" = var_mean_69[0]
        getitem_232: "f32[8, 196, 1]" = var_mean_69[1];  var_mean_69 = None
        add_330: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_231, 1e-05);  getitem_231 = None
        rsqrt_69: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_330);  add_330 = None
        sub_100: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_848, getitem_232);  getitem_232 = None
        mul_259: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_69);  sub_100 = rsqrt_69 = None
        mul_260: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_259, arg106_1);  mul_259 = arg106_1 = None
        add_331: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_260, arg107_1);  mul_260 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_849: "f32[1568, 512]" = torch.ops.aten.view.default(add_331, [1568, 512]);  add_331 = None
        permute_321: "f32[512, 2048]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_123: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg109_1, view_849, permute_321);  arg109_1 = view_849 = permute_321 = None
        view_850: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_123, [8, 196, 2048]);  addmm_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_261: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_850, 0.5)
        mul_262: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_850, 0.7071067811865476);  view_850 = None
        erf_30: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_262);  mul_262 = None
        add_332: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_263: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_261, add_332);  mul_261 = add_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_851: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_263, [1568, 2048]);  mul_263 = None
        permute_322: "f32[2048, 512]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_124: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg111_1, view_851, permute_322);  arg111_1 = view_851 = permute_322 = None
        view_852: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_124, [8, 196, 512]);  addmm_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_333: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_848, view_852);  view_848 = view_852 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_853: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_333, [8, 14, 14, 512]);  add_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_70 = torch.ops.aten.var_mean.correction(view_853, [3], correction = 0, keepdim = True)
        getitem_233: "f32[8, 14, 14, 1]" = var_mean_70[0]
        getitem_234: "f32[8, 14, 14, 1]" = var_mean_70[1];  var_mean_70 = None
        add_334: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_233, 1e-05);  getitem_233 = None
        rsqrt_70: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
        sub_101: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_853, getitem_234);  getitem_234 = None
        mul_264: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_70);  sub_101 = rsqrt_70 = None
        mul_265: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_264, arg112_1);  mul_264 = arg112_1 = None
        add_335: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_265, arg113_1);  mul_265 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_56: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_336: "i64[14]" = torch.ops.aten.add.Tensor(iota_56, 3);  iota_56 = None
        fmod_56: "i64[14]" = torch.ops.aten.fmod.Scalar(add_336, 14);  add_336 = None
        index_87: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_335, [None, fmod_56]);  add_335 = fmod_56 = None
        iota_57: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_337: "i64[14]" = torch.ops.aten.add.Tensor(iota_57, 3);  iota_57 = None
        fmod_57: "i64[14]" = torch.ops.aten.fmod.Scalar(add_337, 14);  add_337 = None
        index_88: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_87, [None, None, fmod_57]);  index_87 = fmod_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_854: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_88, [8, 2, 7, 2, 7, 512]);  index_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_323: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_854, [0, 1, 3, 2, 4, 5]);  view_854 = None
        clone_345: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
        view_855: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_345, [-1, 7, 7, 512]);  clone_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_856: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_855, [-1, 49, 512]);  view_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_857: "f32[1568, 512]" = torch.ops.aten.view.default(view_856, [1568, 512]);  view_856 = None
        permute_324: "f32[512, 1536]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_125: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg116_1, view_857, permute_324);  arg116_1 = view_857 = permute_324 = None
        view_858: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_125, [32, 49, 1536]);  addmm_125 = None
        view_859: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_858, [32, 49, 3, 16, -1]);  view_858 = None
        permute_325: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_859, [2, 0, 3, 1, 4]);  view_859 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_31 = torch.ops.aten.unbind.int(permute_325);  permute_325 = None
        getitem_235: "f32[32, 16, 49, 32]" = unbind_31[0]
        getitem_236: "f32[32, 16, 49, 32]" = unbind_31[1]
        getitem_237: "f32[32, 16, 49, 32]" = unbind_31[2];  unbind_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_266: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_235, 0.1767766952966369);  getitem_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_326: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_236, [0, 1, 3, 2]);  getitem_236 = None
        expand_124: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_266, [32, 16, 49, 32]);  mul_266 = None
        clone_346: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
        view_860: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_346, [512, 49, 32]);  clone_346 = None
        expand_125: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_326, [32, 16, 32, 49]);  permute_326 = None
        clone_347: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_861: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_347, [512, 32, 49]);  clone_347 = None
        bmm_62: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_860, view_861);  view_860 = view_861 = None
        view_862: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_62, [32, 16, 49, 49]);  bmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_863: "i64[2401]" = torch.ops.aten.view.default(arg118_1, [-1]);  arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_89: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg117_1, [view_863]);  arg117_1 = view_863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_864: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_89, [49, 49, -1]);  index_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_327: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_864, [2, 0, 1]);  view_864 = None
        clone_348: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_59: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_348, 0);  clone_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_338: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_862, unsqueeze_59);  view_862 = unsqueeze_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_865: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_338, [-1, 4, 16, 49, 49]);  add_338 = None
        unsqueeze_60: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg114_1, 1);  arg114_1 = None
        unsqueeze_61: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 0);  unsqueeze_60 = None
        add_339: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_865, unsqueeze_61);  view_865 = unsqueeze_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_866: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_339, [-1, 16, 49, 49]);  add_339 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_31: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_866, [-1], True)
        sub_102: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_866, amax_31);  view_866 = amax_31 = None
        exp_31: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_102);  sub_102 = None
        sum_32: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_31: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_126: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_31, [32, 16, 49, 49]);  div_31 = None
        view_867: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_126, [512, 49, 49]);  expand_126 = None
        expand_127: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_237, [32, 16, 49, 32]);  getitem_237 = None
        clone_350: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_868: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_350, [512, 49, 32]);  clone_350 = None
        bmm_63: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_867, view_868);  view_867 = view_868 = None
        view_869: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_63, [32, 16, 49, 32]);  bmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_328: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_869, [0, 2, 1, 3]);  view_869 = None
        clone_351: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        view_870: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_351, [32, 49, 512]);  clone_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_871: "f32[1568, 512]" = torch.ops.aten.view.default(view_870, [1568, 512]);  view_870 = None
        permute_329: "f32[512, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_126: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg120_1, view_871, permute_329);  arg120_1 = view_871 = permute_329 = None
        view_872: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_126, [32, 49, 512]);  addmm_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_873: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_872, [-1, 7, 7, 512]);  view_872 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_874: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_873, [-1, 2, 2, 7, 7, 512]);  view_873 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_330: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_874, [0, 1, 3, 2, 4, 5]);  view_874 = None
        clone_353: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_875: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_353, [-1, 14, 14, 512]);  clone_353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_58: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_340: "i64[14]" = torch.ops.aten.add.Tensor(iota_58, 11);  iota_58 = None
        fmod_58: "i64[14]" = torch.ops.aten.fmod.Scalar(add_340, 14);  add_340 = None
        index_90: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_875, [None, fmod_58]);  view_875 = fmod_58 = None
        iota_59: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_341: "i64[14]" = torch.ops.aten.add.Tensor(iota_59, 11);  iota_59 = None
        fmod_59: "i64[14]" = torch.ops.aten.fmod.Scalar(add_341, 14);  add_341 = None
        index_91: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_90, [None, None, fmod_59]);  index_90 = fmod_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_342: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_853, index_91);  view_853 = index_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_876: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_342, [8, -1, 512]);  add_342 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_71 = torch.ops.aten.var_mean.correction(view_876, [2], correction = 0, keepdim = True)
        getitem_238: "f32[8, 196, 1]" = var_mean_71[0]
        getitem_239: "f32[8, 196, 1]" = var_mean_71[1];  var_mean_71 = None
        add_343: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_71: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
        sub_103: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_876, getitem_239);  getitem_239 = None
        mul_267: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_71);  sub_103 = rsqrt_71 = None
        mul_268: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_267, arg121_1);  mul_267 = arg121_1 = None
        add_344: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_268, arg122_1);  mul_268 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_877: "f32[1568, 512]" = torch.ops.aten.view.default(add_344, [1568, 512]);  add_344 = None
        permute_331: "f32[512, 2048]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_127: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg124_1, view_877, permute_331);  arg124_1 = view_877 = permute_331 = None
        view_878: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_127, [8, 196, 2048]);  addmm_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_269: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_878, 0.5)
        mul_270: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_878, 0.7071067811865476);  view_878 = None
        erf_31: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_270);  mul_270 = None
        add_345: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_271: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_269, add_345);  mul_269 = add_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_879: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_271, [1568, 2048]);  mul_271 = None
        permute_332: "f32[2048, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_128: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg126_1, view_879, permute_332);  arg126_1 = view_879 = permute_332 = None
        view_880: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_128, [8, 196, 512]);  addmm_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_346: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_876, view_880);  view_876 = view_880 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_881: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_346, [8, 14, 14, 512]);  add_346 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_72 = torch.ops.aten.var_mean.correction(view_881, [3], correction = 0, keepdim = True)
        getitem_240: "f32[8, 14, 14, 1]" = var_mean_72[0]
        getitem_241: "f32[8, 14, 14, 1]" = var_mean_72[1];  var_mean_72 = None
        add_347: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_72: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_347);  add_347 = None
        sub_104: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_881, getitem_241);  getitem_241 = None
        mul_272: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_72);  sub_104 = rsqrt_72 = None
        mul_273: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_272, arg127_1);  mul_272 = arg127_1 = None
        add_348: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_273, arg128_1);  mul_273 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_882: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_348, [8, 2, 7, 2, 7, 512]);  add_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_333: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_882, [0, 1, 3, 2, 4, 5]);  view_882 = None
        clone_356: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
        view_883: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_356, [-1, 7, 7, 512]);  clone_356 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_884: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_883, [-1, 49, 512]);  view_883 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_885: "f32[1568, 512]" = torch.ops.aten.view.default(view_884, [1568, 512]);  view_884 = None
        permute_334: "f32[512, 1536]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_129: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg130_1, view_885, permute_334);  arg130_1 = view_885 = permute_334 = None
        view_886: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_129, [32, 49, 1536]);  addmm_129 = None
        view_887: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_886, [32, 49, 3, 16, -1]);  view_886 = None
        permute_335: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_887, [2, 0, 3, 1, 4]);  view_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_32 = torch.ops.aten.unbind.int(permute_335);  permute_335 = None
        getitem_242: "f32[32, 16, 49, 32]" = unbind_32[0]
        getitem_243: "f32[32, 16, 49, 32]" = unbind_32[1]
        getitem_244: "f32[32, 16, 49, 32]" = unbind_32[2];  unbind_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_274: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_242, 0.1767766952966369);  getitem_242 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_336: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_243, [0, 1, 3, 2]);  getitem_243 = None
        expand_128: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_274, [32, 16, 49, 32]);  mul_274 = None
        clone_357: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
        view_888: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_357, [512, 49, 32]);  clone_357 = None
        expand_129: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_336, [32, 16, 32, 49]);  permute_336 = None
        clone_358: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_889: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_358, [512, 32, 49]);  clone_358 = None
        bmm_64: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_888, view_889);  view_888 = view_889 = None
        view_890: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_64, [32, 16, 49, 49]);  bmm_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_891: "i64[2401]" = torch.ops.aten.view.default(arg132_1, [-1]);  arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_92: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg131_1, [view_891]);  arg131_1 = view_891 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_892: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_92, [49, 49, -1]);  index_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_337: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_892, [2, 0, 1]);  view_892 = None
        clone_359: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_62: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_359, 0);  clone_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_349: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_890, unsqueeze_62);  view_890 = unsqueeze_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_32: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_349, [-1], True)
        sub_105: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_349, amax_32);  add_349 = amax_32 = None
        exp_32: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_105);  sub_105 = None
        sum_33: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_32: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_130: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_32, [32, 16, 49, 49]);  div_32 = None
        view_893: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_130, [512, 49, 49]);  expand_130 = None
        expand_131: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_244, [32, 16, 49, 32]);  getitem_244 = None
        clone_361: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_894: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_361, [512, 49, 32]);  clone_361 = None
        bmm_65: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_893, view_894);  view_893 = view_894 = None
        view_895: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_65, [32, 16, 49, 32]);  bmm_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_338: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_895, [0, 2, 1, 3]);  view_895 = None
        clone_362: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_338, memory_format = torch.contiguous_format);  permute_338 = None
        view_896: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_362, [32, 49, 512]);  clone_362 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_897: "f32[1568, 512]" = torch.ops.aten.view.default(view_896, [1568, 512]);  view_896 = None
        permute_339: "f32[512, 512]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_130: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg134_1, view_897, permute_339);  arg134_1 = view_897 = permute_339 = None
        view_898: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_130, [32, 49, 512]);  addmm_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_899: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_898, [-1, 7, 7, 512]);  view_898 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_900: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_899, [-1, 2, 2, 7, 7, 512]);  view_899 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_340: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_900, [0, 1, 3, 2, 4, 5]);  view_900 = None
        clone_364: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_340, memory_format = torch.contiguous_format);  permute_340 = None
        view_901: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_364, [-1, 14, 14, 512]);  clone_364 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_350: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_881, view_901);  view_881 = view_901 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_902: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_350, [8, -1, 512]);  add_350 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_73 = torch.ops.aten.var_mean.correction(view_902, [2], correction = 0, keepdim = True)
        getitem_245: "f32[8, 196, 1]" = var_mean_73[0]
        getitem_246: "f32[8, 196, 1]" = var_mean_73[1];  var_mean_73 = None
        add_351: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_245, 1e-05);  getitem_245 = None
        rsqrt_73: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_106: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_902, getitem_246);  getitem_246 = None
        mul_275: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_73);  sub_106 = rsqrt_73 = None
        mul_276: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_275, arg135_1);  mul_275 = arg135_1 = None
        add_352: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_276, arg136_1);  mul_276 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_903: "f32[1568, 512]" = torch.ops.aten.view.default(add_352, [1568, 512]);  add_352 = None
        permute_341: "f32[512, 2048]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_131: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg138_1, view_903, permute_341);  arg138_1 = view_903 = permute_341 = None
        view_904: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_131, [8, 196, 2048]);  addmm_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_277: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_904, 0.5)
        mul_278: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_904, 0.7071067811865476);  view_904 = None
        erf_32: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_278);  mul_278 = None
        add_353: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_279: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_277, add_353);  mul_277 = add_353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_905: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_279, [1568, 2048]);  mul_279 = None
        permute_342: "f32[2048, 512]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_132: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg140_1, view_905, permute_342);  arg140_1 = view_905 = permute_342 = None
        view_906: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_132, [8, 196, 512]);  addmm_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_354: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_902, view_906);  view_902 = view_906 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_907: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_354, [8, 14, 14, 512]);  add_354 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_74 = torch.ops.aten.var_mean.correction(view_907, [3], correction = 0, keepdim = True)
        getitem_247: "f32[8, 14, 14, 1]" = var_mean_74[0]
        getitem_248: "f32[8, 14, 14, 1]" = var_mean_74[1];  var_mean_74 = None
        add_355: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_247, 1e-05);  getitem_247 = None
        rsqrt_74: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
        sub_107: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_907, getitem_248);  getitem_248 = None
        mul_280: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_74);  sub_107 = rsqrt_74 = None
        mul_281: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_280, arg141_1);  mul_280 = arg141_1 = None
        add_356: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_281, arg142_1);  mul_281 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_60: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_357: "i64[14]" = torch.ops.aten.add.Tensor(iota_60, 3);  iota_60 = None
        fmod_60: "i64[14]" = torch.ops.aten.fmod.Scalar(add_357, 14);  add_357 = None
        index_93: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_356, [None, fmod_60]);  add_356 = fmod_60 = None
        iota_61: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_358: "i64[14]" = torch.ops.aten.add.Tensor(iota_61, 3);  iota_61 = None
        fmod_61: "i64[14]" = torch.ops.aten.fmod.Scalar(add_358, 14);  add_358 = None
        index_94: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_93, [None, None, fmod_61]);  index_93 = fmod_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_908: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_94, [8, 2, 7, 2, 7, 512]);  index_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_343: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_908, [0, 1, 3, 2, 4, 5]);  view_908 = None
        clone_367: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
        view_909: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_367, [-1, 7, 7, 512]);  clone_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_910: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_909, [-1, 49, 512]);  view_909 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_911: "f32[1568, 512]" = torch.ops.aten.view.default(view_910, [1568, 512]);  view_910 = None
        permute_344: "f32[512, 1536]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_133: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg145_1, view_911, permute_344);  arg145_1 = view_911 = permute_344 = None
        view_912: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_133, [32, 49, 1536]);  addmm_133 = None
        view_913: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_912, [32, 49, 3, 16, -1]);  view_912 = None
        permute_345: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_913, [2, 0, 3, 1, 4]);  view_913 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_33 = torch.ops.aten.unbind.int(permute_345);  permute_345 = None
        getitem_249: "f32[32, 16, 49, 32]" = unbind_33[0]
        getitem_250: "f32[32, 16, 49, 32]" = unbind_33[1]
        getitem_251: "f32[32, 16, 49, 32]" = unbind_33[2];  unbind_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_282: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_249, 0.1767766952966369);  getitem_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_346: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_250, [0, 1, 3, 2]);  getitem_250 = None
        expand_132: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_282, [32, 16, 49, 32]);  mul_282 = None
        clone_368: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
        view_914: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_368, [512, 49, 32]);  clone_368 = None
        expand_133: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_346, [32, 16, 32, 49]);  permute_346 = None
        clone_369: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_915: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_369, [512, 32, 49]);  clone_369 = None
        bmm_66: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_914, view_915);  view_914 = view_915 = None
        view_916: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_66, [32, 16, 49, 49]);  bmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_917: "i64[2401]" = torch.ops.aten.view.default(arg147_1, [-1]);  arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_95: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg146_1, [view_917]);  arg146_1 = view_917 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_918: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_95, [49, 49, -1]);  index_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_347: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_918, [2, 0, 1]);  view_918 = None
        clone_370: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_63: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_370, 0);  clone_370 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_359: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_916, unsqueeze_63);  view_916 = unsqueeze_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_919: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_359, [-1, 4, 16, 49, 49]);  add_359 = None
        unsqueeze_64: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg143_1, 1);  arg143_1 = None
        unsqueeze_65: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 0);  unsqueeze_64 = None
        add_360: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_919, unsqueeze_65);  view_919 = unsqueeze_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_920: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_360, [-1, 16, 49, 49]);  add_360 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_33: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_920, [-1], True)
        sub_108: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_920, amax_33);  view_920 = amax_33 = None
        exp_33: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_108);  sub_108 = None
        sum_34: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_33: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_134: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_33, [32, 16, 49, 49]);  div_33 = None
        view_921: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_134, [512, 49, 49]);  expand_134 = None
        expand_135: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_251, [32, 16, 49, 32]);  getitem_251 = None
        clone_372: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
        view_922: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_372, [512, 49, 32]);  clone_372 = None
        bmm_67: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_921, view_922);  view_921 = view_922 = None
        view_923: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_67, [32, 16, 49, 32]);  bmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_348: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None
        clone_373: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
        view_924: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_373, [32, 49, 512]);  clone_373 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_925: "f32[1568, 512]" = torch.ops.aten.view.default(view_924, [1568, 512]);  view_924 = None
        permute_349: "f32[512, 512]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_134: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg149_1, view_925, permute_349);  arg149_1 = view_925 = permute_349 = None
        view_926: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_134, [32, 49, 512]);  addmm_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_927: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_926, [-1, 7, 7, 512]);  view_926 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_928: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_927, [-1, 2, 2, 7, 7, 512]);  view_927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_350: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_928, [0, 1, 3, 2, 4, 5]);  view_928 = None
        clone_375: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
        view_929: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_375, [-1, 14, 14, 512]);  clone_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_62: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_361: "i64[14]" = torch.ops.aten.add.Tensor(iota_62, 11);  iota_62 = None
        fmod_62: "i64[14]" = torch.ops.aten.fmod.Scalar(add_361, 14);  add_361 = None
        index_96: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_929, [None, fmod_62]);  view_929 = fmod_62 = None
        iota_63: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_362: "i64[14]" = torch.ops.aten.add.Tensor(iota_63, 11);  iota_63 = None
        fmod_63: "i64[14]" = torch.ops.aten.fmod.Scalar(add_362, 14);  add_362 = None
        index_97: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_96, [None, None, fmod_63]);  index_96 = fmod_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_363: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_907, index_97);  view_907 = index_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_930: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_363, [8, -1, 512]);  add_363 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_75 = torch.ops.aten.var_mean.correction(view_930, [2], correction = 0, keepdim = True)
        getitem_252: "f32[8, 196, 1]" = var_mean_75[0]
        getitem_253: "f32[8, 196, 1]" = var_mean_75[1];  var_mean_75 = None
        add_364: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_252, 1e-05);  getitem_252 = None
        rsqrt_75: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_364);  add_364 = None
        sub_109: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_930, getitem_253);  getitem_253 = None
        mul_283: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_75);  sub_109 = rsqrt_75 = None
        mul_284: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_283, arg150_1);  mul_283 = arg150_1 = None
        add_365: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_284, arg151_1);  mul_284 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_931: "f32[1568, 512]" = torch.ops.aten.view.default(add_365, [1568, 512]);  add_365 = None
        permute_351: "f32[512, 2048]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_135: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg153_1, view_931, permute_351);  arg153_1 = view_931 = permute_351 = None
        view_932: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_135, [8, 196, 2048]);  addmm_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_285: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_932, 0.5)
        mul_286: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_932, 0.7071067811865476);  view_932 = None
        erf_33: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_286);  mul_286 = None
        add_366: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_287: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_285, add_366);  mul_285 = add_366 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_933: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_287, [1568, 2048]);  mul_287 = None
        permute_352: "f32[2048, 512]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_136: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg155_1, view_933, permute_352);  arg155_1 = view_933 = permute_352 = None
        view_934: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_136, [8, 196, 512]);  addmm_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_367: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_930, view_934);  view_930 = view_934 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_935: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_367, [8, 14, 14, 512]);  add_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_76 = torch.ops.aten.var_mean.correction(view_935, [3], correction = 0, keepdim = True)
        getitem_254: "f32[8, 14, 14, 1]" = var_mean_76[0]
        getitem_255: "f32[8, 14, 14, 1]" = var_mean_76[1];  var_mean_76 = None
        add_368: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-05);  getitem_254 = None
        rsqrt_76: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
        sub_110: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_935, getitem_255);  getitem_255 = None
        mul_288: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_76);  sub_110 = rsqrt_76 = None
        mul_289: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_288, arg156_1);  mul_288 = arg156_1 = None
        add_369: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_289, arg157_1);  mul_289 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_936: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_369, [8, 2, 7, 2, 7, 512]);  add_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_353: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_936, [0, 1, 3, 2, 4, 5]);  view_936 = None
        clone_378: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_937: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_378, [-1, 7, 7, 512]);  clone_378 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_938: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_937, [-1, 49, 512]);  view_937 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_939: "f32[1568, 512]" = torch.ops.aten.view.default(view_938, [1568, 512]);  view_938 = None
        permute_354: "f32[512, 1536]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_137: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg159_1, view_939, permute_354);  arg159_1 = view_939 = permute_354 = None
        view_940: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_137, [32, 49, 1536]);  addmm_137 = None
        view_941: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_940, [32, 49, 3, 16, -1]);  view_940 = None
        permute_355: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_941, [2, 0, 3, 1, 4]);  view_941 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_34 = torch.ops.aten.unbind.int(permute_355);  permute_355 = None
        getitem_256: "f32[32, 16, 49, 32]" = unbind_34[0]
        getitem_257: "f32[32, 16, 49, 32]" = unbind_34[1]
        getitem_258: "f32[32, 16, 49, 32]" = unbind_34[2];  unbind_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_290: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_256, 0.1767766952966369);  getitem_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_356: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_257, [0, 1, 3, 2]);  getitem_257 = None
        expand_136: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_290, [32, 16, 49, 32]);  mul_290 = None
        clone_379: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
        view_942: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_379, [512, 49, 32]);  clone_379 = None
        expand_137: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_356, [32, 16, 32, 49]);  permute_356 = None
        clone_380: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_943: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_380, [512, 32, 49]);  clone_380 = None
        bmm_68: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_942, view_943);  view_942 = view_943 = None
        view_944: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_68, [32, 16, 49, 49]);  bmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_945: "i64[2401]" = torch.ops.aten.view.default(arg161_1, [-1]);  arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_98: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg160_1, [view_945]);  arg160_1 = view_945 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_946: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_98, [49, 49, -1]);  index_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_357: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_946, [2, 0, 1]);  view_946 = None
        clone_381: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_66: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_381, 0);  clone_381 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_370: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_944, unsqueeze_66);  view_944 = unsqueeze_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_34: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_370, [-1], True)
        sub_111: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_370, amax_34);  add_370 = amax_34 = None
        exp_34: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_111);  sub_111 = None
        sum_35: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_34: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_138: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_34, [32, 16, 49, 49]);  div_34 = None
        view_947: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_138, [512, 49, 49]);  expand_138 = None
        expand_139: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_258, [32, 16, 49, 32]);  getitem_258 = None
        clone_383: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_948: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_383, [512, 49, 32]);  clone_383 = None
        bmm_69: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_947, view_948);  view_947 = view_948 = None
        view_949: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_69, [32, 16, 49, 32]);  bmm_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_358: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_949, [0, 2, 1, 3]);  view_949 = None
        clone_384: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_950: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_384, [32, 49, 512]);  clone_384 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_951: "f32[1568, 512]" = torch.ops.aten.view.default(view_950, [1568, 512]);  view_950 = None
        permute_359: "f32[512, 512]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_138: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg163_1, view_951, permute_359);  arg163_1 = view_951 = permute_359 = None
        view_952: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_138, [32, 49, 512]);  addmm_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_953: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_952, [-1, 7, 7, 512]);  view_952 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_954: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_953, [-1, 2, 2, 7, 7, 512]);  view_953 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_360: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_954, [0, 1, 3, 2, 4, 5]);  view_954 = None
        clone_386: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
        view_955: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_386, [-1, 14, 14, 512]);  clone_386 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_371: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_935, view_955);  view_935 = view_955 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_956: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_371, [8, -1, 512]);  add_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_77 = torch.ops.aten.var_mean.correction(view_956, [2], correction = 0, keepdim = True)
        getitem_259: "f32[8, 196, 1]" = var_mean_77[0]
        getitem_260: "f32[8, 196, 1]" = var_mean_77[1];  var_mean_77 = None
        add_372: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_259, 1e-05);  getitem_259 = None
        rsqrt_77: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_372);  add_372 = None
        sub_112: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_956, getitem_260);  getitem_260 = None
        mul_291: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_77);  sub_112 = rsqrt_77 = None
        mul_292: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_291, arg164_1);  mul_291 = arg164_1 = None
        add_373: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_292, arg165_1);  mul_292 = arg165_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_957: "f32[1568, 512]" = torch.ops.aten.view.default(add_373, [1568, 512]);  add_373 = None
        permute_361: "f32[512, 2048]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_139: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg167_1, view_957, permute_361);  arg167_1 = view_957 = permute_361 = None
        view_958: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_139, [8, 196, 2048]);  addmm_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_293: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_958, 0.5)
        mul_294: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_958, 0.7071067811865476);  view_958 = None
        erf_34: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
        add_374: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_295: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_293, add_374);  mul_293 = add_374 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_959: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_295, [1568, 2048]);  mul_295 = None
        permute_362: "f32[2048, 512]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_140: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg169_1, view_959, permute_362);  arg169_1 = view_959 = permute_362 = None
        view_960: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_140, [8, 196, 512]);  addmm_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_375: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_956, view_960);  view_956 = view_960 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_961: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_375, [8, 14, 14, 512]);  add_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_78 = torch.ops.aten.var_mean.correction(view_961, [3], correction = 0, keepdim = True)
        getitem_261: "f32[8, 14, 14, 1]" = var_mean_78[0]
        getitem_262: "f32[8, 14, 14, 1]" = var_mean_78[1];  var_mean_78 = None
        add_376: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_261, 1e-05);  getitem_261 = None
        rsqrt_78: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        sub_113: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_961, getitem_262);  getitem_262 = None
        mul_296: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_78);  sub_113 = rsqrt_78 = None
        mul_297: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_296, arg170_1);  mul_296 = arg170_1 = None
        add_377: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_297, arg171_1);  mul_297 = arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_64: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_378: "i64[14]" = torch.ops.aten.add.Tensor(iota_64, 3);  iota_64 = None
        fmod_64: "i64[14]" = torch.ops.aten.fmod.Scalar(add_378, 14);  add_378 = None
        index_99: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_377, [None, fmod_64]);  add_377 = fmod_64 = None
        iota_65: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_379: "i64[14]" = torch.ops.aten.add.Tensor(iota_65, 3);  iota_65 = None
        fmod_65: "i64[14]" = torch.ops.aten.fmod.Scalar(add_379, 14);  add_379 = None
        index_100: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_99, [None, None, fmod_65]);  index_99 = fmod_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_962: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_100, [8, 2, 7, 2, 7, 512]);  index_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_363: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_962, [0, 1, 3, 2, 4, 5]);  view_962 = None
        clone_389: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
        view_963: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_389, [-1, 7, 7, 512]);  clone_389 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_964: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_963, [-1, 49, 512]);  view_963 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_965: "f32[1568, 512]" = torch.ops.aten.view.default(view_964, [1568, 512]);  view_964 = None
        permute_364: "f32[512, 1536]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_141: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg174_1, view_965, permute_364);  arg174_1 = view_965 = permute_364 = None
        view_966: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_141, [32, 49, 1536]);  addmm_141 = None
        view_967: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_966, [32, 49, 3, 16, -1]);  view_966 = None
        permute_365: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_967, [2, 0, 3, 1, 4]);  view_967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_35 = torch.ops.aten.unbind.int(permute_365);  permute_365 = None
        getitem_263: "f32[32, 16, 49, 32]" = unbind_35[0]
        getitem_264: "f32[32, 16, 49, 32]" = unbind_35[1]
        getitem_265: "f32[32, 16, 49, 32]" = unbind_35[2];  unbind_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_298: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_263, 0.1767766952966369);  getitem_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_366: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_264, [0, 1, 3, 2]);  getitem_264 = None
        expand_140: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_298, [32, 16, 49, 32]);  mul_298 = None
        clone_390: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
        view_968: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_390, [512, 49, 32]);  clone_390 = None
        expand_141: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_366, [32, 16, 32, 49]);  permute_366 = None
        clone_391: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
        view_969: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_391, [512, 32, 49]);  clone_391 = None
        bmm_70: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_968, view_969);  view_968 = view_969 = None
        view_970: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_70, [32, 16, 49, 49]);  bmm_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_971: "i64[2401]" = torch.ops.aten.view.default(arg176_1, [-1]);  arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_101: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg175_1, [view_971]);  arg175_1 = view_971 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_972: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_101, [49, 49, -1]);  index_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_367: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_972, [2, 0, 1]);  view_972 = None
        clone_392: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_67: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_392, 0);  clone_392 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_380: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_970, unsqueeze_67);  view_970 = unsqueeze_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_973: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_380, [-1, 4, 16, 49, 49]);  add_380 = None
        unsqueeze_68: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg172_1, 1);  arg172_1 = None
        unsqueeze_69: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, 0);  unsqueeze_68 = None
        add_381: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_973, unsqueeze_69);  view_973 = unsqueeze_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_974: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_381, [-1, 16, 49, 49]);  add_381 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_35: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_974, [-1], True)
        sub_114: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_974, amax_35);  view_974 = amax_35 = None
        exp_35: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_36: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_35: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_142: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_35, [32, 16, 49, 49]);  div_35 = None
        view_975: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_142, [512, 49, 49]);  expand_142 = None
        expand_143: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_265, [32, 16, 49, 32]);  getitem_265 = None
        clone_394: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
        view_976: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_394, [512, 49, 32]);  clone_394 = None
        bmm_71: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_975, view_976);  view_975 = view_976 = None
        view_977: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_71, [32, 16, 49, 32]);  bmm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_368: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_977, [0, 2, 1, 3]);  view_977 = None
        clone_395: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_978: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_395, [32, 49, 512]);  clone_395 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_979: "f32[1568, 512]" = torch.ops.aten.view.default(view_978, [1568, 512]);  view_978 = None
        permute_369: "f32[512, 512]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_142: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg178_1, view_979, permute_369);  arg178_1 = view_979 = permute_369 = None
        view_980: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_142, [32, 49, 512]);  addmm_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_981: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_980, [-1, 7, 7, 512]);  view_980 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_982: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_981, [-1, 2, 2, 7, 7, 512]);  view_981 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_370: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_982, [0, 1, 3, 2, 4, 5]);  view_982 = None
        clone_397: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
        view_983: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_397, [-1, 14, 14, 512]);  clone_397 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_66: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_382: "i64[14]" = torch.ops.aten.add.Tensor(iota_66, 11);  iota_66 = None
        fmod_66: "i64[14]" = torch.ops.aten.fmod.Scalar(add_382, 14);  add_382 = None
        index_102: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_983, [None, fmod_66]);  view_983 = fmod_66 = None
        iota_67: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_383: "i64[14]" = torch.ops.aten.add.Tensor(iota_67, 11);  iota_67 = None
        fmod_67: "i64[14]" = torch.ops.aten.fmod.Scalar(add_383, 14);  add_383 = None
        index_103: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_102, [None, None, fmod_67]);  index_102 = fmod_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_384: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_961, index_103);  view_961 = index_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_984: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_384, [8, -1, 512]);  add_384 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_79 = torch.ops.aten.var_mean.correction(view_984, [2], correction = 0, keepdim = True)
        getitem_266: "f32[8, 196, 1]" = var_mean_79[0]
        getitem_267: "f32[8, 196, 1]" = var_mean_79[1];  var_mean_79 = None
        add_385: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-05);  getitem_266 = None
        rsqrt_79: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_385);  add_385 = None
        sub_115: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_984, getitem_267);  getitem_267 = None
        mul_299: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_79);  sub_115 = rsqrt_79 = None
        mul_300: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_299, arg179_1);  mul_299 = arg179_1 = None
        add_386: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_300, arg180_1);  mul_300 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_985: "f32[1568, 512]" = torch.ops.aten.view.default(add_386, [1568, 512]);  add_386 = None
        permute_371: "f32[512, 2048]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_143: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg182_1, view_985, permute_371);  arg182_1 = view_985 = permute_371 = None
        view_986: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_143, [8, 196, 2048]);  addmm_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_301: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_986, 0.5)
        mul_302: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_986, 0.7071067811865476);  view_986 = None
        erf_35: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_302);  mul_302 = None
        add_387: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_303: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_301, add_387);  mul_301 = add_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_987: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_303, [1568, 2048]);  mul_303 = None
        permute_372: "f32[2048, 512]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_144: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg184_1, view_987, permute_372);  arg184_1 = view_987 = permute_372 = None
        view_988: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_144, [8, 196, 512]);  addmm_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_388: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_984, view_988);  view_984 = view_988 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_989: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_388, [8, 14, 14, 512]);  add_388 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_80 = torch.ops.aten.var_mean.correction(view_989, [3], correction = 0, keepdim = True)
        getitem_268: "f32[8, 14, 14, 1]" = var_mean_80[0]
        getitem_269: "f32[8, 14, 14, 1]" = var_mean_80[1];  var_mean_80 = None
        add_389: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-05);  getitem_268 = None
        rsqrt_80: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_389);  add_389 = None
        sub_116: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_989, getitem_269);  getitem_269 = None
        mul_304: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_80);  sub_116 = rsqrt_80 = None
        mul_305: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_304, arg185_1);  mul_304 = arg185_1 = None
        add_390: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_305, arg186_1);  mul_305 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_990: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_390, [8, 2, 7, 2, 7, 512]);  add_390 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_373: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_990, [0, 1, 3, 2, 4, 5]);  view_990 = None
        clone_400: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
        view_991: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_400, [-1, 7, 7, 512]);  clone_400 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_992: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_991, [-1, 49, 512]);  view_991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_993: "f32[1568, 512]" = torch.ops.aten.view.default(view_992, [1568, 512]);  view_992 = None
        permute_374: "f32[512, 1536]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_145: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg188_1, view_993, permute_374);  arg188_1 = view_993 = permute_374 = None
        view_994: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_145, [32, 49, 1536]);  addmm_145 = None
        view_995: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_994, [32, 49, 3, 16, -1]);  view_994 = None
        permute_375: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_995, [2, 0, 3, 1, 4]);  view_995 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_36 = torch.ops.aten.unbind.int(permute_375);  permute_375 = None
        getitem_270: "f32[32, 16, 49, 32]" = unbind_36[0]
        getitem_271: "f32[32, 16, 49, 32]" = unbind_36[1]
        getitem_272: "f32[32, 16, 49, 32]" = unbind_36[2];  unbind_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_306: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_270, 0.1767766952966369);  getitem_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_376: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_271, [0, 1, 3, 2]);  getitem_271 = None
        expand_144: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_306, [32, 16, 49, 32]);  mul_306 = None
        clone_401: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_144, memory_format = torch.contiguous_format);  expand_144 = None
        view_996: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_401, [512, 49, 32]);  clone_401 = None
        expand_145: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_376, [32, 16, 32, 49]);  permute_376 = None
        clone_402: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_997: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_402, [512, 32, 49]);  clone_402 = None
        bmm_72: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_996, view_997);  view_996 = view_997 = None
        view_998: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_72, [32, 16, 49, 49]);  bmm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_999: "i64[2401]" = torch.ops.aten.view.default(arg190_1, [-1]);  arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_104: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg189_1, [view_999]);  arg189_1 = view_999 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1000: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_104, [49, 49, -1]);  index_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_377: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1000, [2, 0, 1]);  view_1000 = None
        clone_403: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_70: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_403, 0);  clone_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_391: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_998, unsqueeze_70);  view_998 = unsqueeze_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_36: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_391, [-1], True)
        sub_117: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_391, amax_36);  add_391 = amax_36 = None
        exp_36: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_117);  sub_117 = None
        sum_37: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [-1], True)
        div_36: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_36, sum_37);  exp_36 = sum_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_146: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_36, [32, 16, 49, 49]);  div_36 = None
        view_1001: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_146, [512, 49, 49]);  expand_146 = None
        expand_147: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_272, [32, 16, 49, 32]);  getitem_272 = None
        clone_405: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
        view_1002: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_405, [512, 49, 32]);  clone_405 = None
        bmm_73: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1001, view_1002);  view_1001 = view_1002 = None
        view_1003: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_73, [32, 16, 49, 32]);  bmm_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_378: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1003, [0, 2, 1, 3]);  view_1003 = None
        clone_406: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format);  permute_378 = None
        view_1004: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_406, [32, 49, 512]);  clone_406 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1005: "f32[1568, 512]" = torch.ops.aten.view.default(view_1004, [1568, 512]);  view_1004 = None
        permute_379: "f32[512, 512]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_146: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg192_1, view_1005, permute_379);  arg192_1 = view_1005 = permute_379 = None
        view_1006: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_146, [32, 49, 512]);  addmm_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1007: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1006, [-1, 7, 7, 512]);  view_1006 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1008: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1007, [-1, 2, 2, 7, 7, 512]);  view_1007 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_380: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1008, [0, 1, 3, 2, 4, 5]);  view_1008 = None
        clone_408: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_1009: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_408, [-1, 14, 14, 512]);  clone_408 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_392: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_989, view_1009);  view_989 = view_1009 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1010: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_392, [8, -1, 512]);  add_392 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_81 = torch.ops.aten.var_mean.correction(view_1010, [2], correction = 0, keepdim = True)
        getitem_273: "f32[8, 196, 1]" = var_mean_81[0]
        getitem_274: "f32[8, 196, 1]" = var_mean_81[1];  var_mean_81 = None
        add_393: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_273, 1e-05);  getitem_273 = None
        rsqrt_81: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_393);  add_393 = None
        sub_118: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1010, getitem_274);  getitem_274 = None
        mul_307: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_81);  sub_118 = rsqrt_81 = None
        mul_308: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_307, arg193_1);  mul_307 = arg193_1 = None
        add_394: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_308, arg194_1);  mul_308 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1011: "f32[1568, 512]" = torch.ops.aten.view.default(add_394, [1568, 512]);  add_394 = None
        permute_381: "f32[512, 2048]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_147: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg196_1, view_1011, permute_381);  arg196_1 = view_1011 = permute_381 = None
        view_1012: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_147, [8, 196, 2048]);  addmm_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_309: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1012, 0.5)
        mul_310: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1012, 0.7071067811865476);  view_1012 = None
        erf_36: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_310);  mul_310 = None
        add_395: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_311: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_309, add_395);  mul_309 = add_395 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1013: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_311, [1568, 2048]);  mul_311 = None
        permute_382: "f32[2048, 512]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_148: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg198_1, view_1013, permute_382);  arg198_1 = view_1013 = permute_382 = None
        view_1014: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_148, [8, 196, 512]);  addmm_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_396: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1010, view_1014);  view_1010 = view_1014 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1015: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_396, [8, 14, 14, 512]);  add_396 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_82 = torch.ops.aten.var_mean.correction(view_1015, [3], correction = 0, keepdim = True)
        getitem_275: "f32[8, 14, 14, 1]" = var_mean_82[0]
        getitem_276: "f32[8, 14, 14, 1]" = var_mean_82[1];  var_mean_82 = None
        add_397: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_275, 1e-05);  getitem_275 = None
        rsqrt_82: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
        sub_119: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1015, getitem_276);  getitem_276 = None
        mul_312: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_82);  sub_119 = rsqrt_82 = None
        mul_313: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_312, arg199_1);  mul_312 = arg199_1 = None
        add_398: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_313, arg200_1);  mul_313 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_68: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_399: "i64[14]" = torch.ops.aten.add.Tensor(iota_68, 3);  iota_68 = None
        fmod_68: "i64[14]" = torch.ops.aten.fmod.Scalar(add_399, 14);  add_399 = None
        index_105: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_398, [None, fmod_68]);  add_398 = fmod_68 = None
        iota_69: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_400: "i64[14]" = torch.ops.aten.add.Tensor(iota_69, 3);  iota_69 = None
        fmod_69: "i64[14]" = torch.ops.aten.fmod.Scalar(add_400, 14);  add_400 = None
        index_106: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_105, [None, None, fmod_69]);  index_105 = fmod_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1016: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_106, [8, 2, 7, 2, 7, 512]);  index_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_383: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1016, [0, 1, 3, 2, 4, 5]);  view_1016 = None
        clone_411: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
        view_1017: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_411, [-1, 7, 7, 512]);  clone_411 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1018: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1017, [-1, 49, 512]);  view_1017 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1019: "f32[1568, 512]" = torch.ops.aten.view.default(view_1018, [1568, 512]);  view_1018 = None
        permute_384: "f32[512, 1536]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        addmm_149: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg203_1, view_1019, permute_384);  arg203_1 = view_1019 = permute_384 = None
        view_1020: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_149, [32, 49, 1536]);  addmm_149 = None
        view_1021: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1020, [32, 49, 3, 16, -1]);  view_1020 = None
        permute_385: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1021, [2, 0, 3, 1, 4]);  view_1021 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_37 = torch.ops.aten.unbind.int(permute_385);  permute_385 = None
        getitem_277: "f32[32, 16, 49, 32]" = unbind_37[0]
        getitem_278: "f32[32, 16, 49, 32]" = unbind_37[1]
        getitem_279: "f32[32, 16, 49, 32]" = unbind_37[2];  unbind_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_314: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_277, 0.1767766952966369);  getitem_277 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_386: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_278, [0, 1, 3, 2]);  getitem_278 = None
        expand_148: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_314, [32, 16, 49, 32]);  mul_314 = None
        clone_412: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_1022: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_412, [512, 49, 32]);  clone_412 = None
        expand_149: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_386, [32, 16, 32, 49]);  permute_386 = None
        clone_413: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_1023: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_413, [512, 32, 49]);  clone_413 = None
        bmm_74: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1022, view_1023);  view_1022 = view_1023 = None
        view_1024: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_74, [32, 16, 49, 49]);  bmm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1025: "i64[2401]" = torch.ops.aten.view.default(arg205_1, [-1]);  arg205_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_107: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg204_1, [view_1025]);  arg204_1 = view_1025 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1026: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_107, [49, 49, -1]);  index_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_387: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1026, [2, 0, 1]);  view_1026 = None
        clone_414: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_71: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_414, 0);  clone_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_401: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1024, unsqueeze_71);  view_1024 = unsqueeze_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_1027: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_401, [-1, 4, 16, 49, 49]);  add_401 = None
        unsqueeze_72: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg201_1, 1);  arg201_1 = None
        unsqueeze_73: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 0);  unsqueeze_72 = None
        add_402: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1027, unsqueeze_73);  view_1027 = unsqueeze_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_1028: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_402, [-1, 16, 49, 49]);  add_402 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_37: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_1028, [-1], True)
        sub_120: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_1028, amax_37);  view_1028 = amax_37 = None
        exp_37: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_120);  sub_120 = None
        sum_38: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_37: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_37, sum_38);  exp_37 = sum_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_150: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_37, [32, 16, 49, 49]);  div_37 = None
        view_1029: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_150, [512, 49, 49]);  expand_150 = None
        expand_151: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_279, [32, 16, 49, 32]);  getitem_279 = None
        clone_416: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_1030: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_416, [512, 49, 32]);  clone_416 = None
        bmm_75: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1029, view_1030);  view_1029 = view_1030 = None
        view_1031: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_75, [32, 16, 49, 32]);  bmm_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_388: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1031, [0, 2, 1, 3]);  view_1031 = None
        clone_417: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
        view_1032: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_417, [32, 49, 512]);  clone_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1033: "f32[1568, 512]" = torch.ops.aten.view.default(view_1032, [1568, 512]);  view_1032 = None
        permute_389: "f32[512, 512]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_150: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg207_1, view_1033, permute_389);  arg207_1 = view_1033 = permute_389 = None
        view_1034: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_150, [32, 49, 512]);  addmm_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1035: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1034, [-1, 7, 7, 512]);  view_1034 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1036: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1035, [-1, 2, 2, 7, 7, 512]);  view_1035 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_390: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1036, [0, 1, 3, 2, 4, 5]);  view_1036 = None
        clone_419: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
        view_1037: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_419, [-1, 14, 14, 512]);  clone_419 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_70: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_403: "i64[14]" = torch.ops.aten.add.Tensor(iota_70, 11);  iota_70 = None
        fmod_70: "i64[14]" = torch.ops.aten.fmod.Scalar(add_403, 14);  add_403 = None
        index_108: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1037, [None, fmod_70]);  view_1037 = fmod_70 = None
        iota_71: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_404: "i64[14]" = torch.ops.aten.add.Tensor(iota_71, 11);  iota_71 = None
        fmod_71: "i64[14]" = torch.ops.aten.fmod.Scalar(add_404, 14);  add_404 = None
        index_109: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_108, [None, None, fmod_71]);  index_108 = fmod_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_405: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1015, index_109);  view_1015 = index_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1038: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_405, [8, -1, 512]);  add_405 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_83 = torch.ops.aten.var_mean.correction(view_1038, [2], correction = 0, keepdim = True)
        getitem_280: "f32[8, 196, 1]" = var_mean_83[0]
        getitem_281: "f32[8, 196, 1]" = var_mean_83[1];  var_mean_83 = None
        add_406: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_280, 1e-05);  getitem_280 = None
        rsqrt_83: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
        sub_121: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1038, getitem_281);  getitem_281 = None
        mul_315: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_83);  sub_121 = rsqrt_83 = None
        mul_316: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_315, arg208_1);  mul_315 = arg208_1 = None
        add_407: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_316, arg209_1);  mul_316 = arg209_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1039: "f32[1568, 512]" = torch.ops.aten.view.default(add_407, [1568, 512]);  add_407 = None
        permute_391: "f32[512, 2048]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_151: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg211_1, view_1039, permute_391);  arg211_1 = view_1039 = permute_391 = None
        view_1040: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_151, [8, 196, 2048]);  addmm_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_317: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1040, 0.5)
        mul_318: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1040, 0.7071067811865476);  view_1040 = None
        erf_37: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_318);  mul_318 = None
        add_408: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_319: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_317, add_408);  mul_317 = add_408 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1041: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_319, [1568, 2048]);  mul_319 = None
        permute_392: "f32[2048, 512]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
        addmm_152: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg213_1, view_1041, permute_392);  arg213_1 = view_1041 = permute_392 = None
        view_1042: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_152, [8, 196, 512]);  addmm_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_409: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1038, view_1042);  view_1038 = view_1042 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1043: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_409, [8, 14, 14, 512]);  add_409 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_84 = torch.ops.aten.var_mean.correction(view_1043, [3], correction = 0, keepdim = True)
        getitem_282: "f32[8, 14, 14, 1]" = var_mean_84[0]
        getitem_283: "f32[8, 14, 14, 1]" = var_mean_84[1];  var_mean_84 = None
        add_410: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-05);  getitem_282 = None
        rsqrt_84: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_410);  add_410 = None
        sub_122: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1043, getitem_283);  getitem_283 = None
        mul_320: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_84);  sub_122 = rsqrt_84 = None
        mul_321: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_320, arg214_1);  mul_320 = arg214_1 = None
        add_411: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_321, arg215_1);  mul_321 = arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1044: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_411, [8, 2, 7, 2, 7, 512]);  add_411 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_393: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1044, [0, 1, 3, 2, 4, 5]);  view_1044 = None
        clone_422: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
        view_1045: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_422, [-1, 7, 7, 512]);  clone_422 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1046: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1045, [-1, 49, 512]);  view_1045 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1047: "f32[1568, 512]" = torch.ops.aten.view.default(view_1046, [1568, 512]);  view_1046 = None
        permute_394: "f32[512, 1536]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        addmm_153: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg217_1, view_1047, permute_394);  arg217_1 = view_1047 = permute_394 = None
        view_1048: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_153, [32, 49, 1536]);  addmm_153 = None
        view_1049: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1048, [32, 49, 3, 16, -1]);  view_1048 = None
        permute_395: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1049, [2, 0, 3, 1, 4]);  view_1049 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_38 = torch.ops.aten.unbind.int(permute_395);  permute_395 = None
        getitem_284: "f32[32, 16, 49, 32]" = unbind_38[0]
        getitem_285: "f32[32, 16, 49, 32]" = unbind_38[1]
        getitem_286: "f32[32, 16, 49, 32]" = unbind_38[2];  unbind_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_322: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_284, 0.1767766952966369);  getitem_284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_396: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_285, [0, 1, 3, 2]);  getitem_285 = None
        expand_152: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_322, [32, 16, 49, 32]);  mul_322 = None
        clone_423: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
        view_1050: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_423, [512, 49, 32]);  clone_423 = None
        expand_153: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_396, [32, 16, 32, 49]);  permute_396 = None
        clone_424: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_1051: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_424, [512, 32, 49]);  clone_424 = None
        bmm_76: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1050, view_1051);  view_1050 = view_1051 = None
        view_1052: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_76, [32, 16, 49, 49]);  bmm_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1053: "i64[2401]" = torch.ops.aten.view.default(arg219_1, [-1]);  arg219_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_110: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg218_1, [view_1053]);  arg218_1 = view_1053 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1054: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_110, [49, 49, -1]);  index_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_397: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1054, [2, 0, 1]);  view_1054 = None
        clone_425: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_397, memory_format = torch.contiguous_format);  permute_397 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_74: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_425, 0);  clone_425 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_412: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1052, unsqueeze_74);  view_1052 = unsqueeze_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_38: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_412, [-1], True)
        sub_123: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_412, amax_38);  add_412 = amax_38 = None
        exp_38: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_123);  sub_123 = None
        sum_39: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_38: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_38, sum_39);  exp_38 = sum_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_154: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_38, [32, 16, 49, 49]);  div_38 = None
        view_1055: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_154, [512, 49, 49]);  expand_154 = None
        expand_155: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_286, [32, 16, 49, 32]);  getitem_286 = None
        clone_427: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
        view_1056: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_427, [512, 49, 32]);  clone_427 = None
        bmm_77: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1055, view_1056);  view_1055 = view_1056 = None
        view_1057: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_77, [32, 16, 49, 32]);  bmm_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_398: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1057, [0, 2, 1, 3]);  view_1057 = None
        clone_428: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_398, memory_format = torch.contiguous_format);  permute_398 = None
        view_1058: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_428, [32, 49, 512]);  clone_428 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1059: "f32[1568, 512]" = torch.ops.aten.view.default(view_1058, [1568, 512]);  view_1058 = None
        permute_399: "f32[512, 512]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_154: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg221_1, view_1059, permute_399);  arg221_1 = view_1059 = permute_399 = None
        view_1060: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_154, [32, 49, 512]);  addmm_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1061: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1060, [-1, 7, 7, 512]);  view_1060 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1062: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1061, [-1, 2, 2, 7, 7, 512]);  view_1061 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_400: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1062, [0, 1, 3, 2, 4, 5]);  view_1062 = None
        clone_430: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
        view_1063: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_430, [-1, 14, 14, 512]);  clone_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_413: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1043, view_1063);  view_1043 = view_1063 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1064: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_413, [8, -1, 512]);  add_413 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_85 = torch.ops.aten.var_mean.correction(view_1064, [2], correction = 0, keepdim = True)
        getitem_287: "f32[8, 196, 1]" = var_mean_85[0]
        getitem_288: "f32[8, 196, 1]" = var_mean_85[1];  var_mean_85 = None
        add_414: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_287, 1e-05);  getitem_287 = None
        rsqrt_85: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_414);  add_414 = None
        sub_124: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1064, getitem_288);  getitem_288 = None
        mul_323: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_85);  sub_124 = rsqrt_85 = None
        mul_324: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_323, arg222_1);  mul_323 = arg222_1 = None
        add_415: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_324, arg223_1);  mul_324 = arg223_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1065: "f32[1568, 512]" = torch.ops.aten.view.default(add_415, [1568, 512]);  add_415 = None
        permute_401: "f32[512, 2048]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_155: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg225_1, view_1065, permute_401);  arg225_1 = view_1065 = permute_401 = None
        view_1066: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_155, [8, 196, 2048]);  addmm_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_325: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1066, 0.5)
        mul_326: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1066, 0.7071067811865476);  view_1066 = None
        erf_38: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_326);  mul_326 = None
        add_416: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_327: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_325, add_416);  mul_325 = add_416 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1067: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_327, [1568, 2048]);  mul_327 = None
        permute_402: "f32[2048, 512]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_156: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg227_1, view_1067, permute_402);  arg227_1 = view_1067 = permute_402 = None
        view_1068: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_156, [8, 196, 512]);  addmm_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_417: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1064, view_1068);  view_1064 = view_1068 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1069: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_417, [8, 14, 14, 512]);  add_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_86 = torch.ops.aten.var_mean.correction(view_1069, [3], correction = 0, keepdim = True)
        getitem_289: "f32[8, 14, 14, 1]" = var_mean_86[0]
        getitem_290: "f32[8, 14, 14, 1]" = var_mean_86[1];  var_mean_86 = None
        add_418: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_289, 1e-05);  getitem_289 = None
        rsqrt_86: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
        sub_125: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1069, getitem_290);  getitem_290 = None
        mul_328: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_86);  sub_125 = rsqrt_86 = None
        mul_329: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_328, arg228_1);  mul_328 = arg228_1 = None
        add_419: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_329, arg229_1);  mul_329 = arg229_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_72: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_420: "i64[14]" = torch.ops.aten.add.Tensor(iota_72, 3);  iota_72 = None
        fmod_72: "i64[14]" = torch.ops.aten.fmod.Scalar(add_420, 14);  add_420 = None
        index_111: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_419, [None, fmod_72]);  add_419 = fmod_72 = None
        iota_73: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_421: "i64[14]" = torch.ops.aten.add.Tensor(iota_73, 3);  iota_73 = None
        fmod_73: "i64[14]" = torch.ops.aten.fmod.Scalar(add_421, 14);  add_421 = None
        index_112: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_111, [None, None, fmod_73]);  index_111 = fmod_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1070: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_112, [8, 2, 7, 2, 7, 512]);  index_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_403: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1070, [0, 1, 3, 2, 4, 5]);  view_1070 = None
        clone_433: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_403, memory_format = torch.contiguous_format);  permute_403 = None
        view_1071: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_433, [-1, 7, 7, 512]);  clone_433 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1072: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1071, [-1, 49, 512]);  view_1071 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1073: "f32[1568, 512]" = torch.ops.aten.view.default(view_1072, [1568, 512]);  view_1072 = None
        permute_404: "f32[512, 1536]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_157: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg232_1, view_1073, permute_404);  arg232_1 = view_1073 = permute_404 = None
        view_1074: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_157, [32, 49, 1536]);  addmm_157 = None
        view_1075: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1074, [32, 49, 3, 16, -1]);  view_1074 = None
        permute_405: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1075, [2, 0, 3, 1, 4]);  view_1075 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_39 = torch.ops.aten.unbind.int(permute_405);  permute_405 = None
        getitem_291: "f32[32, 16, 49, 32]" = unbind_39[0]
        getitem_292: "f32[32, 16, 49, 32]" = unbind_39[1]
        getitem_293: "f32[32, 16, 49, 32]" = unbind_39[2];  unbind_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_330: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_291, 0.1767766952966369);  getitem_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_406: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_292, [0, 1, 3, 2]);  getitem_292 = None
        expand_156: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_330, [32, 16, 49, 32]);  mul_330 = None
        clone_434: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_1076: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_434, [512, 49, 32]);  clone_434 = None
        expand_157: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_406, [32, 16, 32, 49]);  permute_406 = None
        clone_435: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_1077: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_435, [512, 32, 49]);  clone_435 = None
        bmm_78: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1076, view_1077);  view_1076 = view_1077 = None
        view_1078: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_78, [32, 16, 49, 49]);  bmm_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1079: "i64[2401]" = torch.ops.aten.view.default(arg234_1, [-1]);  arg234_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_113: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg233_1, [view_1079]);  arg233_1 = view_1079 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1080: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_113, [49, 49, -1]);  index_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_407: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1080, [2, 0, 1]);  view_1080 = None
        clone_436: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_75: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_436, 0);  clone_436 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_422: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1078, unsqueeze_75);  view_1078 = unsqueeze_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_1081: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_422, [-1, 4, 16, 49, 49]);  add_422 = None
        unsqueeze_76: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg230_1, 1);  arg230_1 = None
        unsqueeze_77: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 0);  unsqueeze_76 = None
        add_423: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1081, unsqueeze_77);  view_1081 = unsqueeze_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_1082: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_423, [-1, 16, 49, 49]);  add_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_39: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_1082, [-1], True)
        sub_126: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_1082, amax_39);  view_1082 = amax_39 = None
        exp_39: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_126);  sub_126 = None
        sum_40: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_39, [-1], True)
        div_39: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_39, sum_40);  exp_39 = sum_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_158: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_39, [32, 16, 49, 49]);  div_39 = None
        view_1083: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_158, [512, 49, 49]);  expand_158 = None
        expand_159: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_293, [32, 16, 49, 32]);  getitem_293 = None
        clone_438: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_159, memory_format = torch.contiguous_format);  expand_159 = None
        view_1084: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_438, [512, 49, 32]);  clone_438 = None
        bmm_79: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1083, view_1084);  view_1083 = view_1084 = None
        view_1085: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_79, [32, 16, 49, 32]);  bmm_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_408: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1085, [0, 2, 1, 3]);  view_1085 = None
        clone_439: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_408, memory_format = torch.contiguous_format);  permute_408 = None
        view_1086: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_439, [32, 49, 512]);  clone_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1087: "f32[1568, 512]" = torch.ops.aten.view.default(view_1086, [1568, 512]);  view_1086 = None
        permute_409: "f32[512, 512]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_158: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg236_1, view_1087, permute_409);  arg236_1 = view_1087 = permute_409 = None
        view_1088: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_158, [32, 49, 512]);  addmm_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1089: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1088, [-1, 7, 7, 512]);  view_1088 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1090: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1089, [-1, 2, 2, 7, 7, 512]);  view_1089 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_410: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1090, [0, 1, 3, 2, 4, 5]);  view_1090 = None
        clone_441: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
        view_1091: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_441, [-1, 14, 14, 512]);  clone_441 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_74: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_424: "i64[14]" = torch.ops.aten.add.Tensor(iota_74, 11);  iota_74 = None
        fmod_74: "i64[14]" = torch.ops.aten.fmod.Scalar(add_424, 14);  add_424 = None
        index_114: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1091, [None, fmod_74]);  view_1091 = fmod_74 = None
        iota_75: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_425: "i64[14]" = torch.ops.aten.add.Tensor(iota_75, 11);  iota_75 = None
        fmod_75: "i64[14]" = torch.ops.aten.fmod.Scalar(add_425, 14);  add_425 = None
        index_115: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_114, [None, None, fmod_75]);  index_114 = fmod_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_426: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1069, index_115);  view_1069 = index_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1092: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_426, [8, -1, 512]);  add_426 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_87 = torch.ops.aten.var_mean.correction(view_1092, [2], correction = 0, keepdim = True)
        getitem_294: "f32[8, 196, 1]" = var_mean_87[0]
        getitem_295: "f32[8, 196, 1]" = var_mean_87[1];  var_mean_87 = None
        add_427: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_294, 1e-05);  getitem_294 = None
        rsqrt_87: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_427);  add_427 = None
        sub_127: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1092, getitem_295);  getitem_295 = None
        mul_331: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_87);  sub_127 = rsqrt_87 = None
        mul_332: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_331, arg237_1);  mul_331 = arg237_1 = None
        add_428: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_332, arg238_1);  mul_332 = arg238_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1093: "f32[1568, 512]" = torch.ops.aten.view.default(add_428, [1568, 512]);  add_428 = None
        permute_411: "f32[512, 2048]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_159: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg240_1, view_1093, permute_411);  arg240_1 = view_1093 = permute_411 = None
        view_1094: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_159, [8, 196, 2048]);  addmm_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_333: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1094, 0.5)
        mul_334: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1094, 0.7071067811865476);  view_1094 = None
        erf_39: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_334);  mul_334 = None
        add_429: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_335: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_333, add_429);  mul_333 = add_429 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1095: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_335, [1568, 2048]);  mul_335 = None
        permute_412: "f32[2048, 512]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_160: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg242_1, view_1095, permute_412);  arg242_1 = view_1095 = permute_412 = None
        view_1096: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_160, [8, 196, 512]);  addmm_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_430: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1092, view_1096);  view_1092 = view_1096 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1097: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_430, [8, 14, 14, 512]);  add_430 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_88 = torch.ops.aten.var_mean.correction(view_1097, [3], correction = 0, keepdim = True)
        getitem_296: "f32[8, 14, 14, 1]" = var_mean_88[0]
        getitem_297: "f32[8, 14, 14, 1]" = var_mean_88[1];  var_mean_88 = None
        add_431: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-05);  getitem_296 = None
        rsqrt_88: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
        sub_128: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1097, getitem_297);  getitem_297 = None
        mul_336: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_88);  sub_128 = rsqrt_88 = None
        mul_337: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_336, arg243_1);  mul_336 = arg243_1 = None
        add_432: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_337, arg244_1);  mul_337 = arg244_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1098: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_432, [8, 2, 7, 2, 7, 512]);  add_432 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_413: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1098, [0, 1, 3, 2, 4, 5]);  view_1098 = None
        clone_444: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_1099: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_444, [-1, 7, 7, 512]);  clone_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1100: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1099, [-1, 49, 512]);  view_1099 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1101: "f32[1568, 512]" = torch.ops.aten.view.default(view_1100, [1568, 512]);  view_1100 = None
        permute_414: "f32[512, 1536]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_161: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg246_1, view_1101, permute_414);  arg246_1 = view_1101 = permute_414 = None
        view_1102: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_161, [32, 49, 1536]);  addmm_161 = None
        view_1103: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1102, [32, 49, 3, 16, -1]);  view_1102 = None
        permute_415: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1103, [2, 0, 3, 1, 4]);  view_1103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_40 = torch.ops.aten.unbind.int(permute_415);  permute_415 = None
        getitem_298: "f32[32, 16, 49, 32]" = unbind_40[0]
        getitem_299: "f32[32, 16, 49, 32]" = unbind_40[1]
        getitem_300: "f32[32, 16, 49, 32]" = unbind_40[2];  unbind_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_338: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_298, 0.1767766952966369);  getitem_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_416: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_299, [0, 1, 3, 2]);  getitem_299 = None
        expand_160: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_338, [32, 16, 49, 32]);  mul_338 = None
        clone_445: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
        view_1104: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_445, [512, 49, 32]);  clone_445 = None
        expand_161: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_416, [32, 16, 32, 49]);  permute_416 = None
        clone_446: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_1105: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_446, [512, 32, 49]);  clone_446 = None
        bmm_80: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1104, view_1105);  view_1104 = view_1105 = None
        view_1106: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_80, [32, 16, 49, 49]);  bmm_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1107: "i64[2401]" = torch.ops.aten.view.default(arg248_1, [-1]);  arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_116: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg247_1, [view_1107]);  arg247_1 = view_1107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1108: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_116, [49, 49, -1]);  index_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_417: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1108, [2, 0, 1]);  view_1108 = None
        clone_447: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_78: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_447, 0);  clone_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_433: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1106, unsqueeze_78);  view_1106 = unsqueeze_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_40: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_433, [-1], True)
        sub_129: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_433, amax_40);  add_433 = amax_40 = None
        exp_40: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_129);  sub_129 = None
        sum_41: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_40, [-1], True)
        div_40: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_40, sum_41);  exp_40 = sum_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_162: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_40, [32, 16, 49, 49]);  div_40 = None
        view_1109: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_162, [512, 49, 49]);  expand_162 = None
        expand_163: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_300, [32, 16, 49, 32]);  getitem_300 = None
        clone_449: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_1110: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_449, [512, 49, 32]);  clone_449 = None
        bmm_81: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1109, view_1110);  view_1109 = view_1110 = None
        view_1111: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_81, [32, 16, 49, 32]);  bmm_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_418: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1111, [0, 2, 1, 3]);  view_1111 = None
        clone_450: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_1112: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_450, [32, 49, 512]);  clone_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1113: "f32[1568, 512]" = torch.ops.aten.view.default(view_1112, [1568, 512]);  view_1112 = None
        permute_419: "f32[512, 512]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_162: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg250_1, view_1113, permute_419);  arg250_1 = view_1113 = permute_419 = None
        view_1114: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_162, [32, 49, 512]);  addmm_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1115: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1114, [-1, 7, 7, 512]);  view_1114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1116: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1115, [-1, 2, 2, 7, 7, 512]);  view_1115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_420: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1116, [0, 1, 3, 2, 4, 5]);  view_1116 = None
        clone_452: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
        view_1117: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_452, [-1, 14, 14, 512]);  clone_452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_434: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1097, view_1117);  view_1097 = view_1117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1118: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_434, [8, -1, 512]);  add_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_89 = torch.ops.aten.var_mean.correction(view_1118, [2], correction = 0, keepdim = True)
        getitem_301: "f32[8, 196, 1]" = var_mean_89[0]
        getitem_302: "f32[8, 196, 1]" = var_mean_89[1];  var_mean_89 = None
        add_435: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_301, 1e-05);  getitem_301 = None
        rsqrt_89: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_435);  add_435 = None
        sub_130: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1118, getitem_302);  getitem_302 = None
        mul_339: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_89);  sub_130 = rsqrt_89 = None
        mul_340: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_339, arg251_1);  mul_339 = arg251_1 = None
        add_436: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_340, arg252_1);  mul_340 = arg252_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1119: "f32[1568, 512]" = torch.ops.aten.view.default(add_436, [1568, 512]);  add_436 = None
        permute_421: "f32[512, 2048]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_163: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg254_1, view_1119, permute_421);  arg254_1 = view_1119 = permute_421 = None
        view_1120: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_163, [8, 196, 2048]);  addmm_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_341: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1120, 0.5)
        mul_342: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1120, 0.7071067811865476);  view_1120 = None
        erf_40: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_342);  mul_342 = None
        add_437: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_343: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_341, add_437);  mul_341 = add_437 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1121: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_343, [1568, 2048]);  mul_343 = None
        permute_422: "f32[2048, 512]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_164: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg256_1, view_1121, permute_422);  arg256_1 = view_1121 = permute_422 = None
        view_1122: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_164, [8, 196, 512]);  addmm_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_438: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1118, view_1122);  view_1118 = view_1122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1123: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_438, [8, 14, 14, 512]);  add_438 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_90 = torch.ops.aten.var_mean.correction(view_1123, [3], correction = 0, keepdim = True)
        getitem_303: "f32[8, 14, 14, 1]" = var_mean_90[0]
        getitem_304: "f32[8, 14, 14, 1]" = var_mean_90[1];  var_mean_90 = None
        add_439: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_303, 1e-05);  getitem_303 = None
        rsqrt_90: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_439);  add_439 = None
        sub_131: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1123, getitem_304);  getitem_304 = None
        mul_344: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_90);  sub_131 = rsqrt_90 = None
        mul_345: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_344, arg257_1);  mul_344 = arg257_1 = None
        add_440: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_345, arg258_1);  mul_345 = arg258_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_76: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_441: "i64[14]" = torch.ops.aten.add.Tensor(iota_76, 3);  iota_76 = None
        fmod_76: "i64[14]" = torch.ops.aten.fmod.Scalar(add_441, 14);  add_441 = None
        index_117: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_440, [None, fmod_76]);  add_440 = fmod_76 = None
        iota_77: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_442: "i64[14]" = torch.ops.aten.add.Tensor(iota_77, 3);  iota_77 = None
        fmod_77: "i64[14]" = torch.ops.aten.fmod.Scalar(add_442, 14);  add_442 = None
        index_118: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_117, [None, None, fmod_77]);  index_117 = fmod_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1124: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_118, [8, 2, 7, 2, 7, 512]);  index_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_423: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1124, [0, 1, 3, 2, 4, 5]);  view_1124 = None
        clone_455: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
        view_1125: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_455, [-1, 7, 7, 512]);  clone_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1126: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1125, [-1, 49, 512]);  view_1125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1127: "f32[1568, 512]" = torch.ops.aten.view.default(view_1126, [1568, 512]);  view_1126 = None
        permute_424: "f32[512, 1536]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_165: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg261_1, view_1127, permute_424);  arg261_1 = view_1127 = permute_424 = None
        view_1128: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_165, [32, 49, 1536]);  addmm_165 = None
        view_1129: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1128, [32, 49, 3, 16, -1]);  view_1128 = None
        permute_425: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1129, [2, 0, 3, 1, 4]);  view_1129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_41 = torch.ops.aten.unbind.int(permute_425);  permute_425 = None
        getitem_305: "f32[32, 16, 49, 32]" = unbind_41[0]
        getitem_306: "f32[32, 16, 49, 32]" = unbind_41[1]
        getitem_307: "f32[32, 16, 49, 32]" = unbind_41[2];  unbind_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_346: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_305, 0.1767766952966369);  getitem_305 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_426: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_306, [0, 1, 3, 2]);  getitem_306 = None
        expand_164: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_346, [32, 16, 49, 32]);  mul_346 = None
        clone_456: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_1130: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_456, [512, 49, 32]);  clone_456 = None
        expand_165: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_426, [32, 16, 32, 49]);  permute_426 = None
        clone_457: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_1131: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_457, [512, 32, 49]);  clone_457 = None
        bmm_82: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1130, view_1131);  view_1130 = view_1131 = None
        view_1132: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_82, [32, 16, 49, 49]);  bmm_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1133: "i64[2401]" = torch.ops.aten.view.default(arg263_1, [-1]);  arg263_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_119: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg262_1, [view_1133]);  arg262_1 = view_1133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1134: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_119, [49, 49, -1]);  index_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_427: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1134, [2, 0, 1]);  view_1134 = None
        clone_458: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_79: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_458, 0);  clone_458 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_443: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1132, unsqueeze_79);  view_1132 = unsqueeze_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_1135: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_443, [-1, 4, 16, 49, 49]);  add_443 = None
        unsqueeze_80: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg259_1, 1);  arg259_1 = None
        unsqueeze_81: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 0);  unsqueeze_80 = None
        add_444: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1135, unsqueeze_81);  view_1135 = unsqueeze_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_1136: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_444, [-1, 16, 49, 49]);  add_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_41: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_1136, [-1], True)
        sub_132: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_1136, amax_41);  view_1136 = amax_41 = None
        exp_41: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_132);  sub_132 = None
        sum_42: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_41, [-1], True)
        div_41: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_41, sum_42);  exp_41 = sum_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_166: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_41, [32, 16, 49, 49]);  div_41 = None
        view_1137: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_166, [512, 49, 49]);  expand_166 = None
        expand_167: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_307, [32, 16, 49, 32]);  getitem_307 = None
        clone_460: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_167, memory_format = torch.contiguous_format);  expand_167 = None
        view_1138: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_460, [512, 49, 32]);  clone_460 = None
        bmm_83: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1137, view_1138);  view_1137 = view_1138 = None
        view_1139: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_83, [32, 16, 49, 32]);  bmm_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_428: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1139, [0, 2, 1, 3]);  view_1139 = None
        clone_461: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_428, memory_format = torch.contiguous_format);  permute_428 = None
        view_1140: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_461, [32, 49, 512]);  clone_461 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1141: "f32[1568, 512]" = torch.ops.aten.view.default(view_1140, [1568, 512]);  view_1140 = None
        permute_429: "f32[512, 512]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_166: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg265_1, view_1141, permute_429);  arg265_1 = view_1141 = permute_429 = None
        view_1142: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_166, [32, 49, 512]);  addmm_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1143: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1142, [-1, 7, 7, 512]);  view_1142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1144: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1143, [-1, 2, 2, 7, 7, 512]);  view_1143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_430: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1144, [0, 1, 3, 2, 4, 5]);  view_1144 = None
        clone_463: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_430, memory_format = torch.contiguous_format);  permute_430 = None
        view_1145: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_463, [-1, 14, 14, 512]);  clone_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_78: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_445: "i64[14]" = torch.ops.aten.add.Tensor(iota_78, 11);  iota_78 = None
        fmod_78: "i64[14]" = torch.ops.aten.fmod.Scalar(add_445, 14);  add_445 = None
        index_120: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1145, [None, fmod_78]);  view_1145 = fmod_78 = None
        iota_79: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_446: "i64[14]" = torch.ops.aten.add.Tensor(iota_79, 11);  iota_79 = None
        fmod_79: "i64[14]" = torch.ops.aten.fmod.Scalar(add_446, 14);  add_446 = None
        index_121: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_120, [None, None, fmod_79]);  index_120 = fmod_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_447: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1123, index_121);  view_1123 = index_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1146: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_447, [8, -1, 512]);  add_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_91 = torch.ops.aten.var_mean.correction(view_1146, [2], correction = 0, keepdim = True)
        getitem_308: "f32[8, 196, 1]" = var_mean_91[0]
        getitem_309: "f32[8, 196, 1]" = var_mean_91[1];  var_mean_91 = None
        add_448: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-05);  getitem_308 = None
        rsqrt_91: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_448);  add_448 = None
        sub_133: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1146, getitem_309);  getitem_309 = None
        mul_347: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_91);  sub_133 = rsqrt_91 = None
        mul_348: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_347, arg266_1);  mul_347 = arg266_1 = None
        add_449: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_348, arg267_1);  mul_348 = arg267_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1147: "f32[1568, 512]" = torch.ops.aten.view.default(add_449, [1568, 512]);  add_449 = None
        permute_431: "f32[512, 2048]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_167: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg269_1, view_1147, permute_431);  arg269_1 = view_1147 = permute_431 = None
        view_1148: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_167, [8, 196, 2048]);  addmm_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_349: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1148, 0.5)
        mul_350: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1148, 0.7071067811865476);  view_1148 = None
        erf_41: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_450: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_351: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_349, add_450);  mul_349 = add_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1149: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_351, [1568, 2048]);  mul_351 = None
        permute_432: "f32[2048, 512]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
        addmm_168: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg271_1, view_1149, permute_432);  arg271_1 = view_1149 = permute_432 = None
        view_1150: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_168, [8, 196, 512]);  addmm_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_451: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1146, view_1150);  view_1146 = view_1150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1151: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_451, [8, 14, 14, 512]);  add_451 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_92 = torch.ops.aten.var_mean.correction(view_1151, [3], correction = 0, keepdim = True)
        getitem_310: "f32[8, 14, 14, 1]" = var_mean_92[0]
        getitem_311: "f32[8, 14, 14, 1]" = var_mean_92[1];  var_mean_92 = None
        add_452: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-05);  getitem_310 = None
        rsqrt_92: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_452);  add_452 = None
        sub_134: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1151, getitem_311);  getitem_311 = None
        mul_352: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_92);  sub_134 = rsqrt_92 = None
        mul_353: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_352, arg272_1);  mul_352 = arg272_1 = None
        add_453: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_353, arg273_1);  mul_353 = arg273_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1152: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_453, [8, 2, 7, 2, 7, 512]);  add_453 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_433: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1152, [0, 1, 3, 2, 4, 5]);  view_1152 = None
        clone_466: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
        view_1153: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_466, [-1, 7, 7, 512]);  clone_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1154: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1153, [-1, 49, 512]);  view_1153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1155: "f32[1568, 512]" = torch.ops.aten.view.default(view_1154, [1568, 512]);  view_1154 = None
        permute_434: "f32[512, 1536]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        addmm_169: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg275_1, view_1155, permute_434);  arg275_1 = view_1155 = permute_434 = None
        view_1156: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_169, [32, 49, 1536]);  addmm_169 = None
        view_1157: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1156, [32, 49, 3, 16, -1]);  view_1156 = None
        permute_435: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1157, [2, 0, 3, 1, 4]);  view_1157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_42 = torch.ops.aten.unbind.int(permute_435);  permute_435 = None
        getitem_312: "f32[32, 16, 49, 32]" = unbind_42[0]
        getitem_313: "f32[32, 16, 49, 32]" = unbind_42[1]
        getitem_314: "f32[32, 16, 49, 32]" = unbind_42[2];  unbind_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_354: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_312, 0.1767766952966369);  getitem_312 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_436: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_313, [0, 1, 3, 2]);  getitem_313 = None
        expand_168: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_354, [32, 16, 49, 32]);  mul_354 = None
        clone_467: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
        view_1158: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_467, [512, 49, 32]);  clone_467 = None
        expand_169: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_436, [32, 16, 32, 49]);  permute_436 = None
        clone_468: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_1159: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_468, [512, 32, 49]);  clone_468 = None
        bmm_84: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1158, view_1159);  view_1158 = view_1159 = None
        view_1160: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_84, [32, 16, 49, 49]);  bmm_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1161: "i64[2401]" = torch.ops.aten.view.default(arg277_1, [-1]);  arg277_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_122: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg276_1, [view_1161]);  arg276_1 = view_1161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1162: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_122, [49, 49, -1]);  index_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_437: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1162, [2, 0, 1]);  view_1162 = None
        clone_469: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_82: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_469, 0);  clone_469 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_454: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1160, unsqueeze_82);  view_1160 = unsqueeze_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_42: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_454, [-1], True)
        sub_135: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_454, amax_42);  add_454 = amax_42 = None
        exp_42: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        sum_43: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_42, [-1], True)
        div_42: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_42, sum_43);  exp_42 = sum_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_170: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_42, [32, 16, 49, 49]);  div_42 = None
        view_1163: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_170, [512, 49, 49]);  expand_170 = None
        expand_171: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_314, [32, 16, 49, 32]);  getitem_314 = None
        clone_471: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
        view_1164: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_471, [512, 49, 32]);  clone_471 = None
        bmm_85: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1163, view_1164);  view_1163 = view_1164 = None
        view_1165: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_85, [32, 16, 49, 32]);  bmm_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_438: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1165, [0, 2, 1, 3]);  view_1165 = None
        clone_472: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_438, memory_format = torch.contiguous_format);  permute_438 = None
        view_1166: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_472, [32, 49, 512]);  clone_472 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1167: "f32[1568, 512]" = torch.ops.aten.view.default(view_1166, [1568, 512]);  view_1166 = None
        permute_439: "f32[512, 512]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_170: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg279_1, view_1167, permute_439);  arg279_1 = view_1167 = permute_439 = None
        view_1168: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_170, [32, 49, 512]);  addmm_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1169: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1168, [-1, 7, 7, 512]);  view_1168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1170: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1169, [-1, 2, 2, 7, 7, 512]);  view_1169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_440: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1170, [0, 1, 3, 2, 4, 5]);  view_1170 = None
        clone_474: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_440, memory_format = torch.contiguous_format);  permute_440 = None
        view_1171: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_474, [-1, 14, 14, 512]);  clone_474 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_455: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1151, view_1171);  view_1151 = view_1171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1172: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_455, [8, -1, 512]);  add_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_93 = torch.ops.aten.var_mean.correction(view_1172, [2], correction = 0, keepdim = True)
        getitem_315: "f32[8, 196, 1]" = var_mean_93[0]
        getitem_316: "f32[8, 196, 1]" = var_mean_93[1];  var_mean_93 = None
        add_456: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_315, 1e-05);  getitem_315 = None
        rsqrt_93: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_456);  add_456 = None
        sub_136: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1172, getitem_316);  getitem_316 = None
        mul_355: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_93);  sub_136 = rsqrt_93 = None
        mul_356: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_355, arg280_1);  mul_355 = arg280_1 = None
        add_457: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_356, arg281_1);  mul_356 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1173: "f32[1568, 512]" = torch.ops.aten.view.default(add_457, [1568, 512]);  add_457 = None
        permute_441: "f32[512, 2048]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_171: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg283_1, view_1173, permute_441);  arg283_1 = view_1173 = permute_441 = None
        view_1174: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_171, [8, 196, 2048]);  addmm_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_357: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1174, 0.5)
        mul_358: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1174, 0.7071067811865476);  view_1174 = None
        erf_42: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_358);  mul_358 = None
        add_458: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_359: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_357, add_458);  mul_357 = add_458 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1175: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_359, [1568, 2048]);  mul_359 = None
        permute_442: "f32[2048, 512]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        addmm_172: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg285_1, view_1175, permute_442);  arg285_1 = view_1175 = permute_442 = None
        view_1176: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_172, [8, 196, 512]);  addmm_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_459: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1172, view_1176);  view_1172 = view_1176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1177: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_459, [8, 14, 14, 512]);  add_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_94 = torch.ops.aten.var_mean.correction(view_1177, [3], correction = 0, keepdim = True)
        getitem_317: "f32[8, 14, 14, 1]" = var_mean_94[0]
        getitem_318: "f32[8, 14, 14, 1]" = var_mean_94[1];  var_mean_94 = None
        add_460: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_317, 1e-05);  getitem_317 = None
        rsqrt_94: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_460);  add_460 = None
        sub_137: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1177, getitem_318);  getitem_318 = None
        mul_360: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_94);  sub_137 = rsqrt_94 = None
        mul_361: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_360, arg286_1);  mul_360 = arg286_1 = None
        add_461: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_361, arg287_1);  mul_361 = arg287_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_80: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_462: "i64[14]" = torch.ops.aten.add.Tensor(iota_80, 3);  iota_80 = None
        fmod_80: "i64[14]" = torch.ops.aten.fmod.Scalar(add_462, 14);  add_462 = None
        index_123: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_461, [None, fmod_80]);  add_461 = fmod_80 = None
        iota_81: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_463: "i64[14]" = torch.ops.aten.add.Tensor(iota_81, 3);  iota_81 = None
        fmod_81: "i64[14]" = torch.ops.aten.fmod.Scalar(add_463, 14);  add_463 = None
        index_124: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_123, [None, None, fmod_81]);  index_123 = fmod_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1178: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_124, [8, 2, 7, 2, 7, 512]);  index_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_443: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1178, [0, 1, 3, 2, 4, 5]);  view_1178 = None
        clone_477: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
        view_1179: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_477, [-1, 7, 7, 512]);  clone_477 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1180: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1179, [-1, 49, 512]);  view_1179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1181: "f32[1568, 512]" = torch.ops.aten.view.default(view_1180, [1568, 512]);  view_1180 = None
        permute_444: "f32[512, 1536]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_173: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg290_1, view_1181, permute_444);  arg290_1 = view_1181 = permute_444 = None
        view_1182: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_173, [32, 49, 1536]);  addmm_173 = None
        view_1183: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1182, [32, 49, 3, 16, -1]);  view_1182 = None
        permute_445: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1183, [2, 0, 3, 1, 4]);  view_1183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_43 = torch.ops.aten.unbind.int(permute_445);  permute_445 = None
        getitem_319: "f32[32, 16, 49, 32]" = unbind_43[0]
        getitem_320: "f32[32, 16, 49, 32]" = unbind_43[1]
        getitem_321: "f32[32, 16, 49, 32]" = unbind_43[2];  unbind_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_362: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_319, 0.1767766952966369);  getitem_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_446: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_320, [0, 1, 3, 2]);  getitem_320 = None
        expand_172: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_362, [32, 16, 49, 32]);  mul_362 = None
        clone_478: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_1184: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_478, [512, 49, 32]);  clone_478 = None
        expand_173: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_446, [32, 16, 32, 49]);  permute_446 = None
        clone_479: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_1185: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_479, [512, 32, 49]);  clone_479 = None
        bmm_86: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1184, view_1185);  view_1184 = view_1185 = None
        view_1186: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_86, [32, 16, 49, 49]);  bmm_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1187: "i64[2401]" = torch.ops.aten.view.default(arg292_1, [-1]);  arg292_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_125: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg291_1, [view_1187]);  arg291_1 = view_1187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1188: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_125, [49, 49, -1]);  index_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_447: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1188, [2, 0, 1]);  view_1188 = None
        clone_480: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_83: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_480, 0);  clone_480 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_464: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1186, unsqueeze_83);  view_1186 = unsqueeze_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_1189: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_464, [-1, 4, 16, 49, 49]);  add_464 = None
        unsqueeze_84: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg288_1, 1);  arg288_1 = None
        unsqueeze_85: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 0);  unsqueeze_84 = None
        add_465: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1189, unsqueeze_85);  view_1189 = unsqueeze_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_1190: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_465, [-1, 16, 49, 49]);  add_465 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_43: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_1190, [-1], True)
        sub_138: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_1190, amax_43);  view_1190 = amax_43 = None
        exp_43: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_138);  sub_138 = None
        sum_44: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_43, [-1], True)
        div_43: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_43, sum_44);  exp_43 = sum_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_174: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_43, [32, 16, 49, 49]);  div_43 = None
        view_1191: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_174, [512, 49, 49]);  expand_174 = None
        expand_175: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_321, [32, 16, 49, 32]);  getitem_321 = None
        clone_482: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_175, memory_format = torch.contiguous_format);  expand_175 = None
        view_1192: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_482, [512, 49, 32]);  clone_482 = None
        bmm_87: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1191, view_1192);  view_1191 = view_1192 = None
        view_1193: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_87, [32, 16, 49, 32]);  bmm_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_448: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1193, [0, 2, 1, 3]);  view_1193 = None
        clone_483: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
        view_1194: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_483, [32, 49, 512]);  clone_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1195: "f32[1568, 512]" = torch.ops.aten.view.default(view_1194, [1568, 512]);  view_1194 = None
        permute_449: "f32[512, 512]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_174: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg294_1, view_1195, permute_449);  arg294_1 = view_1195 = permute_449 = None
        view_1196: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_174, [32, 49, 512]);  addmm_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1197: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1196, [-1, 7, 7, 512]);  view_1196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1198: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1197, [-1, 2, 2, 7, 7, 512]);  view_1197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_450: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1198, [0, 1, 3, 2, 4, 5]);  view_1198 = None
        clone_485: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
        view_1199: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_485, [-1, 14, 14, 512]);  clone_485 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_82: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_466: "i64[14]" = torch.ops.aten.add.Tensor(iota_82, 11);  iota_82 = None
        fmod_82: "i64[14]" = torch.ops.aten.fmod.Scalar(add_466, 14);  add_466 = None
        index_126: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1199, [None, fmod_82]);  view_1199 = fmod_82 = None
        iota_83: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_467: "i64[14]" = torch.ops.aten.add.Tensor(iota_83, 11);  iota_83 = None
        fmod_83: "i64[14]" = torch.ops.aten.fmod.Scalar(add_467, 14);  add_467 = None
        index_127: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_126, [None, None, fmod_83]);  index_126 = fmod_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_468: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1177, index_127);  view_1177 = index_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1200: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_468, [8, -1, 512]);  add_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_95 = torch.ops.aten.var_mean.correction(view_1200, [2], correction = 0, keepdim = True)
        getitem_322: "f32[8, 196, 1]" = var_mean_95[0]
        getitem_323: "f32[8, 196, 1]" = var_mean_95[1];  var_mean_95 = None
        add_469: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_322, 1e-05);  getitem_322 = None
        rsqrt_95: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_469);  add_469 = None
        sub_139: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1200, getitem_323);  getitem_323 = None
        mul_363: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_95);  sub_139 = rsqrt_95 = None
        mul_364: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_363, arg295_1);  mul_363 = arg295_1 = None
        add_470: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_364, arg296_1);  mul_364 = arg296_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1201: "f32[1568, 512]" = torch.ops.aten.view.default(add_470, [1568, 512]);  add_470 = None
        permute_451: "f32[512, 2048]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_175: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg298_1, view_1201, permute_451);  arg298_1 = view_1201 = permute_451 = None
        view_1202: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_175, [8, 196, 2048]);  addmm_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_365: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1202, 0.5)
        mul_366: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1202, 0.7071067811865476);  view_1202 = None
        erf_43: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_366);  mul_366 = None
        add_471: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_367: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_365, add_471);  mul_365 = add_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1203: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_367, [1568, 2048]);  mul_367 = None
        permute_452: "f32[2048, 512]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_176: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg300_1, view_1203, permute_452);  arg300_1 = view_1203 = permute_452 = None
        view_1204: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_176, [8, 196, 512]);  addmm_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_472: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1200, view_1204);  view_1200 = view_1204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1205: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_472, [8, 14, 14, 512]);  add_472 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_96 = torch.ops.aten.var_mean.correction(view_1205, [3], correction = 0, keepdim = True)
        getitem_324: "f32[8, 14, 14, 1]" = var_mean_96[0]
        getitem_325: "f32[8, 14, 14, 1]" = var_mean_96[1];  var_mean_96 = None
        add_473: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_324, 1e-05);  getitem_324 = None
        rsqrt_96: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_473);  add_473 = None
        sub_140: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1205, getitem_325);  getitem_325 = None
        mul_368: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_96);  sub_140 = rsqrt_96 = None
        mul_369: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_368, arg301_1);  mul_368 = arg301_1 = None
        add_474: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_369, arg302_1);  mul_369 = arg302_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1206: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_474, [8, 2, 7, 2, 7, 512]);  add_474 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_453: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1206, [0, 1, 3, 2, 4, 5]);  view_1206 = None
        clone_488: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
        view_1207: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_488, [-1, 7, 7, 512]);  clone_488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1208: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1207, [-1, 49, 512]);  view_1207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1209: "f32[1568, 512]" = torch.ops.aten.view.default(view_1208, [1568, 512]);  view_1208 = None
        permute_454: "f32[512, 1536]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_177: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg304_1, view_1209, permute_454);  arg304_1 = view_1209 = permute_454 = None
        view_1210: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_177, [32, 49, 1536]);  addmm_177 = None
        view_1211: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1210, [32, 49, 3, 16, -1]);  view_1210 = None
        permute_455: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1211, [2, 0, 3, 1, 4]);  view_1211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_44 = torch.ops.aten.unbind.int(permute_455);  permute_455 = None
        getitem_326: "f32[32, 16, 49, 32]" = unbind_44[0]
        getitem_327: "f32[32, 16, 49, 32]" = unbind_44[1]
        getitem_328: "f32[32, 16, 49, 32]" = unbind_44[2];  unbind_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_370: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_326, 0.1767766952966369);  getitem_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_456: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_327, [0, 1, 3, 2]);  getitem_327 = None
        expand_176: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_370, [32, 16, 49, 32]);  mul_370 = None
        clone_489: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
        view_1212: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_489, [512, 49, 32]);  clone_489 = None
        expand_177: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_456, [32, 16, 32, 49]);  permute_456 = None
        clone_490: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_1213: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_490, [512, 32, 49]);  clone_490 = None
        bmm_88: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1212, view_1213);  view_1212 = view_1213 = None
        view_1214: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_88, [32, 16, 49, 49]);  bmm_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1215: "i64[2401]" = torch.ops.aten.view.default(arg306_1, [-1]);  arg306_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_128: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg305_1, [view_1215]);  arg305_1 = view_1215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1216: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_128, [49, 49, -1]);  index_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_457: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1216, [2, 0, 1]);  view_1216 = None
        clone_491: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_86: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_491, 0);  clone_491 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_475: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1214, unsqueeze_86);  view_1214 = unsqueeze_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_44: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(add_475, [-1], True)
        sub_141: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_475, amax_44);  add_475 = amax_44 = None
        exp_44: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_141);  sub_141 = None
        sum_45: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_44, [-1], True)
        div_44: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_44, sum_45);  exp_44 = sum_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_178: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_44, [32, 16, 49, 49]);  div_44 = None
        view_1217: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_178, [512, 49, 49]);  expand_178 = None
        expand_179: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_328, [32, 16, 49, 32]);  getitem_328 = None
        clone_493: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
        view_1218: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_493, [512, 49, 32]);  clone_493 = None
        bmm_89: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1217, view_1218);  view_1217 = view_1218 = None
        view_1219: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_89, [32, 16, 49, 32]);  bmm_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_458: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1219, [0, 2, 1, 3]);  view_1219 = None
        clone_494: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
        view_1220: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_494, [32, 49, 512]);  clone_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1221: "f32[1568, 512]" = torch.ops.aten.view.default(view_1220, [1568, 512]);  view_1220 = None
        permute_459: "f32[512, 512]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_178: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg308_1, view_1221, permute_459);  arg308_1 = view_1221 = permute_459 = None
        view_1222: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_178, [32, 49, 512]);  addmm_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1223: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1222, [-1, 7, 7, 512]);  view_1222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1224: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1223, [-1, 2, 2, 7, 7, 512]);  view_1223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_460: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1224, [0, 1, 3, 2, 4, 5]);  view_1224 = None
        clone_496: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
        view_1225: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_496, [-1, 14, 14, 512]);  clone_496 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_476: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1205, view_1225);  view_1205 = view_1225 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1226: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_476, [8, -1, 512]);  add_476 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_97 = torch.ops.aten.var_mean.correction(view_1226, [2], correction = 0, keepdim = True)
        getitem_329: "f32[8, 196, 1]" = var_mean_97[0]
        getitem_330: "f32[8, 196, 1]" = var_mean_97[1];  var_mean_97 = None
        add_477: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_329, 1e-05);  getitem_329 = None
        rsqrt_97: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_477);  add_477 = None
        sub_142: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1226, getitem_330);  getitem_330 = None
        mul_371: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_97);  sub_142 = rsqrt_97 = None
        mul_372: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_371, arg309_1);  mul_371 = arg309_1 = None
        add_478: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_372, arg310_1);  mul_372 = arg310_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1227: "f32[1568, 512]" = torch.ops.aten.view.default(add_478, [1568, 512]);  add_478 = None
        permute_461: "f32[512, 2048]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_179: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg312_1, view_1227, permute_461);  arg312_1 = view_1227 = permute_461 = None
        view_1228: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_179, [8, 196, 2048]);  addmm_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_373: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1228, 0.5)
        mul_374: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1228, 0.7071067811865476);  view_1228 = None
        erf_44: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_374);  mul_374 = None
        add_479: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_375: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_373, add_479);  mul_373 = add_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1229: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_375, [1568, 2048]);  mul_375 = None
        permute_462: "f32[2048, 512]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_180: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg314_1, view_1229, permute_462);  arg314_1 = view_1229 = permute_462 = None
        view_1230: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_180, [8, 196, 512]);  addmm_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_480: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1226, view_1230);  view_1226 = view_1230 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1231: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_480, [8, 14, 14, 512]);  add_480 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_98 = torch.ops.aten.var_mean.correction(view_1231, [3], correction = 0, keepdim = True)
        getitem_331: "f32[8, 14, 14, 1]" = var_mean_98[0]
        getitem_332: "f32[8, 14, 14, 1]" = var_mean_98[1];  var_mean_98 = None
        add_481: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_331, 1e-05);  getitem_331 = None
        rsqrt_98: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
        sub_143: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_1231, getitem_332);  getitem_332 = None
        mul_376: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_98);  sub_143 = rsqrt_98 = None
        mul_377: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_376, arg315_1);  mul_376 = arg315_1 = None
        add_482: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_377, arg316_1);  mul_377 = arg316_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:371 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_84: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_483: "i64[14]" = torch.ops.aten.add.Tensor(iota_84, 3);  iota_84 = None
        fmod_84: "i64[14]" = torch.ops.aten.fmod.Scalar(add_483, 14);  add_483 = None
        index_129: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_482, [None, fmod_84]);  add_482 = fmod_84 = None
        iota_85: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_484: "i64[14]" = torch.ops.aten.add.Tensor(iota_85, 3);  iota_85 = None
        fmod_85: "i64[14]" = torch.ops.aten.fmod.Scalar(add_484, 14);  add_484 = None
        index_130: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_129, [None, None, fmod_85]);  index_129 = fmod_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1232: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_130, [8, 2, 7, 2, 7, 512]);  index_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_463: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1232, [0, 1, 3, 2, 4, 5]);  view_1232 = None
        clone_499: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_463, memory_format = torch.contiguous_format);  permute_463 = None
        view_1233: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(clone_499, [-1, 7, 7, 512]);  clone_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1234: "f32[32, 49, 512]" = torch.ops.aten.view.default(view_1233, [-1, 49, 512]);  view_1233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1235: "f32[1568, 512]" = torch.ops.aten.view.default(view_1234, [1568, 512]);  view_1234 = None
        permute_464: "f32[512, 1536]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_181: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg319_1, view_1235, permute_464);  arg319_1 = view_1235 = permute_464 = None
        view_1236: "f32[32, 49, 1536]" = torch.ops.aten.view.default(addmm_181, [32, 49, 1536]);  addmm_181 = None
        view_1237: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_1236, [32, 49, 3, 16, -1]);  view_1236 = None
        permute_465: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1237, [2, 0, 3, 1, 4]);  view_1237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_45 = torch.ops.aten.unbind.int(permute_465);  permute_465 = None
        getitem_333: "f32[32, 16, 49, 32]" = unbind_45[0]
        getitem_334: "f32[32, 16, 49, 32]" = unbind_45[1]
        getitem_335: "f32[32, 16, 49, 32]" = unbind_45[2];  unbind_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_378: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_333, 0.1767766952966369);  getitem_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_466: "f32[32, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_334, [0, 1, 3, 2]);  getitem_334 = None
        expand_180: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(mul_378, [32, 16, 49, 32]);  mul_378 = None
        clone_500: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_1238: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_500, [512, 49, 32]);  clone_500 = None
        expand_181: "f32[32, 16, 32, 49]" = torch.ops.aten.expand.default(permute_466, [32, 16, 32, 49]);  permute_466 = None
        clone_501: "f32[32, 16, 32, 49]" = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_1239: "f32[512, 32, 49]" = torch.ops.aten.view.default(clone_501, [512, 32, 49]);  clone_501 = None
        bmm_90: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1238, view_1239);  view_1238 = view_1239 = None
        view_1240: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(bmm_90, [32, 16, 49, 49]);  bmm_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1241: "i64[2401]" = torch.ops.aten.view.default(arg321_1, [-1]);  arg321_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_131: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg320_1, [view_1241]);  arg320_1 = view_1241 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1242: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_131, [49, 49, -1]);  index_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_467: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_1242, [2, 0, 1]);  view_1242 = None
        clone_502: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_87: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_502, 0);  clone_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_485: "f32[32, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1240, unsqueeze_87);  view_1240 = unsqueeze_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:197 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_1243: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_485, [-1, 4, 16, 49, 49]);  add_485 = None
        unsqueeze_88: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg317_1, 1);  arg317_1 = None
        unsqueeze_89: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 0);  unsqueeze_88 = None
        add_486: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1243, unsqueeze_89);  view_1243 = unsqueeze_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:198 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_1244: "f32[32, 16, 49, 49]" = torch.ops.aten.view.default(add_486, [-1, 16, 49, 49]);  add_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_45: "f32[32, 16, 49, 1]" = torch.ops.aten.amax.default(view_1244, [-1], True)
        sub_144: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_1244, amax_45);  view_1244 = amax_45 = None
        exp_45: "f32[32, 16, 49, 49]" = torch.ops.aten.exp.default(sub_144);  sub_144 = None
        sum_46: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_45, [-1], True)
        div_45: "f32[32, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_45, sum_46);  exp_45 = sum_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_182: "f32[32, 16, 49, 49]" = torch.ops.aten.expand.default(div_45, [32, 16, 49, 49]);  div_45 = None
        view_1245: "f32[512, 49, 49]" = torch.ops.aten.view.default(expand_182, [512, 49, 49]);  expand_182 = None
        expand_183: "f32[32, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_335, [32, 16, 49, 32]);  getitem_335 = None
        clone_504: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(expand_183, memory_format = torch.contiguous_format);  expand_183 = None
        view_1246: "f32[512, 49, 32]" = torch.ops.aten.view.default(clone_504, [512, 49, 32]);  clone_504 = None
        bmm_91: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1245, view_1246);  view_1245 = view_1246 = None
        view_1247: "f32[32, 16, 49, 32]" = torch.ops.aten.view.default(bmm_91, [32, 16, 49, 32]);  bmm_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_468: "f32[32, 49, 16, 32]" = torch.ops.aten.permute.default(view_1247, [0, 2, 1, 3]);  view_1247 = None
        clone_505: "f32[32, 49, 16, 32]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_1248: "f32[32, 49, 512]" = torch.ops.aten.view.default(clone_505, [32, 49, 512]);  clone_505 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1249: "f32[1568, 512]" = torch.ops.aten.view.default(view_1248, [1568, 512]);  view_1248 = None
        permute_469: "f32[512, 512]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        addmm_182: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg323_1, view_1249, permute_469);  arg323_1 = view_1249 = permute_469 = None
        view_1250: "f32[32, 49, 512]" = torch.ops.aten.view.default(addmm_182, [32, 49, 512]);  addmm_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1251: "f32[32, 7, 7, 512]" = torch.ops.aten.view.default(view_1250, [-1, 7, 7, 512]);  view_1250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1252: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1251, [-1, 2, 2, 7, 7, 512]);  view_1251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_470: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1252, [0, 1, 3, 2, 4, 5]);  view_1252 = None
        clone_507: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None
        view_1253: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(clone_507, [-1, 14, 14, 512]);  clone_507 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:399 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        iota_86: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_487: "i64[14]" = torch.ops.aten.add.Tensor(iota_86, 11);  iota_86 = None
        fmod_86: "i64[14]" = torch.ops.aten.fmod.Scalar(add_487, 14);  add_487 = None
        index_132: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1253, [None, fmod_86]);  view_1253 = fmod_86 = None
        iota_87: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_488: "i64[14]" = torch.ops.aten.add.Tensor(iota_87, 11);  iota_87 = None
        fmod_87: "i64[14]" = torch.ops.aten.fmod.Scalar(add_488, 14);  add_488 = None
        index_133: "f32[8, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_132, [None, None, fmod_87]);  index_132 = fmod_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_489: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1231, index_133);  view_1231 = index_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1254: "f32[8, 196, 512]" = torch.ops.aten.view.default(add_489, [8, -1, 512]);  add_489 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_99 = torch.ops.aten.var_mean.correction(view_1254, [2], correction = 0, keepdim = True)
        getitem_336: "f32[8, 196, 1]" = var_mean_99[0]
        getitem_337: "f32[8, 196, 1]" = var_mean_99[1];  var_mean_99 = None
        add_490: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_336, 1e-05);  getitem_336 = None
        rsqrt_99: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_490);  add_490 = None
        sub_145: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(view_1254, getitem_337);  getitem_337 = None
        mul_379: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_99);  sub_145 = rsqrt_99 = None
        mul_380: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_379, arg324_1);  mul_379 = arg324_1 = None
        add_491: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(mul_380, arg325_1);  mul_380 = arg325_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1255: "f32[1568, 512]" = torch.ops.aten.view.default(add_491, [1568, 512]);  add_491 = None
        permute_471: "f32[512, 2048]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_183: "f32[1568, 2048]" = torch.ops.aten.addmm.default(arg327_1, view_1255, permute_471);  arg327_1 = view_1255 = permute_471 = None
        view_1256: "f32[8, 196, 2048]" = torch.ops.aten.view.default(addmm_183, [8, 196, 2048]);  addmm_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_381: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1256, 0.5)
        mul_382: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1256, 0.7071067811865476);  view_1256 = None
        erf_45: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_382);  mul_382 = None
        add_492: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_383: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_381, add_492);  mul_381 = add_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1257: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_383, [1568, 2048]);  mul_383 = None
        permute_472: "f32[2048, 512]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_184: "f32[1568, 512]" = torch.ops.aten.addmm.default(arg329_1, view_1257, permute_472);  arg329_1 = view_1257 = permute_472 = None
        view_1258: "f32[8, 196, 512]" = torch.ops.aten.view.default(addmm_184, [8, 196, 512]);  addmm_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_493: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1254, view_1258);  view_1254 = view_1258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1259: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(add_493, [8, 14, 14, 512]);  add_493 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:442 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_1260: "f32[8, 7, 2, 7, 2, 512]" = torch.ops.aten.view.default(view_1259, [8, 7, 2, 7, 2, 512]);  view_1259 = None
        permute_473: "f32[8, 7, 7, 2, 2, 512]" = torch.ops.aten.permute.default(view_1260, [0, 1, 3, 4, 2, 5]);  view_1260 = None
        clone_510: "f32[8, 7, 7, 2, 2, 512]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
        view_1261: "f32[8, 7, 7, 2048]" = torch.ops.aten.view.default(clone_510, [8, 7, 7, 2048]);  clone_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in forward, code: x = self.norm(x)
        var_mean_100 = torch.ops.aten.var_mean.correction(view_1261, [3], correction = 0, keepdim = True)
        getitem_338: "f32[8, 7, 7, 1]" = var_mean_100[0]
        getitem_339: "f32[8, 7, 7, 1]" = var_mean_100[1];  var_mean_100 = None
        add_494: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-05);  getitem_338 = None
        rsqrt_100: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_494);  add_494 = None
        sub_146: "f32[8, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(view_1261, getitem_339);  view_1261 = getitem_339 = None
        mul_384: "f32[8, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_100);  sub_146 = rsqrt_100 = None
        mul_385: "f32[8, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_384, arg330_1);  mul_384 = arg330_1 = None
        add_495: "f32[8, 7, 7, 2048]" = torch.ops.aten.add.Tensor(mul_385, arg331_1);  mul_385 = arg331_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:444 in forward, code: x = self.reduction(x)
        permute_474: "f32[2048, 1024]" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        view_1262: "f32[392, 2048]" = torch.ops.aten.view.default(add_495, [392, 2048]);  add_495 = None
        mm_5: "f32[392, 1024]" = torch.ops.aten.mm.default(view_1262, permute_474);  view_1262 = permute_474 = None
        view_1263: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(mm_5, [8, 7, 7, 1024]);  mm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_101 = torch.ops.aten.var_mean.correction(view_1263, [3], correction = 0, keepdim = True)
        getitem_340: "f32[8, 7, 7, 1]" = var_mean_101[0]
        getitem_341: "f32[8, 7, 7, 1]" = var_mean_101[1];  var_mean_101 = None
        add_496: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_340, 1e-05);  getitem_340 = None
        rsqrt_101: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_496);  add_496 = None
        sub_147: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_1263, getitem_341);  getitem_341 = None
        mul_386: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_101);  sub_147 = rsqrt_101 = None
        mul_387: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_386, arg333_1);  mul_386 = arg333_1 = None
        add_497: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_387, arg334_1);  mul_387 = arg334_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1264: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(add_497, [8, 1, 7, 1, 7, 1024]);  add_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_475: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_1264, [0, 1, 3, 2, 4, 5]);  view_1264 = None
        view_1265: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_475, [-1, 7, 7, 1024]);  permute_475 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1266: "f32[8, 49, 1024]" = torch.ops.aten.view.default(view_1265, [-1, 49, 1024]);  view_1265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1267: "f32[392, 1024]" = torch.ops.aten.view.default(view_1266, [392, 1024]);  view_1266 = None
        permute_476: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_185: "f32[392, 3072]" = torch.ops.aten.addmm.default(arg336_1, view_1267, permute_476);  arg336_1 = view_1267 = permute_476 = None
        view_1268: "f32[8, 49, 3072]" = torch.ops.aten.view.default(addmm_185, [8, 49, 3072]);  addmm_185 = None
        view_1269: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.view.default(view_1268, [8, 49, 3, 32, -1]);  view_1268 = None
        permute_477: "f32[3, 8, 32, 49, 32]" = torch.ops.aten.permute.default(view_1269, [2, 0, 3, 1, 4]);  view_1269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_46 = torch.ops.aten.unbind.int(permute_477);  permute_477 = None
        getitem_342: "f32[8, 32, 49, 32]" = unbind_46[0]
        getitem_343: "f32[8, 32, 49, 32]" = unbind_46[1]
        getitem_344: "f32[8, 32, 49, 32]" = unbind_46[2];  unbind_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_388: "f32[8, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_342, 0.1767766952966369);  getitem_342 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_478: "f32[8, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_343, [0, 1, 3, 2]);  getitem_343 = None
        expand_184: "f32[8, 32, 49, 32]" = torch.ops.aten.expand.default(mul_388, [8, 32, 49, 32]);  mul_388 = None
        clone_511: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
        view_1270: "f32[256, 49, 32]" = torch.ops.aten.view.default(clone_511, [256, 49, 32]);  clone_511 = None
        expand_185: "f32[8, 32, 32, 49]" = torch.ops.aten.expand.default(permute_478, [8, 32, 32, 49]);  permute_478 = None
        clone_512: "f32[8, 32, 32, 49]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_1271: "f32[256, 32, 49]" = torch.ops.aten.view.default(clone_512, [256, 32, 49]);  clone_512 = None
        bmm_92: "f32[256, 49, 49]" = torch.ops.aten.bmm.default(view_1270, view_1271);  view_1270 = view_1271 = None
        view_1272: "f32[8, 32, 49, 49]" = torch.ops.aten.view.default(bmm_92, [8, 32, 49, 49]);  bmm_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1273: "i64[2401]" = torch.ops.aten.view.default(arg338_1, [-1]);  arg338_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_134: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg337_1, [view_1273]);  arg337_1 = view_1273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1274: "f32[49, 49, 32]" = torch.ops.aten.view.default(index_134, [49, 49, -1]);  index_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_479: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_1274, [2, 0, 1]);  view_1274 = None
        clone_513: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_90: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_513, 0);  clone_513 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_498: "f32[8, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_1272, unsqueeze_90);  view_1272 = unsqueeze_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_46: "f32[8, 32, 49, 1]" = torch.ops.aten.amax.default(add_498, [-1], True)
        sub_148: "f32[8, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_498, amax_46);  add_498 = amax_46 = None
        exp_46: "f32[8, 32, 49, 49]" = torch.ops.aten.exp.default(sub_148);  sub_148 = None
        sum_47: "f32[8, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_46, [-1], True)
        div_46: "f32[8, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_46, sum_47);  exp_46 = sum_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_186: "f32[8, 32, 49, 49]" = torch.ops.aten.expand.default(div_46, [8, 32, 49, 49]);  div_46 = None
        view_1275: "f32[256, 49, 49]" = torch.ops.aten.view.default(expand_186, [256, 49, 49]);  expand_186 = None
        expand_187: "f32[8, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_344, [8, 32, 49, 32]);  getitem_344 = None
        clone_515: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
        view_1276: "f32[256, 49, 32]" = torch.ops.aten.view.default(clone_515, [256, 49, 32]);  clone_515 = None
        bmm_93: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(view_1275, view_1276);  view_1275 = view_1276 = None
        view_1277: "f32[8, 32, 49, 32]" = torch.ops.aten.view.default(bmm_93, [8, 32, 49, 32]);  bmm_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_480: "f32[8, 49, 32, 32]" = torch.ops.aten.permute.default(view_1277, [0, 2, 1, 3]);  view_1277 = None
        clone_516: "f32[8, 49, 32, 32]" = torch.ops.aten.clone.default(permute_480, memory_format = torch.contiguous_format);  permute_480 = None
        view_1278: "f32[8, 49, 1024]" = torch.ops.aten.view.default(clone_516, [8, 49, 1024]);  clone_516 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1279: "f32[392, 1024]" = torch.ops.aten.view.default(view_1278, [392, 1024]);  view_1278 = None
        permute_481: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_186: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg340_1, view_1279, permute_481);  arg340_1 = view_1279 = permute_481 = None
        view_1280: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_186, [8, 49, 1024]);  addmm_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1281: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(view_1280, [-1, 7, 7, 1024]);  view_1280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1282: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_1281, [-1, 1, 1, 7, 7, 1024]);  view_1281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_482: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_1282, [0, 1, 3, 2, 4, 5]);  view_1282 = None
        view_1283: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_482, [-1, 7, 7, 1024]);  permute_482 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_499: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_1263, view_1283);  view_1263 = view_1283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1284: "f32[8, 49, 1024]" = torch.ops.aten.view.default(add_499, [8, -1, 1024]);  add_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_102 = torch.ops.aten.var_mean.correction(view_1284, [2], correction = 0, keepdim = True)
        getitem_345: "f32[8, 49, 1]" = var_mean_102[0]
        getitem_346: "f32[8, 49, 1]" = var_mean_102[1];  var_mean_102 = None
        add_500: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_345, 1e-05);  getitem_345 = None
        rsqrt_102: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_500);  add_500 = None
        sub_149: "f32[8, 49, 1024]" = torch.ops.aten.sub.Tensor(view_1284, getitem_346);  getitem_346 = None
        mul_389: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_102);  sub_149 = rsqrt_102 = None
        mul_390: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_389, arg341_1);  mul_389 = arg341_1 = None
        add_501: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(mul_390, arg342_1);  mul_390 = arg342_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1285: "f32[392, 1024]" = torch.ops.aten.view.default(add_501, [392, 1024]);  add_501 = None
        permute_483: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_187: "f32[392, 4096]" = torch.ops.aten.addmm.default(arg344_1, view_1285, permute_483);  arg344_1 = view_1285 = permute_483 = None
        view_1286: "f32[8, 49, 4096]" = torch.ops.aten.view.default(addmm_187, [8, 49, 4096]);  addmm_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_391: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_1286, 0.5)
        mul_392: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_1286, 0.7071067811865476);  view_1286 = None
        erf_46: "f32[8, 49, 4096]" = torch.ops.aten.erf.default(mul_392);  mul_392 = None
        add_502: "f32[8, 49, 4096]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_393: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_391, add_502);  mul_391 = add_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1287: "f32[392, 4096]" = torch.ops.aten.view.default(mul_393, [392, 4096]);  mul_393 = None
        permute_484: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_188: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg346_1, view_1287, permute_484);  arg346_1 = view_1287 = permute_484 = None
        view_1288: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_188, [8, 49, 1024]);  addmm_188 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_503: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(view_1284, view_1288);  view_1284 = view_1288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1289: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(add_503, [8, 7, 7, 1024]);  add_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_103 = torch.ops.aten.var_mean.correction(view_1289, [3], correction = 0, keepdim = True)
        getitem_347: "f32[8, 7, 7, 1]" = var_mean_103[0]
        getitem_348: "f32[8, 7, 7, 1]" = var_mean_103[1];  var_mean_103 = None
        add_504: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_347, 1e-05);  getitem_347 = None
        rsqrt_103: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_504);  add_504 = None
        sub_150: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_1289, getitem_348);  getitem_348 = None
        mul_394: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_103);  sub_150 = rsqrt_103 = None
        mul_395: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_394, arg347_1);  mul_394 = arg347_1 = None
        add_505: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_395, arg348_1);  mul_395 = arg348_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_1290: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(add_505, [8, 1, 7, 1, 7, 1024]);  add_505 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:58 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_485: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_1290, [0, 1, 3, 2, 4, 5]);  view_1290 = None
        view_1291: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_485, [-1, 7, 7, 1024]);  permute_485 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:383 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1292: "f32[8, 49, 1024]" = torch.ops.aten.view.default(view_1291, [-1, 49, 1024]);  view_1291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:177 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_1293: "f32[392, 1024]" = torch.ops.aten.view.default(view_1292, [392, 1024]);  view_1292 = None
        permute_486: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_189: "f32[392, 3072]" = torch.ops.aten.addmm.default(arg350_1, view_1293, permute_486);  arg350_1 = view_1293 = permute_486 = None
        view_1294: "f32[8, 49, 3072]" = torch.ops.aten.view.default(addmm_189, [8, 49, 3072]);  addmm_189 = None
        view_1295: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.view.default(view_1294, [8, 49, 3, 32, -1]);  view_1294 = None
        permute_487: "f32[3, 8, 32, 49, 32]" = torch.ops.aten.permute.default(view_1295, [2, 0, 3, 1, 4]);  view_1295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:178 in forward, code: q, k, v = qkv.unbind(0)
        unbind_47 = torch.ops.aten.unbind.int(permute_487);  permute_487 = None
        getitem_349: "f32[8, 32, 49, 32]" = unbind_47[0]
        getitem_350: "f32[8, 32, 49, 32]" = unbind_47[1]
        getitem_351: "f32[8, 32, 49, 32]" = unbind_47[2];  unbind_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:192 in forward, code: q = q * self.scale
        mul_396: "f32[8, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_349, 0.1767766952966369);  getitem_349 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:193 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_488: "f32[8, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_350, [0, 1, 3, 2]);  getitem_350 = None
        expand_188: "f32[8, 32, 49, 32]" = torch.ops.aten.expand.default(mul_396, [8, 32, 49, 32]);  mul_396 = None
        clone_520: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(expand_188, memory_format = torch.contiguous_format);  expand_188 = None
        view_1296: "f32[256, 49, 32]" = torch.ops.aten.view.default(clone_520, [256, 49, 32]);  clone_520 = None
        expand_189: "f32[8, 32, 32, 49]" = torch.ops.aten.expand.default(permute_488, [8, 32, 32, 49]);  permute_488 = None
        clone_521: "f32[8, 32, 32, 49]" = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_1297: "f32[256, 32, 49]" = torch.ops.aten.view.default(clone_521, [256, 32, 49]);  clone_521 = None
        bmm_94: "f32[256, 49, 49]" = torch.ops.aten.bmm.default(view_1296, view_1297);  view_1296 = view_1297 = None
        view_1298: "f32[8, 32, 49, 49]" = torch.ops.aten.view.default(bmm_94, [8, 32, 49, 49]);  bmm_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1299: "i64[2401]" = torch.ops.aten.view.default(arg352_1, [-1]);  arg352_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:165 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_135: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg351_1, [view_1299]);  arg351_1 = view_1299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:166 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1300: "f32[49, 49, 32]" = torch.ops.aten.view.default(index_135, [49, 49, -1]);  index_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:167 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_489: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_1300, [2, 0, 1]);  view_1300 = None
        clone_522: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:168 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_91: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_522, 0);  clone_522 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:194 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_506: "f32[8, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_1298, unsqueeze_91);  view_1298 = unsqueeze_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:199 in forward, code: attn = self.softmax(attn)
        amax_47: "f32[8, 32, 49, 1]" = torch.ops.aten.amax.default(add_506, [-1], True)
        sub_151: "f32[8, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_506, amax_47);  add_506 = amax_47 = None
        exp_47: "f32[8, 32, 49, 49]" = torch.ops.aten.exp.default(sub_151);  sub_151 = None
        sum_48: "f32[8, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_47, [-1], True)
        div_47: "f32[8, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_47, sum_48);  exp_47 = sum_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:201 in forward, code: x = attn @ v
        expand_190: "f32[8, 32, 49, 49]" = torch.ops.aten.expand.default(div_47, [8, 32, 49, 49]);  div_47 = None
        view_1301: "f32[256, 49, 49]" = torch.ops.aten.view.default(expand_190, [256, 49, 49]);  expand_190 = None
        expand_191: "f32[8, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_351, [8, 32, 49, 32]);  getitem_351 = None
        clone_524: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(expand_191, memory_format = torch.contiguous_format);  expand_191 = None
        view_1302: "f32[256, 49, 32]" = torch.ops.aten.view.default(clone_524, [256, 49, 32]);  clone_524 = None
        bmm_95: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(view_1301, view_1302);  view_1301 = view_1302 = None
        view_1303: "f32[8, 32, 49, 32]" = torch.ops.aten.view.default(bmm_95, [8, 32, 49, 32]);  bmm_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_490: "f32[8, 49, 32, 32]" = torch.ops.aten.permute.default(view_1303, [0, 2, 1, 3]);  view_1303 = None
        clone_525: "f32[8, 49, 32, 32]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_1304: "f32[8, 49, 1024]" = torch.ops.aten.view.default(clone_525, [8, 49, 1024]);  clone_525 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in forward, code: x = self.proj(x)
        view_1305: "f32[392, 1024]" = torch.ops.aten.view.default(view_1304, [392, 1024]);  view_1304 = None
        permute_491: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_190: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg354_1, view_1305, permute_491);  arg354_1 = view_1305 = permute_491 = None
        view_1306: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_190, [8, 49, 1024]);  addmm_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:393 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1307: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(view_1306, [-1, 7, 7, 1024]);  view_1306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_1308: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_1307, [-1, 1, 1, 7, 7, 1024]);  view_1307 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_492: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_1308, [0, 1, 3, 2, 4, 5]);  view_1308 = None
        view_1309: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(permute_492, [-1, 7, 7, 1024]);  permute_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:406 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_507: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_1289, view_1309);  view_1289 = view_1309 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:407 in forward, code: x = x.reshape(B, -1, C)
        view_1310: "f32[8, 49, 1024]" = torch.ops.aten.view.default(add_507, [8, -1, 1024]);  add_507 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_104 = torch.ops.aten.var_mean.correction(view_1310, [2], correction = 0, keepdim = True)
        getitem_352: "f32[8, 49, 1]" = var_mean_104[0]
        getitem_353: "f32[8, 49, 1]" = var_mean_104[1];  var_mean_104 = None
        add_508: "f32[8, 49, 1]" = torch.ops.aten.add.Tensor(getitem_352, 1e-05);  getitem_352 = None
        rsqrt_104: "f32[8, 49, 1]" = torch.ops.aten.rsqrt.default(add_508);  add_508 = None
        sub_152: "f32[8, 49, 1024]" = torch.ops.aten.sub.Tensor(view_1310, getitem_353);  getitem_353 = None
        mul_397: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_104);  sub_152 = rsqrt_104 = None
        mul_398: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_397, arg355_1);  mul_397 = arg355_1 = None
        add_509: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(mul_398, arg356_1);  mul_398 = arg356_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_1311: "f32[392, 1024]" = torch.ops.aten.view.default(add_509, [392, 1024]);  add_509 = None
        permute_493: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_191: "f32[392, 4096]" = torch.ops.aten.addmm.default(arg358_1, view_1311, permute_493);  arg358_1 = view_1311 = permute_493 = None
        view_1312: "f32[8, 49, 4096]" = torch.ops.aten.view.default(addmm_191, [8, 49, 4096]);  addmm_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        mul_399: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_1312, 0.5)
        mul_400: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_1312, 0.7071067811865476);  view_1312 = None
        erf_47: "f32[8, 49, 4096]" = torch.ops.aten.erf.default(mul_400);  mul_400 = None
        add_510: "f32[8, 49, 4096]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_401: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_399, add_510);  mul_399 = add_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_1313: "f32[392, 4096]" = torch.ops.aten.view.default(mul_401, [392, 4096]);  mul_401 = None
        permute_494: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_192: "f32[392, 1024]" = torch.ops.aten.addmm.default(arg360_1, view_1313, permute_494);  arg360_1 = view_1313 = permute_494 = None
        view_1314: "f32[8, 49, 1024]" = torch.ops.aten.view.default(addmm_192, [8, 49, 1024]);  addmm_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:408 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_511: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(view_1310, view_1314);  view_1310 = view_1314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:409 in forward, code: x = x.reshape(B, H, W, C)
        view_1315: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(add_511, [8, 7, 7, 1024]);  add_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/swin_transformer.py:831 in forward_features, code: x = self.norm(x)
        var_mean_105 = torch.ops.aten.var_mean.correction(view_1315, [3], correction = 0, keepdim = True)
        getitem_354: "f32[8, 7, 7, 1]" = var_mean_105[0]
        getitem_355: "f32[8, 7, 7, 1]" = var_mean_105[1];  var_mean_105 = None
        add_512: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_354, 1e-05);  getitem_354 = None
        rsqrt_105: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_512);  add_512 = None
        sub_153: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_1315, getitem_355);  view_1315 = getitem_355 = None
        mul_402: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt_105);  sub_153 = rsqrt_105 = None
        mul_403: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_402, arg361_1);  mul_402 = arg361_1 = None
        add_513: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_403, arg362_1);  mul_403 = arg362_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:65 in forward, code: return x.mean(self.dim, keepdim=not self.flatten)
        mean_1: "f32[8, 1024]" = torch.ops.aten.mean.dim(add_513, [1, 2]);  add_513 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_495: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_193: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg364_1, mean_1, permute_495);  arg364_1 = mean_1 = permute_495 = None
        return (addmm_193,)
        