class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[16]", arg3_1: "f32[16]", arg4_1: "f32[16]", arg5_1: "f32[16]", arg6_1: "f32[32, 16, 3, 3]", arg7_1: "f32[32]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[64, 32, 3, 3]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[128, 64, 3, 3]", arg17_1: "f32[128]", arg18_1: "f32[128]", arg19_1: "f32[128]", arg20_1: "f32[128]", arg21_1: "f32[256, 128]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[256]", arg25_1: "f32[256]", arg26_1: "f32[4, 196]", arg27_1: "i64[196, 196]", arg28_1: "f32[128, 128]", arg29_1: "f32[128]", arg30_1: "f32[128]", arg31_1: "f32[128]", arg32_1: "f32[128]", arg33_1: "f32[256, 128]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[256]", arg38_1: "f32[128, 256]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[128]", arg42_1: "f32[128]", arg43_1: "f32[256, 128]", arg44_1: "f32[256]", arg45_1: "f32[256]", arg46_1: "f32[256]", arg47_1: "f32[256]", arg48_1: "f32[4, 196]", arg49_1: "i64[196, 196]", arg50_1: "f32[128, 128]", arg51_1: "f32[128]", arg52_1: "f32[128]", arg53_1: "f32[128]", arg54_1: "f32[128]", arg55_1: "f32[256, 128]", arg56_1: "f32[256]", arg57_1: "f32[256]", arg58_1: "f32[256]", arg59_1: "f32[256]", arg60_1: "f32[128, 256]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[256, 128]", arg66_1: "f32[256]", arg67_1: "f32[256]", arg68_1: "f32[256]", arg69_1: "f32[256]", arg70_1: "f32[4, 196]", arg71_1: "i64[196, 196]", arg72_1: "f32[128, 128]", arg73_1: "f32[128]", arg74_1: "f32[128]", arg75_1: "f32[128]", arg76_1: "f32[128]", arg77_1: "f32[256, 128]", arg78_1: "f32[256]", arg79_1: "f32[256]", arg80_1: "f32[256]", arg81_1: "f32[256]", arg82_1: "f32[128, 256]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[128]", arg86_1: "f32[128]", arg87_1: "f32[256, 128]", arg88_1: "f32[256]", arg89_1: "f32[256]", arg90_1: "f32[256]", arg91_1: "f32[256]", arg92_1: "f32[4, 196]", arg93_1: "i64[196, 196]", arg94_1: "f32[128, 128]", arg95_1: "f32[128]", arg96_1: "f32[128]", arg97_1: "f32[128]", arg98_1: "f32[128]", arg99_1: "f32[256, 128]", arg100_1: "f32[256]", arg101_1: "f32[256]", arg102_1: "f32[256]", arg103_1: "f32[256]", arg104_1: "f32[128, 256]", arg105_1: "f32[128]", arg106_1: "f32[128]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "f32[640, 128]", arg110_1: "f32[640]", arg111_1: "f32[640]", arg112_1: "f32[640]", arg113_1: "f32[640]", arg114_1: "f32[128, 128]", arg115_1: "f32[128]", arg116_1: "f32[128]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[8, 196]", arg120_1: "i64[49, 196]", arg121_1: "f32[256, 512]", arg122_1: "f32[256]", arg123_1: "f32[256]", arg124_1: "f32[256]", arg125_1: "f32[256]", arg126_1: "f32[512, 256]", arg127_1: "f32[512]", arg128_1: "f32[512]", arg129_1: "f32[512]", arg130_1: "f32[512]", arg131_1: "f32[256, 512]", arg132_1: "f32[256]", arg133_1: "f32[256]", arg134_1: "f32[256]", arg135_1: "f32[256]", arg136_1: "f32[512, 256]", arg137_1: "f32[512]", arg138_1: "f32[512]", arg139_1: "f32[512]", arg140_1: "f32[512]", arg141_1: "f32[8, 49]", arg142_1: "i64[49, 49]", arg143_1: "f32[256, 256]", arg144_1: "f32[256]", arg145_1: "f32[256]", arg146_1: "f32[256]", arg147_1: "f32[256]", arg148_1: "f32[512, 256]", arg149_1: "f32[512]", arg150_1: "f32[512]", arg151_1: "f32[512]", arg152_1: "f32[512]", arg153_1: "f32[256, 512]", arg154_1: "f32[256]", arg155_1: "f32[256]", arg156_1: "f32[256]", arg157_1: "f32[256]", arg158_1: "f32[512, 256]", arg159_1: "f32[512]", arg160_1: "f32[512]", arg161_1: "f32[512]", arg162_1: "f32[512]", arg163_1: "f32[8, 49]", arg164_1: "i64[49, 49]", arg165_1: "f32[256, 256]", arg166_1: "f32[256]", arg167_1: "f32[256]", arg168_1: "f32[256]", arg169_1: "f32[256]", arg170_1: "f32[512, 256]", arg171_1: "f32[512]", arg172_1: "f32[512]", arg173_1: "f32[512]", arg174_1: "f32[512]", arg175_1: "f32[256, 512]", arg176_1: "f32[256]", arg177_1: "f32[256]", arg178_1: "f32[256]", arg179_1: "f32[256]", arg180_1: "f32[512, 256]", arg181_1: "f32[512]", arg182_1: "f32[512]", arg183_1: "f32[512]", arg184_1: "f32[512]", arg185_1: "f32[8, 49]", arg186_1: "i64[49, 49]", arg187_1: "f32[256, 256]", arg188_1: "f32[256]", arg189_1: "f32[256]", arg190_1: "f32[256]", arg191_1: "f32[256]", arg192_1: "f32[512, 256]", arg193_1: "f32[512]", arg194_1: "f32[512]", arg195_1: "f32[512]", arg196_1: "f32[512]", arg197_1: "f32[256, 512]", arg198_1: "f32[256]", arg199_1: "f32[256]", arg200_1: "f32[256]", arg201_1: "f32[256]", arg202_1: "f32[512, 256]", arg203_1: "f32[512]", arg204_1: "f32[512]", arg205_1: "f32[512]", arg206_1: "f32[512]", arg207_1: "f32[8, 49]", arg208_1: "i64[49, 49]", arg209_1: "f32[256, 256]", arg210_1: "f32[256]", arg211_1: "f32[256]", arg212_1: "f32[256]", arg213_1: "f32[256]", arg214_1: "f32[512, 256]", arg215_1: "f32[512]", arg216_1: "f32[512]", arg217_1: "f32[512]", arg218_1: "f32[512]", arg219_1: "f32[256, 512]", arg220_1: "f32[256]", arg221_1: "f32[256]", arg222_1: "f32[256]", arg223_1: "f32[256]", arg224_1: "f32[1280, 256]", arg225_1: "f32[1280]", arg226_1: "f32[1280]", arg227_1: "f32[1280]", arg228_1: "f32[1280]", arg229_1: "f32[256, 256]", arg230_1: "f32[256]", arg231_1: "f32[256]", arg232_1: "f32[256]", arg233_1: "f32[256]", arg234_1: "f32[16, 49]", arg235_1: "i64[16, 49]", arg236_1: "f32[384, 1024]", arg237_1: "f32[384]", arg238_1: "f32[384]", arg239_1: "f32[384]", arg240_1: "f32[384]", arg241_1: "f32[768, 384]", arg242_1: "f32[768]", arg243_1: "f32[768]", arg244_1: "f32[768]", arg245_1: "f32[768]", arg246_1: "f32[384, 768]", arg247_1: "f32[384]", arg248_1: "f32[384]", arg249_1: "f32[384]", arg250_1: "f32[384]", arg251_1: "f32[768, 384]", arg252_1: "f32[768]", arg253_1: "f32[768]", arg254_1: "f32[768]", arg255_1: "f32[768]", arg256_1: "f32[12, 16]", arg257_1: "i64[16, 16]", arg258_1: "f32[384, 384]", arg259_1: "f32[384]", arg260_1: "f32[384]", arg261_1: "f32[384]", arg262_1: "f32[384]", arg263_1: "f32[768, 384]", arg264_1: "f32[768]", arg265_1: "f32[768]", arg266_1: "f32[768]", arg267_1: "f32[768]", arg268_1: "f32[384, 768]", arg269_1: "f32[384]", arg270_1: "f32[384]", arg271_1: "f32[384]", arg272_1: "f32[384]", arg273_1: "f32[768, 384]", arg274_1: "f32[768]", arg275_1: "f32[768]", arg276_1: "f32[768]", arg277_1: "f32[768]", arg278_1: "f32[12, 16]", arg279_1: "i64[16, 16]", arg280_1: "f32[384, 384]", arg281_1: "f32[384]", arg282_1: "f32[384]", arg283_1: "f32[384]", arg284_1: "f32[384]", arg285_1: "f32[768, 384]", arg286_1: "f32[768]", arg287_1: "f32[768]", arg288_1: "f32[768]", arg289_1: "f32[768]", arg290_1: "f32[384, 768]", arg291_1: "f32[384]", arg292_1: "f32[384]", arg293_1: "f32[384]", arg294_1: "f32[384]", arg295_1: "f32[768, 384]", arg296_1: "f32[768]", arg297_1: "f32[768]", arg298_1: "f32[768]", arg299_1: "f32[768]", arg300_1: "f32[12, 16]", arg301_1: "i64[16, 16]", arg302_1: "f32[384, 384]", arg303_1: "f32[384]", arg304_1: "f32[384]", arg305_1: "f32[384]", arg306_1: "f32[384]", arg307_1: "f32[768, 384]", arg308_1: "f32[768]", arg309_1: "f32[768]", arg310_1: "f32[768]", arg311_1: "f32[768]", arg312_1: "f32[384, 768]", arg313_1: "f32[384]", arg314_1: "f32[384]", arg315_1: "f32[384]", arg316_1: "f32[384]", arg317_1: "f32[768, 384]", arg318_1: "f32[768]", arg319_1: "f32[768]", arg320_1: "f32[768]", arg321_1: "f32[768]", arg322_1: "f32[12, 16]", arg323_1: "i64[16, 16]", arg324_1: "f32[384, 384]", arg325_1: "f32[384]", arg326_1: "f32[384]", arg327_1: "f32[384]", arg328_1: "f32[384]", arg329_1: "f32[768, 384]", arg330_1: "f32[768]", arg331_1: "f32[768]", arg332_1: "f32[768]", arg333_1: "f32[768]", arg334_1: "f32[384, 768]", arg335_1: "f32[384]", arg336_1: "f32[384]", arg337_1: "f32[384]", arg338_1: "f32[384]", arg339_1: "f32[384]", arg340_1: "f32[384]", arg341_1: "f32[384]", arg342_1: "f32[384]", arg343_1: "f32[1000, 384]", arg344_1: "f32[1000]", arg345_1: "f32[384]", arg346_1: "f32[384]", arg347_1: "f32[384]", arg348_1: "f32[384]", arg349_1: "f32[1000, 384]", arg350_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:216 in get_attention_biases, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
        index: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(arg26_1, [None, arg27_1]);  arg26_1 = arg27_1 = None
        index_1: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(arg48_1, [None, arg49_1]);  arg48_1 = arg49_1 = None
        index_2: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(arg70_1, [None, arg71_1]);  arg70_1 = arg71_1 = None
        index_3: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(arg92_1, [None, arg93_1]);  arg92_1 = arg93_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:317 in get_attention_biases, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
        index_4: "f32[8, 49, 196]" = torch.ops.aten.index.Tensor(arg119_1, [None, arg120_1]);  arg119_1 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:216 in get_attention_biases, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
        index_5: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(arg141_1, [None, arg142_1]);  arg141_1 = arg142_1 = None
        index_6: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(arg163_1, [None, arg164_1]);  arg163_1 = arg164_1 = None
        index_7: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(arg185_1, [None, arg186_1]);  arg185_1 = arg186_1 = None
        index_8: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(arg207_1, [None, arg208_1]);  arg207_1 = arg208_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:317 in get_attention_biases, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
        index_9: "f32[16, 16, 49]" = torch.ops.aten.index.Tensor(arg234_1, [None, arg235_1]);  arg234_1 = arg235_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:216 in get_attention_biases, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
        index_10: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(arg256_1, [None, arg257_1]);  arg256_1 = arg257_1 = None
        index_11: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(arg278_1, [None, arg279_1]);  arg278_1 = arg279_1 = None
        index_12: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(arg300_1, [None, arg301_1]);  arg300_1 = arg301_1 = None
        index_13: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(arg322_1, [None, arg323_1]);  arg322_1 = arg323_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:66 in forward, code: return self.bn(self.linear(x))
        convolution_4: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_200: "f32[16]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_64: "f32[16]" = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_64: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_237: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_32: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_33: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_35: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        sub_78: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        mul_238: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_35);  sub_78 = unsqueeze_35 = None
        unsqueeze_36: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_37: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_239: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_37);  mul_238 = unsqueeze_37 = None
        unsqueeze_38: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_39: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_201: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_39);  mul_239 = unsqueeze_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:703 in forward_features, code: x = self.stem(x)
        add_202: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_201, 3)
        clamp_min_31: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_202, 0);  add_202 = None
        clamp_max_31: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_31, 6);  clamp_min_31 = None
        mul_240: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_201, clamp_max_31);  add_201 = clamp_max_31 = None
        div_46: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_240, 6);  mul_240 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:66 in forward, code: return self.bn(self.linear(x))
        convolution_5: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div_46, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_46 = arg6_1 = None
        add_203: "f32[32]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_65: "f32[32]" = torch.ops.aten.sqrt.default(add_203);  add_203 = None
        reciprocal_65: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_241: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_40: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_41: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_241, -1);  mul_241 = None
        unsqueeze_43: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        sub_79: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        mul_242: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_43);  sub_79 = unsqueeze_43 = None
        unsqueeze_44: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_45: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_243: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_45);  mul_242 = unsqueeze_45 = None
        unsqueeze_46: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_47: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_204: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_243, unsqueeze_47);  mul_243 = unsqueeze_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:703 in forward_features, code: x = self.stem(x)
        add_205: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_204, 3)
        clamp_min_32: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_205, 0);  add_205 = None
        clamp_max_32: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_32, 6);  clamp_min_32 = None
        mul_244: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_204, clamp_max_32);  add_204 = clamp_max_32 = None
        div_47: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_244, 6);  mul_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:66 in forward, code: return self.bn(self.linear(x))
        convolution_6: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_47, arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_47 = arg11_1 = None
        add_206: "f32[64]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_66: "f32[64]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_66: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_245: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_245, -1);  mul_245 = None
        unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        sub_80: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        mul_246: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_51);  sub_80 = unsqueeze_51 = None
        unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_247: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_53);  mul_246 = unsqueeze_53 = None
        unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_207: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_247, unsqueeze_55);  mul_247 = unsqueeze_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:703 in forward_features, code: x = self.stem(x)
        add_208: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_207, 3)
        clamp_min_33: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_208, 0);  add_208 = None
        clamp_max_33: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_33, 6);  clamp_min_33 = None
        mul_248: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_207, clamp_max_33);  add_207 = clamp_max_33 = None
        div_48: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_248, 6);  mul_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:66 in forward, code: return self.bn(self.linear(x))
        convolution_7: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_48, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  div_48 = arg16_1 = None
        add_209: "f32[128]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_67: "f32[128]" = torch.ops.aten.sqrt.default(add_209);  add_209 = None
        reciprocal_67: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_249: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
        unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        sub_81: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        mul_250: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_59);  sub_81 = unsqueeze_59 = None
        unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_251: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_61);  mul_250 = unsqueeze_61 = None
        unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_210: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_63);  mul_251 = unsqueeze_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:705 in forward_features, code: x = x.flatten(2).transpose(1, 2)
        view_351: "f32[8, 128, 196]" = torch.ops.aten.view.default(add_210, [8, 128, 196]);  add_210 = None
        permute_117: "f32[8, 196, 128]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_118: "f32[128, 256]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        clone_83: "f32[8, 196, 128]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format)
        view_352: "f32[1568, 128]" = torch.ops.aten.view.default(clone_83, [1568, 128]);  clone_83 = None
        mm_58: "f32[1568, 256]" = torch.ops.aten.mm.default(view_352, permute_118);  view_352 = permute_118 = None
        view_353: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_58, [8, 196, 256]);  mm_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_354: "f32[1568, 256]" = torch.ops.aten.view.default(view_353, [1568, 256]);  view_353 = None
        add_211: "f32[256]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_68: "f32[256]" = torch.ops.aten.sqrt.default(add_211);  add_211 = None
        reciprocal_68: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_252: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        sub_82: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_354, arg22_1);  view_354 = arg22_1 = None
        mul_253: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_82, mul_252);  sub_82 = mul_252 = None
        mul_254: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_253, arg24_1);  mul_253 = arg24_1 = None
        add_212: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_254, arg25_1);  mul_254 = arg25_1 = None
        view_355: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_212, [8, 196, 256]);  add_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_356: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_355, [8, 196, 4, -1]);  view_355 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(view_356, [16, 16, 32], 3);  view_356 = None
        getitem_40: "f32[8, 196, 4, 16]" = split_with_sizes_14[0]
        getitem_41: "f32[8, 196, 4, 16]" = split_with_sizes_14[1]
        getitem_42: "f32[8, 196, 4, 32]" = split_with_sizes_14[2];  split_with_sizes_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_119: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_40, [0, 2, 1, 3]);  getitem_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_120: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_41, [0, 2, 3, 1]);  getitem_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_121: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_56: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_119, [8, 4, 196, 16]);  permute_119 = None
        clone_84: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_357: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_84, [32, 196, 16]);  clone_84 = None
        expand_57: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_120, [8, 4, 16, 196]);  permute_120 = None
        clone_85: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_358: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_85, [32, 16, 196]);  clone_85 = None
        bmm_28: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_357, view_358);  view_357 = view_358 = None
        view_359: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_28, [8, 4, 196, 196]);  bmm_28 = None
        mul_255: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_359, 0.25);  view_359 = None
        add_213: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_255, index);  mul_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_14: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_213, [-1], True)
        sub_83: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_213, amax_14);  add_213 = amax_14 = None
        exp_14: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
        sum_15: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_49: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_58: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_49, [8, 4, 196, 196]);  div_49 = None
        view_360: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_58, [32, 196, 196]);  expand_58 = None
        expand_59: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_121, [8, 4, 196, 32]);  permute_121 = None
        clone_86: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_361: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_86, [32, 196, 32]);  clone_86 = None
        bmm_29: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_360, view_361);  view_360 = view_361 = None
        view_362: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_29, [8, 4, 196, 32]);  bmm_29 = None
        permute_122: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
        clone_87: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_363: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_87, [8, 196, 128]);  clone_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_214: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_363, 3)
        clamp_min_34: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_214, 0);  add_214 = None
        clamp_max_34: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_34, 6);  clamp_min_34 = None
        mul_256: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_363, clamp_max_34);  view_363 = clamp_max_34 = None
        div_50: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_256, 6);  mul_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_123: "f32[128, 128]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        view_364: "f32[1568, 128]" = torch.ops.aten.view.default(div_50, [1568, 128]);  div_50 = None
        mm_59: "f32[1568, 128]" = torch.ops.aten.mm.default(view_364, permute_123);  view_364 = permute_123 = None
        view_365: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_59, [8, 196, 128]);  mm_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_366: "f32[1568, 128]" = torch.ops.aten.view.default(view_365, [1568, 128]);  view_365 = None
        add_215: "f32[128]" = torch.ops.aten.add.Tensor(arg30_1, 1e-05);  arg30_1 = None
        sqrt_69: "f32[128]" = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_69: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_257: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        sub_84: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_366, arg29_1);  view_366 = arg29_1 = None
        mul_258: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_84, mul_257);  sub_84 = mul_257 = None
        mul_259: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_258, arg31_1);  mul_258 = arg31_1 = None
        add_216: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_259, arg32_1);  mul_259 = arg32_1 = None
        view_367: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_216, [8, 196, 128]);  add_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_217: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(permute_117, view_367);  permute_117 = view_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_124: "f32[128, 256]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        clone_88: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_217, memory_format = torch.contiguous_format)
        view_368: "f32[1568, 128]" = torch.ops.aten.view.default(clone_88, [1568, 128]);  clone_88 = None
        mm_60: "f32[1568, 256]" = torch.ops.aten.mm.default(view_368, permute_124);  view_368 = permute_124 = None
        view_369: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_60, [8, 196, 256]);  mm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_370: "f32[1568, 256]" = torch.ops.aten.view.default(view_369, [1568, 256]);  view_369 = None
        add_218: "f32[256]" = torch.ops.aten.add.Tensor(arg35_1, 1e-05);  arg35_1 = None
        sqrt_70: "f32[256]" = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_70: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_260: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        sub_85: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_370, arg34_1);  view_370 = arg34_1 = None
        mul_261: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_85, mul_260);  sub_85 = mul_260 = None
        mul_262: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_261, arg36_1);  mul_261 = arg36_1 = None
        add_219: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_262, arg37_1);  mul_262 = arg37_1 = None
        view_371: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_219, [8, 196, 256]);  add_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_220: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_371, 3)
        clamp_min_35: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_220, 0);  add_220 = None
        clamp_max_35: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_35, 6);  clamp_min_35 = None
        mul_263: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_371, clamp_max_35);  view_371 = clamp_max_35 = None
        div_51: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_263, 6);  mul_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_125: "f32[256, 128]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        view_372: "f32[1568, 256]" = torch.ops.aten.view.default(div_51, [1568, 256]);  div_51 = None
        mm_61: "f32[1568, 128]" = torch.ops.aten.mm.default(view_372, permute_125);  view_372 = permute_125 = None
        view_373: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_61, [8, 196, 128]);  mm_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_374: "f32[1568, 128]" = torch.ops.aten.view.default(view_373, [1568, 128]);  view_373 = None
        add_221: "f32[128]" = torch.ops.aten.add.Tensor(arg40_1, 1e-05);  arg40_1 = None
        sqrt_71: "f32[128]" = torch.ops.aten.sqrt.default(add_221);  add_221 = None
        reciprocal_71: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_264: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        sub_86: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_374, arg39_1);  view_374 = arg39_1 = None
        mul_265: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_86, mul_264);  sub_86 = mul_264 = None
        mul_266: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_265, arg41_1);  mul_265 = arg41_1 = None
        add_222: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_266, arg42_1);  mul_266 = arg42_1 = None
        view_375: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_222, [8, 196, 128]);  add_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_223: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_217, view_375);  add_217 = view_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_126: "f32[128, 256]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        clone_90: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_223, memory_format = torch.contiguous_format)
        view_376: "f32[1568, 128]" = torch.ops.aten.view.default(clone_90, [1568, 128]);  clone_90 = None
        mm_62: "f32[1568, 256]" = torch.ops.aten.mm.default(view_376, permute_126);  view_376 = permute_126 = None
        view_377: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_62, [8, 196, 256]);  mm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_378: "f32[1568, 256]" = torch.ops.aten.view.default(view_377, [1568, 256]);  view_377 = None
        add_224: "f32[256]" = torch.ops.aten.add.Tensor(arg45_1, 1e-05);  arg45_1 = None
        sqrt_72: "f32[256]" = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_72: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_267: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        sub_87: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_378, arg44_1);  view_378 = arg44_1 = None
        mul_268: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_87, mul_267);  sub_87 = mul_267 = None
        mul_269: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_268, arg46_1);  mul_268 = arg46_1 = None
        add_225: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_269, arg47_1);  mul_269 = arg47_1 = None
        view_379: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_225, [8, 196, 256]);  add_225 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_380: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_379, [8, 196, 4, -1]);  view_379 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(view_380, [16, 16, 32], 3);  view_380 = None
        getitem_43: "f32[8, 196, 4, 16]" = split_with_sizes_15[0]
        getitem_44: "f32[8, 196, 4, 16]" = split_with_sizes_15[1]
        getitem_45: "f32[8, 196, 4, 32]" = split_with_sizes_15[2];  split_with_sizes_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_127: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_43, [0, 2, 1, 3]);  getitem_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_128: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_44, [0, 2, 3, 1]);  getitem_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_129: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_60: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_127, [8, 4, 196, 16]);  permute_127 = None
        clone_91: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_381: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_91, [32, 196, 16]);  clone_91 = None
        expand_61: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_128, [8, 4, 16, 196]);  permute_128 = None
        clone_92: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_382: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_92, [32, 16, 196]);  clone_92 = None
        bmm_30: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_381, view_382);  view_381 = view_382 = None
        view_383: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_30, [8, 4, 196, 196]);  bmm_30 = None
        mul_270: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_383, 0.25);  view_383 = None
        add_226: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_270, index_1);  mul_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_15: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_226, [-1], True)
        sub_88: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_226, amax_15);  add_226 = amax_15 = None
        exp_15: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_88);  sub_88 = None
        sum_16: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_52: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_62: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_52, [8, 4, 196, 196]);  div_52 = None
        view_384: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_62, [32, 196, 196]);  expand_62 = None
        expand_63: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_129, [8, 4, 196, 32]);  permute_129 = None
        clone_93: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_385: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_93, [32, 196, 32]);  clone_93 = None
        bmm_31: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_384, view_385);  view_384 = view_385 = None
        view_386: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_31, [8, 4, 196, 32]);  bmm_31 = None
        permute_130: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        clone_94: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_387: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_94, [8, 196, 128]);  clone_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_227: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_387, 3)
        clamp_min_36: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_227, 0);  add_227 = None
        clamp_max_36: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_36, 6);  clamp_min_36 = None
        mul_271: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_387, clamp_max_36);  view_387 = clamp_max_36 = None
        div_53: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_271, 6);  mul_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_131: "f32[128, 128]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        view_388: "f32[1568, 128]" = torch.ops.aten.view.default(div_53, [1568, 128]);  div_53 = None
        mm_63: "f32[1568, 128]" = torch.ops.aten.mm.default(view_388, permute_131);  view_388 = permute_131 = None
        view_389: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_63, [8, 196, 128]);  mm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_390: "f32[1568, 128]" = torch.ops.aten.view.default(view_389, [1568, 128]);  view_389 = None
        add_228: "f32[128]" = torch.ops.aten.add.Tensor(arg52_1, 1e-05);  arg52_1 = None
        sqrt_73: "f32[128]" = torch.ops.aten.sqrt.default(add_228);  add_228 = None
        reciprocal_73: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_272: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        sub_89: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_390, arg51_1);  view_390 = arg51_1 = None
        mul_273: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_89, mul_272);  sub_89 = mul_272 = None
        mul_274: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_273, arg53_1);  mul_273 = arg53_1 = None
        add_229: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_274, arg54_1);  mul_274 = arg54_1 = None
        view_391: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_229, [8, 196, 128]);  add_229 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_230: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_223, view_391);  add_223 = view_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_132: "f32[128, 256]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        clone_95: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_230, memory_format = torch.contiguous_format)
        view_392: "f32[1568, 128]" = torch.ops.aten.view.default(clone_95, [1568, 128]);  clone_95 = None
        mm_64: "f32[1568, 256]" = torch.ops.aten.mm.default(view_392, permute_132);  view_392 = permute_132 = None
        view_393: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_64, [8, 196, 256]);  mm_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_394: "f32[1568, 256]" = torch.ops.aten.view.default(view_393, [1568, 256]);  view_393 = None
        add_231: "f32[256]" = torch.ops.aten.add.Tensor(arg57_1, 1e-05);  arg57_1 = None
        sqrt_74: "f32[256]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_74: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_275: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        sub_90: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_394, arg56_1);  view_394 = arg56_1 = None
        mul_276: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_90, mul_275);  sub_90 = mul_275 = None
        mul_277: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_276, arg58_1);  mul_276 = arg58_1 = None
        add_232: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_277, arg59_1);  mul_277 = arg59_1 = None
        view_395: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_232, [8, 196, 256]);  add_232 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_233: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_395, 3)
        clamp_min_37: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_233, 0);  add_233 = None
        clamp_max_37: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_37, 6);  clamp_min_37 = None
        mul_278: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_395, clamp_max_37);  view_395 = clamp_max_37 = None
        div_54: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_278, 6);  mul_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_133: "f32[256, 128]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        view_396: "f32[1568, 256]" = torch.ops.aten.view.default(div_54, [1568, 256]);  div_54 = None
        mm_65: "f32[1568, 128]" = torch.ops.aten.mm.default(view_396, permute_133);  view_396 = permute_133 = None
        view_397: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_65, [8, 196, 128]);  mm_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_398: "f32[1568, 128]" = torch.ops.aten.view.default(view_397, [1568, 128]);  view_397 = None
        add_234: "f32[128]" = torch.ops.aten.add.Tensor(arg62_1, 1e-05);  arg62_1 = None
        sqrt_75: "f32[128]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_75: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_279: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        sub_91: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_398, arg61_1);  view_398 = arg61_1 = None
        mul_280: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_91, mul_279);  sub_91 = mul_279 = None
        mul_281: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_280, arg63_1);  mul_280 = arg63_1 = None
        add_235: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_281, arg64_1);  mul_281 = arg64_1 = None
        view_399: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_235, [8, 196, 128]);  add_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_236: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_230, view_399);  add_230 = view_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_134: "f32[128, 256]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        clone_97: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_236, memory_format = torch.contiguous_format)
        view_400: "f32[1568, 128]" = torch.ops.aten.view.default(clone_97, [1568, 128]);  clone_97 = None
        mm_66: "f32[1568, 256]" = torch.ops.aten.mm.default(view_400, permute_134);  view_400 = permute_134 = None
        view_401: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_66, [8, 196, 256]);  mm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_402: "f32[1568, 256]" = torch.ops.aten.view.default(view_401, [1568, 256]);  view_401 = None
        add_237: "f32[256]" = torch.ops.aten.add.Tensor(arg67_1, 1e-05);  arg67_1 = None
        sqrt_76: "f32[256]" = torch.ops.aten.sqrt.default(add_237);  add_237 = None
        reciprocal_76: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_282: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        sub_92: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_402, arg66_1);  view_402 = arg66_1 = None
        mul_283: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_92, mul_282);  sub_92 = mul_282 = None
        mul_284: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_283, arg68_1);  mul_283 = arg68_1 = None
        add_238: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_284, arg69_1);  mul_284 = arg69_1 = None
        view_403: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_238, [8, 196, 256]);  add_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_404: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_403, [8, 196, 4, -1]);  view_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(view_404, [16, 16, 32], 3);  view_404 = None
        getitem_46: "f32[8, 196, 4, 16]" = split_with_sizes_16[0]
        getitem_47: "f32[8, 196, 4, 16]" = split_with_sizes_16[1]
        getitem_48: "f32[8, 196, 4, 32]" = split_with_sizes_16[2];  split_with_sizes_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_135: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_46, [0, 2, 1, 3]);  getitem_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_136: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_47, [0, 2, 3, 1]);  getitem_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_137: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_48, [0, 2, 1, 3]);  getitem_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_64: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_135, [8, 4, 196, 16]);  permute_135 = None
        clone_98: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_405: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_98, [32, 196, 16]);  clone_98 = None
        expand_65: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_136, [8, 4, 16, 196]);  permute_136 = None
        clone_99: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_406: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_99, [32, 16, 196]);  clone_99 = None
        bmm_32: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_405, view_406);  view_405 = view_406 = None
        view_407: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_32, [8, 4, 196, 196]);  bmm_32 = None
        mul_285: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_407, 0.25);  view_407 = None
        add_239: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_285, index_2);  mul_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_16: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_239, [-1], True)
        sub_93: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_239, amax_16);  add_239 = amax_16 = None
        exp_16: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_17: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_55: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_66: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_55, [8, 4, 196, 196]);  div_55 = None
        view_408: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_66, [32, 196, 196]);  expand_66 = None
        expand_67: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_137, [8, 4, 196, 32]);  permute_137 = None
        clone_100: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_409: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_100, [32, 196, 32]);  clone_100 = None
        bmm_33: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
        view_410: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_33, [8, 4, 196, 32]);  bmm_33 = None
        permute_138: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_101: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
        view_411: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_101, [8, 196, 128]);  clone_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_240: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_411, 3)
        clamp_min_38: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_240, 0);  add_240 = None
        clamp_max_38: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_38, 6);  clamp_min_38 = None
        mul_286: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_411, clamp_max_38);  view_411 = clamp_max_38 = None
        div_56: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_286, 6);  mul_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_139: "f32[128, 128]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        view_412: "f32[1568, 128]" = torch.ops.aten.view.default(div_56, [1568, 128]);  div_56 = None
        mm_67: "f32[1568, 128]" = torch.ops.aten.mm.default(view_412, permute_139);  view_412 = permute_139 = None
        view_413: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_67, [8, 196, 128]);  mm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_414: "f32[1568, 128]" = torch.ops.aten.view.default(view_413, [1568, 128]);  view_413 = None
        add_241: "f32[128]" = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_77: "f32[128]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_77: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_287: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        sub_94: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_414, arg73_1);  view_414 = arg73_1 = None
        mul_288: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_94, mul_287);  sub_94 = mul_287 = None
        mul_289: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_288, arg75_1);  mul_288 = arg75_1 = None
        add_242: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_289, arg76_1);  mul_289 = arg76_1 = None
        view_415: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_242, [8, 196, 128]);  add_242 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_243: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_236, view_415);  add_236 = view_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_140: "f32[128, 256]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        clone_102: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_243, memory_format = torch.contiguous_format)
        view_416: "f32[1568, 128]" = torch.ops.aten.view.default(clone_102, [1568, 128]);  clone_102 = None
        mm_68: "f32[1568, 256]" = torch.ops.aten.mm.default(view_416, permute_140);  view_416 = permute_140 = None
        view_417: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_68, [8, 196, 256]);  mm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_418: "f32[1568, 256]" = torch.ops.aten.view.default(view_417, [1568, 256]);  view_417 = None
        add_244: "f32[256]" = torch.ops.aten.add.Tensor(arg79_1, 1e-05);  arg79_1 = None
        sqrt_78: "f32[256]" = torch.ops.aten.sqrt.default(add_244);  add_244 = None
        reciprocal_78: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_290: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        sub_95: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_418, arg78_1);  view_418 = arg78_1 = None
        mul_291: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_95, mul_290);  sub_95 = mul_290 = None
        mul_292: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_291, arg80_1);  mul_291 = arg80_1 = None
        add_245: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_292, arg81_1);  mul_292 = arg81_1 = None
        view_419: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_245, [8, 196, 256]);  add_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_246: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_419, 3)
        clamp_min_39: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_246, 0);  add_246 = None
        clamp_max_39: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_39, 6);  clamp_min_39 = None
        mul_293: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_419, clamp_max_39);  view_419 = clamp_max_39 = None
        div_57: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_293, 6);  mul_293 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_141: "f32[256, 128]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        view_420: "f32[1568, 256]" = torch.ops.aten.view.default(div_57, [1568, 256]);  div_57 = None
        mm_69: "f32[1568, 128]" = torch.ops.aten.mm.default(view_420, permute_141);  view_420 = permute_141 = None
        view_421: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_69, [8, 196, 128]);  mm_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_422: "f32[1568, 128]" = torch.ops.aten.view.default(view_421, [1568, 128]);  view_421 = None
        add_247: "f32[128]" = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_79: "f32[128]" = torch.ops.aten.sqrt.default(add_247);  add_247 = None
        reciprocal_79: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_294: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        sub_96: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_422, arg83_1);  view_422 = arg83_1 = None
        mul_295: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_96, mul_294);  sub_96 = mul_294 = None
        mul_296: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_295, arg85_1);  mul_295 = arg85_1 = None
        add_248: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_296, arg86_1);  mul_296 = arg86_1 = None
        view_423: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_248, [8, 196, 128]);  add_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_249: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_243, view_423);  add_243 = view_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_142: "f32[128, 256]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        clone_104: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_249, memory_format = torch.contiguous_format)
        view_424: "f32[1568, 128]" = torch.ops.aten.view.default(clone_104, [1568, 128]);  clone_104 = None
        mm_70: "f32[1568, 256]" = torch.ops.aten.mm.default(view_424, permute_142);  view_424 = permute_142 = None
        view_425: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_70, [8, 196, 256]);  mm_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_426: "f32[1568, 256]" = torch.ops.aten.view.default(view_425, [1568, 256]);  view_425 = None
        add_250: "f32[256]" = torch.ops.aten.add.Tensor(arg89_1, 1e-05);  arg89_1 = None
        sqrt_80: "f32[256]" = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_80: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_297: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        sub_97: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_426, arg88_1);  view_426 = arg88_1 = None
        mul_298: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_97, mul_297);  sub_97 = mul_297 = None
        mul_299: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_298, arg90_1);  mul_298 = arg90_1 = None
        add_251: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_299, arg91_1);  mul_299 = arg91_1 = None
        view_427: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_251, [8, 196, 256]);  add_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_428: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_427, [8, 196, 4, -1]);  view_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(view_428, [16, 16, 32], 3);  view_428 = None
        getitem_49: "f32[8, 196, 4, 16]" = split_with_sizes_17[0]
        getitem_50: "f32[8, 196, 4, 16]" = split_with_sizes_17[1]
        getitem_51: "f32[8, 196, 4, 32]" = split_with_sizes_17[2];  split_with_sizes_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_143: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_144: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 3, 1]);  getitem_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_145: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_51, [0, 2, 1, 3]);  getitem_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_68: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_143, [8, 4, 196, 16]);  permute_143 = None
        clone_105: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_429: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_105, [32, 196, 16]);  clone_105 = None
        expand_69: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_144, [8, 4, 16, 196]);  permute_144 = None
        clone_106: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_430: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_106, [32, 16, 196]);  clone_106 = None
        bmm_34: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_429, view_430);  view_429 = view_430 = None
        view_431: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_34, [8, 4, 196, 196]);  bmm_34 = None
        mul_300: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_431, 0.25);  view_431 = None
        add_252: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_300, index_3);  mul_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_17: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_252, [-1], True)
        sub_98: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_252, amax_17);  add_252 = amax_17 = None
        exp_17: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_98);  sub_98 = None
        sum_18: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_58: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_70: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_58, [8, 4, 196, 196]);  div_58 = None
        view_432: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_70, [32, 196, 196]);  expand_70 = None
        expand_71: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_145, [8, 4, 196, 32]);  permute_145 = None
        clone_107: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_433: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_107, [32, 196, 32]);  clone_107 = None
        bmm_35: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_432, view_433);  view_432 = view_433 = None
        view_434: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_35, [8, 4, 196, 32]);  bmm_35 = None
        permute_146: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        clone_108: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_435: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_108, [8, 196, 128]);  clone_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_253: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_435, 3)
        clamp_min_40: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_253, 0);  add_253 = None
        clamp_max_40: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_40, 6);  clamp_min_40 = None
        mul_301: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_435, clamp_max_40);  view_435 = clamp_max_40 = None
        div_59: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_301, 6);  mul_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_147: "f32[128, 128]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        view_436: "f32[1568, 128]" = torch.ops.aten.view.default(div_59, [1568, 128]);  div_59 = None
        mm_71: "f32[1568, 128]" = torch.ops.aten.mm.default(view_436, permute_147);  view_436 = permute_147 = None
        view_437: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_71, [8, 196, 128]);  mm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_438: "f32[1568, 128]" = torch.ops.aten.view.default(view_437, [1568, 128]);  view_437 = None
        add_254: "f32[128]" = torch.ops.aten.add.Tensor(arg96_1, 1e-05);  arg96_1 = None
        sqrt_81: "f32[128]" = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_81: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_302: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        sub_99: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_438, arg95_1);  view_438 = arg95_1 = None
        mul_303: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_99, mul_302);  sub_99 = mul_302 = None
        mul_304: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_303, arg97_1);  mul_303 = arg97_1 = None
        add_255: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_304, arg98_1);  mul_304 = arg98_1 = None
        view_439: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_255, [8, 196, 128]);  add_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_256: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_249, view_439);  add_249 = view_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_148: "f32[128, 256]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        clone_109: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_256, memory_format = torch.contiguous_format)
        view_440: "f32[1568, 128]" = torch.ops.aten.view.default(clone_109, [1568, 128]);  clone_109 = None
        mm_72: "f32[1568, 256]" = torch.ops.aten.mm.default(view_440, permute_148);  view_440 = permute_148 = None
        view_441: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_72, [8, 196, 256]);  mm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_442: "f32[1568, 256]" = torch.ops.aten.view.default(view_441, [1568, 256]);  view_441 = None
        add_257: "f32[256]" = torch.ops.aten.add.Tensor(arg101_1, 1e-05);  arg101_1 = None
        sqrt_82: "f32[256]" = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_82: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_305: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        sub_100: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_442, arg100_1);  view_442 = arg100_1 = None
        mul_306: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_100, mul_305);  sub_100 = mul_305 = None
        mul_307: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_306, arg102_1);  mul_306 = arg102_1 = None
        add_258: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_307, arg103_1);  mul_307 = arg103_1 = None
        view_443: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_258, [8, 196, 256]);  add_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_259: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_443, 3)
        clamp_min_41: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_259, 0);  add_259 = None
        clamp_max_41: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_41, 6);  clamp_min_41 = None
        mul_308: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_443, clamp_max_41);  view_443 = clamp_max_41 = None
        div_60: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_308, 6);  mul_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_149: "f32[256, 128]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        view_444: "f32[1568, 256]" = torch.ops.aten.view.default(div_60, [1568, 256]);  div_60 = None
        mm_73: "f32[1568, 128]" = torch.ops.aten.mm.default(view_444, permute_149);  view_444 = permute_149 = None
        view_445: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_73, [8, 196, 128]);  mm_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_446: "f32[1568, 128]" = torch.ops.aten.view.default(view_445, [1568, 128]);  view_445 = None
        add_260: "f32[128]" = torch.ops.aten.add.Tensor(arg106_1, 1e-05);  arg106_1 = None
        sqrt_83: "f32[128]" = torch.ops.aten.sqrt.default(add_260);  add_260 = None
        reciprocal_83: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_309: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        sub_101: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_446, arg105_1);  view_446 = arg105_1 = None
        mul_310: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_101, mul_309);  sub_101 = mul_309 = None
        mul_311: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_310, arg107_1);  mul_310 = arg107_1 = None
        add_261: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_311, arg108_1);  mul_311 = arg108_1 = None
        view_447: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_261, [8, 196, 128]);  add_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_262: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_256, view_447);  add_256 = view_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_150: "f32[128, 640]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        clone_111: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_262, memory_format = torch.contiguous_format)
        view_448: "f32[1568, 128]" = torch.ops.aten.view.default(clone_111, [1568, 128]);  clone_111 = None
        mm_74: "f32[1568, 640]" = torch.ops.aten.mm.default(view_448, permute_150);  view_448 = permute_150 = None
        view_449: "f32[8, 196, 640]" = torch.ops.aten.view.default(mm_74, [8, 196, 640]);  mm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_450: "f32[1568, 640]" = torch.ops.aten.view.default(view_449, [1568, 640]);  view_449 = None
        add_263: "f32[640]" = torch.ops.aten.add.Tensor(arg111_1, 1e-05);  arg111_1 = None
        sqrt_84: "f32[640]" = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_84: "f32[640]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_312: "f32[640]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        sub_102: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_450, arg110_1);  view_450 = arg110_1 = None
        mul_313: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(sub_102, mul_312);  sub_102 = mul_312 = None
        mul_314: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(mul_313, arg112_1);  mul_313 = arg112_1 = None
        add_264: "f32[1568, 640]" = torch.ops.aten.add.Tensor(mul_314, arg113_1);  mul_314 = arg113_1 = None
        view_451: "f32[8, 196, 640]" = torch.ops.aten.view.default(add_264, [8, 196, 640]);  add_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:333 in forward, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
        view_452: "f32[8, 196, 8, 80]" = torch.ops.aten.view.default(view_451, [8, 196, 8, -1]);  view_451 = None
        split_with_sizes_18 = torch.ops.aten.split_with_sizes.default(view_452, [16, 64], 3);  view_452 = None
        getitem_52: "f32[8, 196, 8, 16]" = split_with_sizes_18[0]
        getitem_53: "f32[8, 196, 8, 64]" = split_with_sizes_18[1];  split_with_sizes_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:334 in forward, code: k = k.permute(0, 2, 3, 1)  # BHCN
        permute_151: "f32[8, 8, 16, 196]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 3, 1]);  getitem_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:335 in forward, code: v = v.permute(0, 2, 1, 3)  # BHNC
        permute_152: "f32[8, 8, 196, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:158 in forward, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
        view_453: "f32[8, 14, 14, 128]" = torch.ops.aten.view.default(add_262, [8, 14, 14, 128]);  add_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:162 in forward, code: x = x[:, ::self.stride, ::self.stride]
        slice_22: "f32[8, 7, 14, 128]" = torch.ops.aten.slice.Tensor(view_453, 1, 0, 9223372036854775807, 2);  view_453 = None
        slice_23: "f32[8, 7, 7, 128]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 9223372036854775807, 2);  slice_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:163 in forward, code: return x.reshape(B, -1, C)
        clone_112: "f32[8, 7, 7, 128]" = torch.ops.aten.clone.default(slice_23, memory_format = torch.contiguous_format);  slice_23 = None
        view_454: "f32[8, 49, 128]" = torch.ops.aten.view.default(clone_112, [8, 49, 128]);  clone_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_153: "f32[128, 128]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        view_455: "f32[392, 128]" = torch.ops.aten.view.default(view_454, [392, 128]);  view_454 = None
        mm_75: "f32[392, 128]" = torch.ops.aten.mm.default(view_455, permute_153);  view_455 = permute_153 = None
        view_456: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_75, [8, 49, 128]);  mm_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_457: "f32[392, 128]" = torch.ops.aten.view.default(view_456, [392, 128]);  view_456 = None
        add_265: "f32[128]" = torch.ops.aten.add.Tensor(arg116_1, 1e-05);  arg116_1 = None
        sqrt_85: "f32[128]" = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_85: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_315: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        sub_103: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_457, arg115_1);  view_457 = arg115_1 = None
        mul_316: "f32[392, 128]" = torch.ops.aten.mul.Tensor(sub_103, mul_315);  sub_103 = mul_315 = None
        mul_317: "f32[392, 128]" = torch.ops.aten.mul.Tensor(mul_316, arg117_1);  mul_316 = arg117_1 = None
        add_266: "f32[392, 128]" = torch.ops.aten.add.Tensor(mul_317, arg118_1);  mul_317 = arg118_1 = None
        view_458: "f32[8, 49, 128]" = torch.ops.aten.view.default(add_266, [8, 49, 128]);  add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:336 in forward, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        view_459: "f32[8, 49, 8, 16]" = torch.ops.aten.view.default(view_458, [8, -1, 8, 16]);  view_458 = None
        permute_154: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:338 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_72: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_154, [8, 8, 49, 16]);  permute_154 = None
        clone_113: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_460: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_113, [64, 49, 16]);  clone_113 = None
        expand_73: "f32[8, 8, 16, 196]" = torch.ops.aten.expand.default(permute_151, [8, 8, 16, 196]);  permute_151 = None
        clone_114: "f32[8, 8, 16, 196]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_461: "f32[64, 16, 196]" = torch.ops.aten.view.default(clone_114, [64, 16, 196]);  clone_114 = None
        bmm_36: "f32[64, 49, 196]" = torch.ops.aten.bmm.default(view_460, view_461);  view_460 = view_461 = None
        view_462: "f32[8, 8, 49, 196]" = torch.ops.aten.view.default(bmm_36, [8, 8, 49, 196]);  bmm_36 = None
        mul_318: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(view_462, 0.25);  view_462 = None
        add_267: "f32[8, 8, 49, 196]" = torch.ops.aten.add.Tensor(mul_318, index_4);  mul_318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:339 in forward, code: attn = attn.softmax(dim=-1)
        amax_18: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_267, [-1], True)
        sub_104: "f32[8, 8, 49, 196]" = torch.ops.aten.sub.Tensor(add_267, amax_18);  add_267 = amax_18 = None
        exp_18: "f32[8, 8, 49, 196]" = torch.ops.aten.exp.default(sub_104);  sub_104 = None
        sum_19: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_61: "f32[8, 8, 49, 196]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:341 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
        expand_74: "f32[8, 8, 49, 196]" = torch.ops.aten.expand.default(div_61, [8, 8, 49, 196]);  div_61 = None
        view_463: "f32[64, 49, 196]" = torch.ops.aten.view.default(expand_74, [64, 49, 196]);  expand_74 = None
        expand_75: "f32[8, 8, 196, 64]" = torch.ops.aten.expand.default(permute_152, [8, 8, 196, 64]);  permute_152 = None
        clone_115: "f32[8, 8, 196, 64]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_464: "f32[64, 196, 64]" = torch.ops.aten.view.default(clone_115, [64, 196, 64]);  clone_115 = None
        bmm_37: "f32[64, 49, 64]" = torch.ops.aten.bmm.default(view_463, view_464);  view_463 = view_464 = None
        view_465: "f32[8, 8, 49, 64]" = torch.ops.aten.view.default(bmm_37, [8, 8, 49, 64]);  bmm_37 = None
        permute_155: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        clone_116: "f32[8, 49, 8, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_466: "f32[8, 49, 512]" = torch.ops.aten.view.default(clone_116, [8, 49, 512]);  clone_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:342 in forward, code: x = self.proj(x)
        add_268: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_466, 3)
        clamp_min_42: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_268, 0);  add_268 = None
        clamp_max_42: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_42, 6);  clamp_min_42 = None
        mul_319: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_466, clamp_max_42);  view_466 = clamp_max_42 = None
        div_62: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_319, 6);  mul_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_156: "f32[512, 256]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        view_467: "f32[392, 512]" = torch.ops.aten.view.default(div_62, [392, 512]);  div_62 = None
        mm_76: "f32[392, 256]" = torch.ops.aten.mm.default(view_467, permute_156);  view_467 = permute_156 = None
        view_468: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_76, [8, 49, 256]);  mm_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_469: "f32[392, 256]" = torch.ops.aten.view.default(view_468, [392, 256]);  view_468 = None
        add_269: "f32[256]" = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_86: "f32[256]" = torch.ops.aten.sqrt.default(add_269);  add_269 = None
        reciprocal_86: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_320: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        sub_105: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_469, arg122_1);  view_469 = arg122_1 = None
        mul_321: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_105, mul_320);  sub_105 = mul_320 = None
        mul_322: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_321, arg124_1);  mul_321 = arg124_1 = None
        add_270: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_322, arg125_1);  mul_322 = arg125_1 = None
        view_470: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_270, [8, 49, 256]);  add_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_157: "f32[256, 512]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        view_471: "f32[392, 256]" = torch.ops.aten.view.default(view_470, [392, 256])
        mm_77: "f32[392, 512]" = torch.ops.aten.mm.default(view_471, permute_157);  view_471 = permute_157 = None
        view_472: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_77, [8, 49, 512]);  mm_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_473: "f32[392, 512]" = torch.ops.aten.view.default(view_472, [392, 512]);  view_472 = None
        add_271: "f32[512]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_87: "f32[512]" = torch.ops.aten.sqrt.default(add_271);  add_271 = None
        reciprocal_87: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_323: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        sub_106: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_473, arg127_1);  view_473 = arg127_1 = None
        mul_324: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_106, mul_323);  sub_106 = mul_323 = None
        mul_325: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_324, arg129_1);  mul_324 = arg129_1 = None
        add_272: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_325, arg130_1);  mul_325 = arg130_1 = None
        view_474: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_272, [8, 49, 512]);  add_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_273: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_474, 3)
        clamp_min_43: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_273, 0);  add_273 = None
        clamp_max_43: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_43, 6);  clamp_min_43 = None
        mul_326: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_474, clamp_max_43);  view_474 = clamp_max_43 = None
        div_63: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_326, 6);  mul_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_158: "f32[512, 256]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        view_475: "f32[392, 512]" = torch.ops.aten.view.default(div_63, [392, 512]);  div_63 = None
        mm_78: "f32[392, 256]" = torch.ops.aten.mm.default(view_475, permute_158);  view_475 = permute_158 = None
        view_476: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_78, [8, 49, 256]);  mm_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_477: "f32[392, 256]" = torch.ops.aten.view.default(view_476, [392, 256]);  view_476 = None
        add_274: "f32[256]" = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_88: "f32[256]" = torch.ops.aten.sqrt.default(add_274);  add_274 = None
        reciprocal_88: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_327: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        sub_107: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_477, arg132_1);  view_477 = arg132_1 = None
        mul_328: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_107, mul_327);  sub_107 = mul_327 = None
        mul_329: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_328, arg134_1);  mul_328 = arg134_1 = None
        add_275: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_329, arg135_1);  mul_329 = arg135_1 = None
        view_478: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_275, [8, 49, 256]);  add_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:417 in forward, code: x = x + self.drop_path(self.mlp(x))
        add_276: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_470, view_478);  view_470 = view_478 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_159: "f32[256, 512]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        view_479: "f32[392, 256]" = torch.ops.aten.view.default(add_276, [392, 256])
        mm_79: "f32[392, 512]" = torch.ops.aten.mm.default(view_479, permute_159);  view_479 = permute_159 = None
        view_480: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_79, [8, 49, 512]);  mm_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_481: "f32[392, 512]" = torch.ops.aten.view.default(view_480, [392, 512]);  view_480 = None
        add_277: "f32[512]" = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_89: "f32[512]" = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_89: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_330: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        sub_108: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_481, arg137_1);  view_481 = arg137_1 = None
        mul_331: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_108, mul_330);  sub_108 = mul_330 = None
        mul_332: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_331, arg139_1);  mul_331 = arg139_1 = None
        add_278: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_332, arg140_1);  mul_332 = arg140_1 = None
        view_482: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_278, [8, 49, 512]);  add_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_483: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_482, [8, 49, 8, -1]);  view_482 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(view_483, [16, 16, 32], 3);  view_483 = None
        getitem_54: "f32[8, 49, 8, 16]" = split_with_sizes_19[0]
        getitem_55: "f32[8, 49, 8, 16]" = split_with_sizes_19[1]
        getitem_56: "f32[8, 49, 8, 32]" = split_with_sizes_19[2];  split_with_sizes_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_160: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_161: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 3, 1]);  getitem_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_162: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_76: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_160, [8, 8, 49, 16]);  permute_160 = None
        clone_118: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_484: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_118, [64, 49, 16]);  clone_118 = None
        expand_77: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_161, [8, 8, 16, 49]);  permute_161 = None
        clone_119: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_485: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_119, [64, 16, 49]);  clone_119 = None
        bmm_38: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_484, view_485);  view_484 = view_485 = None
        view_486: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_38, [8, 8, 49, 49]);  bmm_38 = None
        mul_333: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_486, 0.25);  view_486 = None
        add_279: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_333, index_5);  mul_333 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_19: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_279, [-1], True)
        sub_109: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_279, amax_19);  add_279 = amax_19 = None
        exp_19: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_109);  sub_109 = None
        sum_20: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_64: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_78: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_64, [8, 8, 49, 49]);  div_64 = None
        view_487: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_78, [64, 49, 49]);  expand_78 = None
        expand_79: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_162, [8, 8, 49, 32]);  permute_162 = None
        clone_120: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_488: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_120, [64, 49, 32]);  clone_120 = None
        bmm_39: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_487, view_488);  view_487 = view_488 = None
        view_489: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_39, [8, 8, 49, 32]);  bmm_39 = None
        permute_163: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
        clone_121: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
        view_490: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_121, [8, 49, 256]);  clone_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_280: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_490, 3)
        clamp_min_44: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_280, 0);  add_280 = None
        clamp_max_44: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_44, 6);  clamp_min_44 = None
        mul_334: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_490, clamp_max_44);  view_490 = clamp_max_44 = None
        div_65: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_334, 6);  mul_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_164: "f32[256, 256]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        view_491: "f32[392, 256]" = torch.ops.aten.view.default(div_65, [392, 256]);  div_65 = None
        mm_80: "f32[392, 256]" = torch.ops.aten.mm.default(view_491, permute_164);  view_491 = permute_164 = None
        view_492: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_80, [8, 49, 256]);  mm_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_493: "f32[392, 256]" = torch.ops.aten.view.default(view_492, [392, 256]);  view_492 = None
        add_281: "f32[256]" = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_90: "f32[256]" = torch.ops.aten.sqrt.default(add_281);  add_281 = None
        reciprocal_90: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_335: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        sub_110: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_493, arg144_1);  view_493 = arg144_1 = None
        mul_336: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_110, mul_335);  sub_110 = mul_335 = None
        mul_337: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_336, arg146_1);  mul_336 = arg146_1 = None
        add_282: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_337, arg147_1);  mul_337 = arg147_1 = None
        view_494: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_282, [8, 49, 256]);  add_282 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_283: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_276, view_494);  add_276 = view_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_165: "f32[256, 512]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        view_495: "f32[392, 256]" = torch.ops.aten.view.default(add_283, [392, 256])
        mm_81: "f32[392, 512]" = torch.ops.aten.mm.default(view_495, permute_165);  view_495 = permute_165 = None
        view_496: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_81, [8, 49, 512]);  mm_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_497: "f32[392, 512]" = torch.ops.aten.view.default(view_496, [392, 512]);  view_496 = None
        add_284: "f32[512]" = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_91: "f32[512]" = torch.ops.aten.sqrt.default(add_284);  add_284 = None
        reciprocal_91: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_338: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        sub_111: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_497, arg149_1);  view_497 = arg149_1 = None
        mul_339: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_111, mul_338);  sub_111 = mul_338 = None
        mul_340: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_339, arg151_1);  mul_339 = arg151_1 = None
        add_285: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_340, arg152_1);  mul_340 = arg152_1 = None
        view_498: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_285, [8, 49, 512]);  add_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_286: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_498, 3)
        clamp_min_45: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_286, 0);  add_286 = None
        clamp_max_45: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_45, 6);  clamp_min_45 = None
        mul_341: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_498, clamp_max_45);  view_498 = clamp_max_45 = None
        div_66: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_341, 6);  mul_341 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_166: "f32[512, 256]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        view_499: "f32[392, 512]" = torch.ops.aten.view.default(div_66, [392, 512]);  div_66 = None
        mm_82: "f32[392, 256]" = torch.ops.aten.mm.default(view_499, permute_166);  view_499 = permute_166 = None
        view_500: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_82, [8, 49, 256]);  mm_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_501: "f32[392, 256]" = torch.ops.aten.view.default(view_500, [392, 256]);  view_500 = None
        add_287: "f32[256]" = torch.ops.aten.add.Tensor(arg155_1, 1e-05);  arg155_1 = None
        sqrt_92: "f32[256]" = torch.ops.aten.sqrt.default(add_287);  add_287 = None
        reciprocal_92: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_342: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        sub_112: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_501, arg154_1);  view_501 = arg154_1 = None
        mul_343: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_112, mul_342);  sub_112 = mul_342 = None
        mul_344: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_343, arg156_1);  mul_343 = arg156_1 = None
        add_288: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_344, arg157_1);  mul_344 = arg157_1 = None
        view_502: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_288, [8, 49, 256]);  add_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_289: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_283, view_502);  add_283 = view_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_167: "f32[256, 512]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        view_503: "f32[392, 256]" = torch.ops.aten.view.default(add_289, [392, 256])
        mm_83: "f32[392, 512]" = torch.ops.aten.mm.default(view_503, permute_167);  view_503 = permute_167 = None
        view_504: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_83, [8, 49, 512]);  mm_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_505: "f32[392, 512]" = torch.ops.aten.view.default(view_504, [392, 512]);  view_504 = None
        add_290: "f32[512]" = torch.ops.aten.add.Tensor(arg160_1, 1e-05);  arg160_1 = None
        sqrt_93: "f32[512]" = torch.ops.aten.sqrt.default(add_290);  add_290 = None
        reciprocal_93: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_345: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        sub_113: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_505, arg159_1);  view_505 = arg159_1 = None
        mul_346: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_113, mul_345);  sub_113 = mul_345 = None
        mul_347: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_346, arg161_1);  mul_346 = arg161_1 = None
        add_291: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_347, arg162_1);  mul_347 = arg162_1 = None
        view_506: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_291, [8, 49, 512]);  add_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_507: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_506, [8, 49, 8, -1]);  view_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_20 = torch.ops.aten.split_with_sizes.default(view_507, [16, 16, 32], 3);  view_507 = None
        getitem_57: "f32[8, 49, 8, 16]" = split_with_sizes_20[0]
        getitem_58: "f32[8, 49, 8, 16]" = split_with_sizes_20[1]
        getitem_59: "f32[8, 49, 8, 32]" = split_with_sizes_20[2];  split_with_sizes_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_168: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_169: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 3, 1]);  getitem_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_170: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_80: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_168, [8, 8, 49, 16]);  permute_168 = None
        clone_123: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_508: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_123, [64, 49, 16]);  clone_123 = None
        expand_81: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_169, [8, 8, 16, 49]);  permute_169 = None
        clone_124: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_509: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_124, [64, 16, 49]);  clone_124 = None
        bmm_40: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_508, view_509);  view_508 = view_509 = None
        view_510: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_40, [8, 8, 49, 49]);  bmm_40 = None
        mul_348: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_510, 0.25);  view_510 = None
        add_292: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_348, index_6);  mul_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_20: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_292, [-1], True)
        sub_114: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_292, amax_20);  add_292 = amax_20 = None
        exp_20: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_114);  sub_114 = None
        sum_21: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_67: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_82: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_67, [8, 8, 49, 49]);  div_67 = None
        view_511: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_82, [64, 49, 49]);  expand_82 = None
        expand_83: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_170, [8, 8, 49, 32]);  permute_170 = None
        clone_125: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_512: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_125, [64, 49, 32]);  clone_125 = None
        bmm_41: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_511, view_512);  view_511 = view_512 = None
        view_513: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_41, [8, 8, 49, 32]);  bmm_41 = None
        permute_171: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
        clone_126: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_514: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_126, [8, 49, 256]);  clone_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_293: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_514, 3)
        clamp_min_46: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_293, 0);  add_293 = None
        clamp_max_46: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_46, 6);  clamp_min_46 = None
        mul_349: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_514, clamp_max_46);  view_514 = clamp_max_46 = None
        div_68: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_349, 6);  mul_349 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_172: "f32[256, 256]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        view_515: "f32[392, 256]" = torch.ops.aten.view.default(div_68, [392, 256]);  div_68 = None
        mm_84: "f32[392, 256]" = torch.ops.aten.mm.default(view_515, permute_172);  view_515 = permute_172 = None
        view_516: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_84, [8, 49, 256]);  mm_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_517: "f32[392, 256]" = torch.ops.aten.view.default(view_516, [392, 256]);  view_516 = None
        add_294: "f32[256]" = torch.ops.aten.add.Tensor(arg167_1, 1e-05);  arg167_1 = None
        sqrt_94: "f32[256]" = torch.ops.aten.sqrt.default(add_294);  add_294 = None
        reciprocal_94: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_350: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        sub_115: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_517, arg166_1);  view_517 = arg166_1 = None
        mul_351: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_115, mul_350);  sub_115 = mul_350 = None
        mul_352: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_351, arg168_1);  mul_351 = arg168_1 = None
        add_295: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_352, arg169_1);  mul_352 = arg169_1 = None
        view_518: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_295, [8, 49, 256]);  add_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_296: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_289, view_518);  add_289 = view_518 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_173: "f32[256, 512]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        view_519: "f32[392, 256]" = torch.ops.aten.view.default(add_296, [392, 256])
        mm_85: "f32[392, 512]" = torch.ops.aten.mm.default(view_519, permute_173);  view_519 = permute_173 = None
        view_520: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_85, [8, 49, 512]);  mm_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_521: "f32[392, 512]" = torch.ops.aten.view.default(view_520, [392, 512]);  view_520 = None
        add_297: "f32[512]" = torch.ops.aten.add.Tensor(arg172_1, 1e-05);  arg172_1 = None
        sqrt_95: "f32[512]" = torch.ops.aten.sqrt.default(add_297);  add_297 = None
        reciprocal_95: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_353: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        sub_116: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_521, arg171_1);  view_521 = arg171_1 = None
        mul_354: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_116, mul_353);  sub_116 = mul_353 = None
        mul_355: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_354, arg173_1);  mul_354 = arg173_1 = None
        add_298: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_355, arg174_1);  mul_355 = arg174_1 = None
        view_522: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_298, [8, 49, 512]);  add_298 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_299: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_522, 3)
        clamp_min_47: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_299, 0);  add_299 = None
        clamp_max_47: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_47, 6);  clamp_min_47 = None
        mul_356: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_522, clamp_max_47);  view_522 = clamp_max_47 = None
        div_69: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_356, 6);  mul_356 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_174: "f32[512, 256]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        view_523: "f32[392, 512]" = torch.ops.aten.view.default(div_69, [392, 512]);  div_69 = None
        mm_86: "f32[392, 256]" = torch.ops.aten.mm.default(view_523, permute_174);  view_523 = permute_174 = None
        view_524: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_86, [8, 49, 256]);  mm_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_525: "f32[392, 256]" = torch.ops.aten.view.default(view_524, [392, 256]);  view_524 = None
        add_300: "f32[256]" = torch.ops.aten.add.Tensor(arg177_1, 1e-05);  arg177_1 = None
        sqrt_96: "f32[256]" = torch.ops.aten.sqrt.default(add_300);  add_300 = None
        reciprocal_96: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_357: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        sub_117: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_525, arg176_1);  view_525 = arg176_1 = None
        mul_358: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_117, mul_357);  sub_117 = mul_357 = None
        mul_359: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_358, arg178_1);  mul_358 = arg178_1 = None
        add_301: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_359, arg179_1);  mul_359 = arg179_1 = None
        view_526: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_301, [8, 49, 256]);  add_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_302: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_296, view_526);  add_296 = view_526 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_175: "f32[256, 512]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        view_527: "f32[392, 256]" = torch.ops.aten.view.default(add_302, [392, 256])
        mm_87: "f32[392, 512]" = torch.ops.aten.mm.default(view_527, permute_175);  view_527 = permute_175 = None
        view_528: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_87, [8, 49, 512]);  mm_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_529: "f32[392, 512]" = torch.ops.aten.view.default(view_528, [392, 512]);  view_528 = None
        add_303: "f32[512]" = torch.ops.aten.add.Tensor(arg182_1, 1e-05);  arg182_1 = None
        sqrt_97: "f32[512]" = torch.ops.aten.sqrt.default(add_303);  add_303 = None
        reciprocal_97: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_360: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        sub_118: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_529, arg181_1);  view_529 = arg181_1 = None
        mul_361: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_118, mul_360);  sub_118 = mul_360 = None
        mul_362: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_361, arg183_1);  mul_361 = arg183_1 = None
        add_304: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_362, arg184_1);  mul_362 = arg184_1 = None
        view_530: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_304, [8, 49, 512]);  add_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_531: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_530, [8, 49, 8, -1]);  view_530 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(view_531, [16, 16, 32], 3);  view_531 = None
        getitem_60: "f32[8, 49, 8, 16]" = split_with_sizes_21[0]
        getitem_61: "f32[8, 49, 8, 16]" = split_with_sizes_21[1]
        getitem_62: "f32[8, 49, 8, 32]" = split_with_sizes_21[2];  split_with_sizes_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_176: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_177: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_61, [0, 2, 3, 1]);  getitem_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_178: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_84: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_176, [8, 8, 49, 16]);  permute_176 = None
        clone_128: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_532: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_128, [64, 49, 16]);  clone_128 = None
        expand_85: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_177, [8, 8, 16, 49]);  permute_177 = None
        clone_129: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_533: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_129, [64, 16, 49]);  clone_129 = None
        bmm_42: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_532, view_533);  view_532 = view_533 = None
        view_534: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_42, [8, 8, 49, 49]);  bmm_42 = None
        mul_363: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_534, 0.25);  view_534 = None
        add_305: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_363, index_7);  mul_363 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_21: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_305, [-1], True)
        sub_119: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_305, amax_21);  add_305 = amax_21 = None
        exp_21: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_119);  sub_119 = None
        sum_22: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_70: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_86: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_70, [8, 8, 49, 49]);  div_70 = None
        view_535: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_86, [64, 49, 49]);  expand_86 = None
        expand_87: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_178, [8, 8, 49, 32]);  permute_178 = None
        clone_130: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_536: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_130, [64, 49, 32]);  clone_130 = None
        bmm_43: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_535, view_536);  view_535 = view_536 = None
        view_537: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_43, [8, 8, 49, 32]);  bmm_43 = None
        permute_179: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
        clone_131: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_538: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_131, [8, 49, 256]);  clone_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_306: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_538, 3)
        clamp_min_48: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_306, 0);  add_306 = None
        clamp_max_48: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_48, 6);  clamp_min_48 = None
        mul_364: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_538, clamp_max_48);  view_538 = clamp_max_48 = None
        div_71: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_364, 6);  mul_364 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_180: "f32[256, 256]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        view_539: "f32[392, 256]" = torch.ops.aten.view.default(div_71, [392, 256]);  div_71 = None
        mm_88: "f32[392, 256]" = torch.ops.aten.mm.default(view_539, permute_180);  view_539 = permute_180 = None
        view_540: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_88, [8, 49, 256]);  mm_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_541: "f32[392, 256]" = torch.ops.aten.view.default(view_540, [392, 256]);  view_540 = None
        add_307: "f32[256]" = torch.ops.aten.add.Tensor(arg189_1, 1e-05);  arg189_1 = None
        sqrt_98: "f32[256]" = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_98: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_365: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        sub_120: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_541, arg188_1);  view_541 = arg188_1 = None
        mul_366: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_120, mul_365);  sub_120 = mul_365 = None
        mul_367: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_366, arg190_1);  mul_366 = arg190_1 = None
        add_308: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_367, arg191_1);  mul_367 = arg191_1 = None
        view_542: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_308, [8, 49, 256]);  add_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_309: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_302, view_542);  add_302 = view_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_181: "f32[256, 512]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        view_543: "f32[392, 256]" = torch.ops.aten.view.default(add_309, [392, 256])
        mm_89: "f32[392, 512]" = torch.ops.aten.mm.default(view_543, permute_181);  view_543 = permute_181 = None
        view_544: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_89, [8, 49, 512]);  mm_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_545: "f32[392, 512]" = torch.ops.aten.view.default(view_544, [392, 512]);  view_544 = None
        add_310: "f32[512]" = torch.ops.aten.add.Tensor(arg194_1, 1e-05);  arg194_1 = None
        sqrt_99: "f32[512]" = torch.ops.aten.sqrt.default(add_310);  add_310 = None
        reciprocal_99: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_368: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        sub_121: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_545, arg193_1);  view_545 = arg193_1 = None
        mul_369: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_121, mul_368);  sub_121 = mul_368 = None
        mul_370: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_369, arg195_1);  mul_369 = arg195_1 = None
        add_311: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_370, arg196_1);  mul_370 = arg196_1 = None
        view_546: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_311, [8, 49, 512]);  add_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_312: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_546, 3)
        clamp_min_49: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_312, 0);  add_312 = None
        clamp_max_49: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_49, 6);  clamp_min_49 = None
        mul_371: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_546, clamp_max_49);  view_546 = clamp_max_49 = None
        div_72: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_371, 6);  mul_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_182: "f32[512, 256]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        view_547: "f32[392, 512]" = torch.ops.aten.view.default(div_72, [392, 512]);  div_72 = None
        mm_90: "f32[392, 256]" = torch.ops.aten.mm.default(view_547, permute_182);  view_547 = permute_182 = None
        view_548: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_90, [8, 49, 256]);  mm_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_549: "f32[392, 256]" = torch.ops.aten.view.default(view_548, [392, 256]);  view_548 = None
        add_313: "f32[256]" = torch.ops.aten.add.Tensor(arg199_1, 1e-05);  arg199_1 = None
        sqrt_100: "f32[256]" = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_100: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_372: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        sub_122: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_549, arg198_1);  view_549 = arg198_1 = None
        mul_373: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_122, mul_372);  sub_122 = mul_372 = None
        mul_374: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_373, arg200_1);  mul_373 = arg200_1 = None
        add_314: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_374, arg201_1);  mul_374 = arg201_1 = None
        view_550: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_314, [8, 49, 256]);  add_314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_315: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_309, view_550);  add_309 = view_550 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_183: "f32[256, 512]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
        view_551: "f32[392, 256]" = torch.ops.aten.view.default(add_315, [392, 256])
        mm_91: "f32[392, 512]" = torch.ops.aten.mm.default(view_551, permute_183);  view_551 = permute_183 = None
        view_552: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_91, [8, 49, 512]);  mm_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_553: "f32[392, 512]" = torch.ops.aten.view.default(view_552, [392, 512]);  view_552 = None
        add_316: "f32[512]" = torch.ops.aten.add.Tensor(arg204_1, 1e-05);  arg204_1 = None
        sqrt_101: "f32[512]" = torch.ops.aten.sqrt.default(add_316);  add_316 = None
        reciprocal_101: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_375: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        sub_123: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_553, arg203_1);  view_553 = arg203_1 = None
        mul_376: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_123, mul_375);  sub_123 = mul_375 = None
        mul_377: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_376, arg205_1);  mul_376 = arg205_1 = None
        add_317: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_377, arg206_1);  mul_377 = arg206_1 = None
        view_554: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_317, [8, 49, 512]);  add_317 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_555: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_554, [8, 49, 8, -1]);  view_554 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(view_555, [16, 16, 32], 3);  view_555 = None
        getitem_63: "f32[8, 49, 8, 16]" = split_with_sizes_22[0]
        getitem_64: "f32[8, 49, 8, 16]" = split_with_sizes_22[1]
        getitem_65: "f32[8, 49, 8, 32]" = split_with_sizes_22[2];  split_with_sizes_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_184: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_185: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_64, [0, 2, 3, 1]);  getitem_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_186: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_65, [0, 2, 1, 3]);  getitem_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_88: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_184, [8, 8, 49, 16]);  permute_184 = None
        clone_133: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_556: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_133, [64, 49, 16]);  clone_133 = None
        expand_89: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_185, [8, 8, 16, 49]);  permute_185 = None
        clone_134: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_557: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_134, [64, 16, 49]);  clone_134 = None
        bmm_44: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_556, view_557);  view_556 = view_557 = None
        view_558: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_44, [8, 8, 49, 49]);  bmm_44 = None
        mul_378: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_558, 0.25);  view_558 = None
        add_318: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_378, index_8);  mul_378 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_22: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_318, [-1], True)
        sub_124: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_318, amax_22);  add_318 = amax_22 = None
        exp_22: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_124);  sub_124 = None
        sum_23: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_73: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_90: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_73, [8, 8, 49, 49]);  div_73 = None
        view_559: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_90, [64, 49, 49]);  expand_90 = None
        expand_91: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_186, [8, 8, 49, 32]);  permute_186 = None
        clone_135: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_560: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_135, [64, 49, 32]);  clone_135 = None
        bmm_45: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_559, view_560);  view_559 = view_560 = None
        view_561: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_45, [8, 8, 49, 32]);  bmm_45 = None
        permute_187: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        clone_136: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_562: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_136, [8, 49, 256]);  clone_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_319: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_562, 3)
        clamp_min_50: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_319, 0);  add_319 = None
        clamp_max_50: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_50, 6);  clamp_min_50 = None
        mul_379: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_562, clamp_max_50);  view_562 = clamp_max_50 = None
        div_74: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_379, 6);  mul_379 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_188: "f32[256, 256]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        view_563: "f32[392, 256]" = torch.ops.aten.view.default(div_74, [392, 256]);  div_74 = None
        mm_92: "f32[392, 256]" = torch.ops.aten.mm.default(view_563, permute_188);  view_563 = permute_188 = None
        view_564: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_92, [8, 49, 256]);  mm_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_565: "f32[392, 256]" = torch.ops.aten.view.default(view_564, [392, 256]);  view_564 = None
        add_320: "f32[256]" = torch.ops.aten.add.Tensor(arg211_1, 1e-05);  arg211_1 = None
        sqrt_102: "f32[256]" = torch.ops.aten.sqrt.default(add_320);  add_320 = None
        reciprocal_102: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_380: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        sub_125: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_565, arg210_1);  view_565 = arg210_1 = None
        mul_381: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_125, mul_380);  sub_125 = mul_380 = None
        mul_382: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_381, arg212_1);  mul_381 = arg212_1 = None
        add_321: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_382, arg213_1);  mul_382 = arg213_1 = None
        view_566: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_321, [8, 49, 256]);  add_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_322: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_315, view_566);  add_315 = view_566 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_189: "f32[256, 512]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        view_567: "f32[392, 256]" = torch.ops.aten.view.default(add_322, [392, 256])
        mm_93: "f32[392, 512]" = torch.ops.aten.mm.default(view_567, permute_189);  view_567 = permute_189 = None
        view_568: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_93, [8, 49, 512]);  mm_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_569: "f32[392, 512]" = torch.ops.aten.view.default(view_568, [392, 512]);  view_568 = None
        add_323: "f32[512]" = torch.ops.aten.add.Tensor(arg216_1, 1e-05);  arg216_1 = None
        sqrt_103: "f32[512]" = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_103: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_383: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        sub_126: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_569, arg215_1);  view_569 = arg215_1 = None
        mul_384: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_126, mul_383);  sub_126 = mul_383 = None
        mul_385: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_384, arg217_1);  mul_384 = arg217_1 = None
        add_324: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_385, arg218_1);  mul_385 = arg218_1 = None
        view_570: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_324, [8, 49, 512]);  add_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_325: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_570, 3)
        clamp_min_51: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_325, 0);  add_325 = None
        clamp_max_51: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_51, 6);  clamp_min_51 = None
        mul_386: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_570, clamp_max_51);  view_570 = clamp_max_51 = None
        div_75: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_386, 6);  mul_386 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_190: "f32[512, 256]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        view_571: "f32[392, 512]" = torch.ops.aten.view.default(div_75, [392, 512]);  div_75 = None
        mm_94: "f32[392, 256]" = torch.ops.aten.mm.default(view_571, permute_190);  view_571 = permute_190 = None
        view_572: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_94, [8, 49, 256]);  mm_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_573: "f32[392, 256]" = torch.ops.aten.view.default(view_572, [392, 256]);  view_572 = None
        add_326: "f32[256]" = torch.ops.aten.add.Tensor(arg221_1, 1e-05);  arg221_1 = None
        sqrt_104: "f32[256]" = torch.ops.aten.sqrt.default(add_326);  add_326 = None
        reciprocal_104: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_387: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        sub_127: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_573, arg220_1);  view_573 = arg220_1 = None
        mul_388: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_127, mul_387);  sub_127 = mul_387 = None
        mul_389: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_388, arg222_1);  mul_388 = arg222_1 = None
        add_327: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_389, arg223_1);  mul_389 = arg223_1 = None
        view_574: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_327, [8, 49, 256]);  add_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_328: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_322, view_574);  add_322 = view_574 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_191: "f32[256, 1280]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        view_575: "f32[392, 256]" = torch.ops.aten.view.default(add_328, [392, 256])
        mm_95: "f32[392, 1280]" = torch.ops.aten.mm.default(view_575, permute_191);  view_575 = permute_191 = None
        view_576: "f32[8, 49, 1280]" = torch.ops.aten.view.default(mm_95, [8, 49, 1280]);  mm_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_577: "f32[392, 1280]" = torch.ops.aten.view.default(view_576, [392, 1280]);  view_576 = None
        add_329: "f32[1280]" = torch.ops.aten.add.Tensor(arg226_1, 1e-05);  arg226_1 = None
        sqrt_105: "f32[1280]" = torch.ops.aten.sqrt.default(add_329);  add_329 = None
        reciprocal_105: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_390: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        sub_128: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_577, arg225_1);  view_577 = arg225_1 = None
        mul_391: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(sub_128, mul_390);  sub_128 = mul_390 = None
        mul_392: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(mul_391, arg227_1);  mul_391 = arg227_1 = None
        add_330: "f32[392, 1280]" = torch.ops.aten.add.Tensor(mul_392, arg228_1);  mul_392 = arg228_1 = None
        view_578: "f32[8, 49, 1280]" = torch.ops.aten.view.default(add_330, [8, 49, 1280]);  add_330 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:333 in forward, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
        view_579: "f32[8, 49, 16, 80]" = torch.ops.aten.view.default(view_578, [8, 49, 16, -1]);  view_578 = None
        split_with_sizes_23 = torch.ops.aten.split_with_sizes.default(view_579, [16, 64], 3);  view_579 = None
        getitem_66: "f32[8, 49, 16, 16]" = split_with_sizes_23[0]
        getitem_67: "f32[8, 49, 16, 64]" = split_with_sizes_23[1];  split_with_sizes_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:334 in forward, code: k = k.permute(0, 2, 3, 1)  # BHCN
        permute_192: "f32[8, 16, 16, 49]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 3, 1]);  getitem_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:335 in forward, code: v = v.permute(0, 2, 1, 3)  # BHNC
        permute_193: "f32[8, 16, 49, 64]" = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3]);  getitem_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:158 in forward, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
        view_580: "f32[8, 7, 7, 256]" = torch.ops.aten.view.default(add_328, [8, 7, 7, 256]);  add_328 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:162 in forward, code: x = x[:, ::self.stride, ::self.stride]
        slice_25: "f32[8, 4, 7, 256]" = torch.ops.aten.slice.Tensor(view_580, 1, 0, 9223372036854775807, 2);  view_580 = None
        slice_26: "f32[8, 4, 4, 256]" = torch.ops.aten.slice.Tensor(slice_25, 2, 0, 9223372036854775807, 2);  slice_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:163 in forward, code: return x.reshape(B, -1, C)
        clone_138: "f32[8, 4, 4, 256]" = torch.ops.aten.clone.default(slice_26, memory_format = torch.contiguous_format);  slice_26 = None
        view_581: "f32[8, 16, 256]" = torch.ops.aten.view.default(clone_138, [8, 16, 256]);  clone_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_194: "f32[256, 256]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        view_582: "f32[128, 256]" = torch.ops.aten.view.default(view_581, [128, 256]);  view_581 = None
        mm_96: "f32[128, 256]" = torch.ops.aten.mm.default(view_582, permute_194);  view_582 = permute_194 = None
        view_583: "f32[8, 16, 256]" = torch.ops.aten.view.default(mm_96, [8, 16, 256]);  mm_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_584: "f32[128, 256]" = torch.ops.aten.view.default(view_583, [128, 256]);  view_583 = None
        add_331: "f32[256]" = torch.ops.aten.add.Tensor(arg231_1, 1e-05);  arg231_1 = None
        sqrt_106: "f32[256]" = torch.ops.aten.sqrt.default(add_331);  add_331 = None
        reciprocal_106: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_393: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        sub_129: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_584, arg230_1);  view_584 = arg230_1 = None
        mul_394: "f32[128, 256]" = torch.ops.aten.mul.Tensor(sub_129, mul_393);  sub_129 = mul_393 = None
        mul_395: "f32[128, 256]" = torch.ops.aten.mul.Tensor(mul_394, arg232_1);  mul_394 = arg232_1 = None
        add_332: "f32[128, 256]" = torch.ops.aten.add.Tensor(mul_395, arg233_1);  mul_395 = arg233_1 = None
        view_585: "f32[8, 16, 256]" = torch.ops.aten.view.default(add_332, [8, 16, 256]);  add_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:336 in forward, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        view_586: "f32[8, 16, 16, 16]" = torch.ops.aten.view.default(view_585, [8, -1, 16, 16]);  view_585 = None
        permute_195: "f32[8, 16, 16, 16]" = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:338 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_92: "f32[8, 16, 16, 16]" = torch.ops.aten.expand.default(permute_195, [8, 16, 16, 16]);  permute_195 = None
        clone_139: "f32[8, 16, 16, 16]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_587: "f32[128, 16, 16]" = torch.ops.aten.view.default(clone_139, [128, 16, 16]);  clone_139 = None
        expand_93: "f32[8, 16, 16, 49]" = torch.ops.aten.expand.default(permute_192, [8, 16, 16, 49]);  permute_192 = None
        clone_140: "f32[8, 16, 16, 49]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_588: "f32[128, 16, 49]" = torch.ops.aten.view.default(clone_140, [128, 16, 49]);  clone_140 = None
        bmm_46: "f32[128, 16, 49]" = torch.ops.aten.bmm.default(view_587, view_588);  view_587 = view_588 = None
        view_589: "f32[8, 16, 16, 49]" = torch.ops.aten.view.default(bmm_46, [8, 16, 16, 49]);  bmm_46 = None
        mul_396: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(view_589, 0.25);  view_589 = None
        add_333: "f32[8, 16, 16, 49]" = torch.ops.aten.add.Tensor(mul_396, index_9);  mul_396 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:339 in forward, code: attn = attn.softmax(dim=-1)
        amax_23: "f32[8, 16, 16, 1]" = torch.ops.aten.amax.default(add_333, [-1], True)
        sub_130: "f32[8, 16, 16, 49]" = torch.ops.aten.sub.Tensor(add_333, amax_23);  add_333 = amax_23 = None
        exp_23: "f32[8, 16, 16, 49]" = torch.ops.aten.exp.default(sub_130);  sub_130 = None
        sum_24: "f32[8, 16, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_76: "f32[8, 16, 16, 49]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:341 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
        expand_94: "f32[8, 16, 16, 49]" = torch.ops.aten.expand.default(div_76, [8, 16, 16, 49]);  div_76 = None
        view_590: "f32[128, 16, 49]" = torch.ops.aten.view.default(expand_94, [128, 16, 49]);  expand_94 = None
        expand_95: "f32[8, 16, 49, 64]" = torch.ops.aten.expand.default(permute_193, [8, 16, 49, 64]);  permute_193 = None
        clone_141: "f32[8, 16, 49, 64]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_591: "f32[128, 49, 64]" = torch.ops.aten.view.default(clone_141, [128, 49, 64]);  clone_141 = None
        bmm_47: "f32[128, 16, 64]" = torch.ops.aten.bmm.default(view_590, view_591);  view_590 = view_591 = None
        view_592: "f32[8, 16, 16, 64]" = torch.ops.aten.view.default(bmm_47, [8, 16, 16, 64]);  bmm_47 = None
        permute_196: "f32[8, 16, 16, 64]" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
        clone_142: "f32[8, 16, 16, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
        view_593: "f32[8, 16, 1024]" = torch.ops.aten.view.default(clone_142, [8, 16, 1024]);  clone_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:342 in forward, code: x = self.proj(x)
        add_334: "f32[8, 16, 1024]" = torch.ops.aten.add.Tensor(view_593, 3)
        clamp_min_52: "f32[8, 16, 1024]" = torch.ops.aten.clamp_min.default(add_334, 0);  add_334 = None
        clamp_max_52: "f32[8, 16, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_52, 6);  clamp_min_52 = None
        mul_397: "f32[8, 16, 1024]" = torch.ops.aten.mul.Tensor(view_593, clamp_max_52);  view_593 = clamp_max_52 = None
        div_77: "f32[8, 16, 1024]" = torch.ops.aten.div.Tensor(mul_397, 6);  mul_397 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_197: "f32[1024, 384]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        view_594: "f32[128, 1024]" = torch.ops.aten.view.default(div_77, [128, 1024]);  div_77 = None
        mm_97: "f32[128, 384]" = torch.ops.aten.mm.default(view_594, permute_197);  view_594 = permute_197 = None
        view_595: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_97, [8, 16, 384]);  mm_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_596: "f32[128, 384]" = torch.ops.aten.view.default(view_595, [128, 384]);  view_595 = None
        add_335: "f32[384]" = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_107: "f32[384]" = torch.ops.aten.sqrt.default(add_335);  add_335 = None
        reciprocal_107: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_398: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        sub_131: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_596, arg237_1);  view_596 = arg237_1 = None
        mul_399: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_131, mul_398);  sub_131 = mul_398 = None
        mul_400: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_399, arg239_1);  mul_399 = arg239_1 = None
        add_336: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_400, arg240_1);  mul_400 = arg240_1 = None
        view_597: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_336, [8, 16, 384]);  add_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_198: "f32[384, 768]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        view_598: "f32[128, 384]" = torch.ops.aten.view.default(view_597, [128, 384])
        mm_98: "f32[128, 768]" = torch.ops.aten.mm.default(view_598, permute_198);  view_598 = permute_198 = None
        view_599: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_98, [8, 16, 768]);  mm_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_600: "f32[128, 768]" = torch.ops.aten.view.default(view_599, [128, 768]);  view_599 = None
        add_337: "f32[768]" = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_108: "f32[768]" = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_108: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_401: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        sub_132: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_600, arg242_1);  view_600 = arg242_1 = None
        mul_402: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_132, mul_401);  sub_132 = mul_401 = None
        mul_403: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_402, arg244_1);  mul_402 = arg244_1 = None
        add_338: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_403, arg245_1);  mul_403 = arg245_1 = None
        view_601: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_338, [8, 16, 768]);  add_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_339: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_601, 3)
        clamp_min_53: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_339, 0);  add_339 = None
        clamp_max_53: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_53, 6);  clamp_min_53 = None
        mul_404: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_601, clamp_max_53);  view_601 = clamp_max_53 = None
        div_78: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_404, 6);  mul_404 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_199: "f32[768, 384]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        view_602: "f32[128, 768]" = torch.ops.aten.view.default(div_78, [128, 768]);  div_78 = None
        mm_99: "f32[128, 384]" = torch.ops.aten.mm.default(view_602, permute_199);  view_602 = permute_199 = None
        view_603: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_99, [8, 16, 384]);  mm_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_604: "f32[128, 384]" = torch.ops.aten.view.default(view_603, [128, 384]);  view_603 = None
        add_340: "f32[384]" = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_109: "f32[384]" = torch.ops.aten.sqrt.default(add_340);  add_340 = None
        reciprocal_109: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_405: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        sub_133: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_604, arg247_1);  view_604 = arg247_1 = None
        mul_406: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_133, mul_405);  sub_133 = mul_405 = None
        mul_407: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_406, arg249_1);  mul_406 = arg249_1 = None
        add_341: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_407, arg250_1);  mul_407 = arg250_1 = None
        view_605: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_341, [8, 16, 384]);  add_341 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:417 in forward, code: x = x + self.drop_path(self.mlp(x))
        add_342: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_597, view_605);  view_597 = view_605 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_200: "f32[384, 768]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        view_606: "f32[128, 384]" = torch.ops.aten.view.default(add_342, [128, 384])
        mm_100: "f32[128, 768]" = torch.ops.aten.mm.default(view_606, permute_200);  view_606 = permute_200 = None
        view_607: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_100, [8, 16, 768]);  mm_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_608: "f32[128, 768]" = torch.ops.aten.view.default(view_607, [128, 768]);  view_607 = None
        add_343: "f32[768]" = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_110: "f32[768]" = torch.ops.aten.sqrt.default(add_343);  add_343 = None
        reciprocal_110: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_408: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        sub_134: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_608, arg252_1);  view_608 = arg252_1 = None
        mul_409: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_134, mul_408);  sub_134 = mul_408 = None
        mul_410: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_409, arg254_1);  mul_409 = arg254_1 = None
        add_344: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_410, arg255_1);  mul_410 = arg255_1 = None
        view_609: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_344, [8, 16, 768]);  add_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_610: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_609, [8, 16, 12, -1]);  view_609 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(view_610, [16, 16, 32], 3);  view_610 = None
        getitem_68: "f32[8, 16, 12, 16]" = split_with_sizes_24[0]
        getitem_69: "f32[8, 16, 12, 16]" = split_with_sizes_24[1]
        getitem_70: "f32[8, 16, 12, 32]" = split_with_sizes_24[2];  split_with_sizes_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_201: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_202: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 3, 1]);  getitem_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_203: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_70, [0, 2, 1, 3]);  getitem_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_96: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_201, [8, 12, 16, 16]);  permute_201 = None
        clone_144: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_611: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_144, [96, 16, 16]);  clone_144 = None
        expand_97: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_202, [8, 12, 16, 16]);  permute_202 = None
        clone_145: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_612: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_145, [96, 16, 16]);  clone_145 = None
        bmm_48: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_611, view_612);  view_611 = view_612 = None
        view_613: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_48, [8, 12, 16, 16]);  bmm_48 = None
        mul_411: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_613, 0.25);  view_613 = None
        add_345: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_411, index_10);  mul_411 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_24: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_345, [-1], True)
        sub_135: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_345, amax_24);  add_345 = amax_24 = None
        exp_24: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_135);  sub_135 = None
        sum_25: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_79: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_98: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_79, [8, 12, 16, 16]);  div_79 = None
        view_614: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_98, [96, 16, 16]);  expand_98 = None
        expand_99: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_203, [8, 12, 16, 32]);  permute_203 = None
        clone_146: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_615: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_146, [96, 16, 32]);  clone_146 = None
        bmm_49: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_614, view_615);  view_614 = view_615 = None
        view_616: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_49, [8, 12, 16, 32]);  bmm_49 = None
        permute_204: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
        clone_147: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_617: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_147, [8, 16, 384]);  clone_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_346: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_617, 3)
        clamp_min_54: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_346, 0);  add_346 = None
        clamp_max_54: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_54, 6);  clamp_min_54 = None
        mul_412: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_617, clamp_max_54);  view_617 = clamp_max_54 = None
        div_80: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_412, 6);  mul_412 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_205: "f32[384, 384]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        view_618: "f32[128, 384]" = torch.ops.aten.view.default(div_80, [128, 384]);  div_80 = None
        mm_101: "f32[128, 384]" = torch.ops.aten.mm.default(view_618, permute_205);  view_618 = permute_205 = None
        view_619: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_101, [8, 16, 384]);  mm_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_620: "f32[128, 384]" = torch.ops.aten.view.default(view_619, [128, 384]);  view_619 = None
        add_347: "f32[384]" = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
        sqrt_111: "f32[384]" = torch.ops.aten.sqrt.default(add_347);  add_347 = None
        reciprocal_111: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_413: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        sub_136: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_620, arg259_1);  view_620 = arg259_1 = None
        mul_414: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_136, mul_413);  sub_136 = mul_413 = None
        mul_415: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_414, arg261_1);  mul_414 = arg261_1 = None
        add_348: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_415, arg262_1);  mul_415 = arg262_1 = None
        view_621: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_348, [8, 16, 384]);  add_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_349: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_342, view_621);  add_342 = view_621 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_206: "f32[384, 768]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        view_622: "f32[128, 384]" = torch.ops.aten.view.default(add_349, [128, 384])
        mm_102: "f32[128, 768]" = torch.ops.aten.mm.default(view_622, permute_206);  view_622 = permute_206 = None
        view_623: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_102, [8, 16, 768]);  mm_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_624: "f32[128, 768]" = torch.ops.aten.view.default(view_623, [128, 768]);  view_623 = None
        add_350: "f32[768]" = torch.ops.aten.add.Tensor(arg265_1, 1e-05);  arg265_1 = None
        sqrt_112: "f32[768]" = torch.ops.aten.sqrt.default(add_350);  add_350 = None
        reciprocal_112: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_416: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        sub_137: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_624, arg264_1);  view_624 = arg264_1 = None
        mul_417: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_137, mul_416);  sub_137 = mul_416 = None
        mul_418: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_417, arg266_1);  mul_417 = arg266_1 = None
        add_351: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_418, arg267_1);  mul_418 = arg267_1 = None
        view_625: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_351, [8, 16, 768]);  add_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_352: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_625, 3)
        clamp_min_55: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_352, 0);  add_352 = None
        clamp_max_55: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_55, 6);  clamp_min_55 = None
        mul_419: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_625, clamp_max_55);  view_625 = clamp_max_55 = None
        div_81: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_419, 6);  mul_419 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_207: "f32[768, 384]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        view_626: "f32[128, 768]" = torch.ops.aten.view.default(div_81, [128, 768]);  div_81 = None
        mm_103: "f32[128, 384]" = torch.ops.aten.mm.default(view_626, permute_207);  view_626 = permute_207 = None
        view_627: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_103, [8, 16, 384]);  mm_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_628: "f32[128, 384]" = torch.ops.aten.view.default(view_627, [128, 384]);  view_627 = None
        add_353: "f32[384]" = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
        sqrt_113: "f32[384]" = torch.ops.aten.sqrt.default(add_353);  add_353 = None
        reciprocal_113: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_420: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        sub_138: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_628, arg269_1);  view_628 = arg269_1 = None
        mul_421: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_138, mul_420);  sub_138 = mul_420 = None
        mul_422: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_421, arg271_1);  mul_421 = arg271_1 = None
        add_354: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_422, arg272_1);  mul_422 = arg272_1 = None
        view_629: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_354, [8, 16, 384]);  add_354 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_355: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_349, view_629);  add_349 = view_629 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_208: "f32[384, 768]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        view_630: "f32[128, 384]" = torch.ops.aten.view.default(add_355, [128, 384])
        mm_104: "f32[128, 768]" = torch.ops.aten.mm.default(view_630, permute_208);  view_630 = permute_208 = None
        view_631: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_104, [8, 16, 768]);  mm_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_632: "f32[128, 768]" = torch.ops.aten.view.default(view_631, [128, 768]);  view_631 = None
        add_356: "f32[768]" = torch.ops.aten.add.Tensor(arg275_1, 1e-05);  arg275_1 = None
        sqrt_114: "f32[768]" = torch.ops.aten.sqrt.default(add_356);  add_356 = None
        reciprocal_114: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_423: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        sub_139: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_632, arg274_1);  view_632 = arg274_1 = None
        mul_424: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_139, mul_423);  sub_139 = mul_423 = None
        mul_425: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_424, arg276_1);  mul_424 = arg276_1 = None
        add_357: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_425, arg277_1);  mul_425 = arg277_1 = None
        view_633: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_357, [8, 16, 768]);  add_357 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_634: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_633, [8, 16, 12, -1]);  view_633 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_25 = torch.ops.aten.split_with_sizes.default(view_634, [16, 16, 32], 3);  view_634 = None
        getitem_71: "f32[8, 16, 12, 16]" = split_with_sizes_25[0]
        getitem_72: "f32[8, 16, 12, 16]" = split_with_sizes_25[1]
        getitem_73: "f32[8, 16, 12, 32]" = split_with_sizes_25[2];  split_with_sizes_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_209: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_210: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 3, 1]);  getitem_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_211: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_73, [0, 2, 1, 3]);  getitem_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_100: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_209, [8, 12, 16, 16]);  permute_209 = None
        clone_149: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_635: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_149, [96, 16, 16]);  clone_149 = None
        expand_101: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_210, [8, 12, 16, 16]);  permute_210 = None
        clone_150: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_636: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_150, [96, 16, 16]);  clone_150 = None
        bmm_50: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_635, view_636);  view_635 = view_636 = None
        view_637: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_50, [8, 12, 16, 16]);  bmm_50 = None
        mul_426: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_637, 0.25);  view_637 = None
        add_358: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_426, index_11);  mul_426 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_25: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_358, [-1], True)
        sub_140: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_358, amax_25);  add_358 = amax_25 = None
        exp_25: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_140);  sub_140 = None
        sum_26: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_82: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_102: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_82, [8, 12, 16, 16]);  div_82 = None
        view_638: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_102, [96, 16, 16]);  expand_102 = None
        expand_103: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_211, [8, 12, 16, 32]);  permute_211 = None
        clone_151: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_639: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_151, [96, 16, 32]);  clone_151 = None
        bmm_51: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_638, view_639);  view_638 = view_639 = None
        view_640: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_51, [8, 12, 16, 32]);  bmm_51 = None
        permute_212: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None
        clone_152: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_641: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_152, [8, 16, 384]);  clone_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_359: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_641, 3)
        clamp_min_56: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_359, 0);  add_359 = None
        clamp_max_56: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_56, 6);  clamp_min_56 = None
        mul_427: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_641, clamp_max_56);  view_641 = clamp_max_56 = None
        div_83: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_427, 6);  mul_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_213: "f32[384, 384]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        view_642: "f32[128, 384]" = torch.ops.aten.view.default(div_83, [128, 384]);  div_83 = None
        mm_105: "f32[128, 384]" = torch.ops.aten.mm.default(view_642, permute_213);  view_642 = permute_213 = None
        view_643: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_105, [8, 16, 384]);  mm_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_644: "f32[128, 384]" = torch.ops.aten.view.default(view_643, [128, 384]);  view_643 = None
        add_360: "f32[384]" = torch.ops.aten.add.Tensor(arg282_1, 1e-05);  arg282_1 = None
        sqrt_115: "f32[384]" = torch.ops.aten.sqrt.default(add_360);  add_360 = None
        reciprocal_115: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_428: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        sub_141: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_644, arg281_1);  view_644 = arg281_1 = None
        mul_429: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_141, mul_428);  sub_141 = mul_428 = None
        mul_430: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_429, arg283_1);  mul_429 = arg283_1 = None
        add_361: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_430, arg284_1);  mul_430 = arg284_1 = None
        view_645: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_361, [8, 16, 384]);  add_361 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_362: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_355, view_645);  add_355 = view_645 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_214: "f32[384, 768]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        view_646: "f32[128, 384]" = torch.ops.aten.view.default(add_362, [128, 384])
        mm_106: "f32[128, 768]" = torch.ops.aten.mm.default(view_646, permute_214);  view_646 = permute_214 = None
        view_647: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_106, [8, 16, 768]);  mm_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_648: "f32[128, 768]" = torch.ops.aten.view.default(view_647, [128, 768]);  view_647 = None
        add_363: "f32[768]" = torch.ops.aten.add.Tensor(arg287_1, 1e-05);  arg287_1 = None
        sqrt_116: "f32[768]" = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_116: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_431: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        sub_142: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_648, arg286_1);  view_648 = arg286_1 = None
        mul_432: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_142, mul_431);  sub_142 = mul_431 = None
        mul_433: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_432, arg288_1);  mul_432 = arg288_1 = None
        add_364: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_433, arg289_1);  mul_433 = arg289_1 = None
        view_649: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_364, [8, 16, 768]);  add_364 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_365: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_649, 3)
        clamp_min_57: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_365, 0);  add_365 = None
        clamp_max_57: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_57, 6);  clamp_min_57 = None
        mul_434: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_649, clamp_max_57);  view_649 = clamp_max_57 = None
        div_84: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_434, 6);  mul_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_215: "f32[768, 384]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        view_650: "f32[128, 768]" = torch.ops.aten.view.default(div_84, [128, 768]);  div_84 = None
        mm_107: "f32[128, 384]" = torch.ops.aten.mm.default(view_650, permute_215);  view_650 = permute_215 = None
        view_651: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_107, [8, 16, 384]);  mm_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_652: "f32[128, 384]" = torch.ops.aten.view.default(view_651, [128, 384]);  view_651 = None
        add_366: "f32[384]" = torch.ops.aten.add.Tensor(arg292_1, 1e-05);  arg292_1 = None
        sqrt_117: "f32[384]" = torch.ops.aten.sqrt.default(add_366);  add_366 = None
        reciprocal_117: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_435: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        sub_143: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_652, arg291_1);  view_652 = arg291_1 = None
        mul_436: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_143, mul_435);  sub_143 = mul_435 = None
        mul_437: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_436, arg293_1);  mul_436 = arg293_1 = None
        add_367: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_437, arg294_1);  mul_437 = arg294_1 = None
        view_653: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_367, [8, 16, 384]);  add_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_368: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_362, view_653);  add_362 = view_653 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_216: "f32[384, 768]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        view_654: "f32[128, 384]" = torch.ops.aten.view.default(add_368, [128, 384])
        mm_108: "f32[128, 768]" = torch.ops.aten.mm.default(view_654, permute_216);  view_654 = permute_216 = None
        view_655: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_108, [8, 16, 768]);  mm_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_656: "f32[128, 768]" = torch.ops.aten.view.default(view_655, [128, 768]);  view_655 = None
        add_369: "f32[768]" = torch.ops.aten.add.Tensor(arg297_1, 1e-05);  arg297_1 = None
        sqrt_118: "f32[768]" = torch.ops.aten.sqrt.default(add_369);  add_369 = None
        reciprocal_118: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_438: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        sub_144: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_656, arg296_1);  view_656 = arg296_1 = None
        mul_439: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_144, mul_438);  sub_144 = mul_438 = None
        mul_440: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_439, arg298_1);  mul_439 = arg298_1 = None
        add_370: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_440, arg299_1);  mul_440 = arg299_1 = None
        view_657: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_370, [8, 16, 768]);  add_370 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_658: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_657, [8, 16, 12, -1]);  view_657 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(view_658, [16, 16, 32], 3);  view_658 = None
        getitem_74: "f32[8, 16, 12, 16]" = split_with_sizes_26[0]
        getitem_75: "f32[8, 16, 12, 16]" = split_with_sizes_26[1]
        getitem_76: "f32[8, 16, 12, 32]" = split_with_sizes_26[2];  split_with_sizes_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_217: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_218: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_75, [0, 2, 3, 1]);  getitem_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_219: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_76, [0, 2, 1, 3]);  getitem_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_104: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_217, [8, 12, 16, 16]);  permute_217 = None
        clone_154: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_659: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_154, [96, 16, 16]);  clone_154 = None
        expand_105: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_218, [8, 12, 16, 16]);  permute_218 = None
        clone_155: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_660: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_155, [96, 16, 16]);  clone_155 = None
        bmm_52: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_659, view_660);  view_659 = view_660 = None
        view_661: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_52, [8, 12, 16, 16]);  bmm_52 = None
        mul_441: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_661, 0.25);  view_661 = None
        add_371: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_441, index_12);  mul_441 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_26: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_371, [-1], True)
        sub_145: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_371, amax_26);  add_371 = amax_26 = None
        exp_26: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_145);  sub_145 = None
        sum_27: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_85: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_106: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_85, [8, 12, 16, 16]);  div_85 = None
        view_662: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_106, [96, 16, 16]);  expand_106 = None
        expand_107: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_219, [8, 12, 16, 32]);  permute_219 = None
        clone_156: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_663: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_156, [96, 16, 32]);  clone_156 = None
        bmm_53: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_662, view_663);  view_662 = view_663 = None
        view_664: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_53, [8, 12, 16, 32]);  bmm_53 = None
        permute_220: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None
        clone_157: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_665: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_157, [8, 16, 384]);  clone_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_372: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_665, 3)
        clamp_min_58: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_372, 0);  add_372 = None
        clamp_max_58: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_58, 6);  clamp_min_58 = None
        mul_442: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_665, clamp_max_58);  view_665 = clamp_max_58 = None
        div_86: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_442, 6);  mul_442 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_221: "f32[384, 384]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        view_666: "f32[128, 384]" = torch.ops.aten.view.default(div_86, [128, 384]);  div_86 = None
        mm_109: "f32[128, 384]" = torch.ops.aten.mm.default(view_666, permute_221);  view_666 = permute_221 = None
        view_667: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_109, [8, 16, 384]);  mm_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_668: "f32[128, 384]" = torch.ops.aten.view.default(view_667, [128, 384]);  view_667 = None
        add_373: "f32[384]" = torch.ops.aten.add.Tensor(arg304_1, 1e-05);  arg304_1 = None
        sqrt_119: "f32[384]" = torch.ops.aten.sqrt.default(add_373);  add_373 = None
        reciprocal_119: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_443: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        sub_146: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_668, arg303_1);  view_668 = arg303_1 = None
        mul_444: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_146, mul_443);  sub_146 = mul_443 = None
        mul_445: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_444, arg305_1);  mul_444 = arg305_1 = None
        add_374: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_445, arg306_1);  mul_445 = arg306_1 = None
        view_669: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_374, [8, 16, 384]);  add_374 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_375: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_368, view_669);  add_368 = view_669 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_222: "f32[384, 768]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        view_670: "f32[128, 384]" = torch.ops.aten.view.default(add_375, [128, 384])
        mm_110: "f32[128, 768]" = torch.ops.aten.mm.default(view_670, permute_222);  view_670 = permute_222 = None
        view_671: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_110, [8, 16, 768]);  mm_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_672: "f32[128, 768]" = torch.ops.aten.view.default(view_671, [128, 768]);  view_671 = None
        add_376: "f32[768]" = torch.ops.aten.add.Tensor(arg309_1, 1e-05);  arg309_1 = None
        sqrt_120: "f32[768]" = torch.ops.aten.sqrt.default(add_376);  add_376 = None
        reciprocal_120: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_446: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        sub_147: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_672, arg308_1);  view_672 = arg308_1 = None
        mul_447: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_147, mul_446);  sub_147 = mul_446 = None
        mul_448: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_447, arg310_1);  mul_447 = arg310_1 = None
        add_377: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_448, arg311_1);  mul_448 = arg311_1 = None
        view_673: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_377, [8, 16, 768]);  add_377 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_378: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_673, 3)
        clamp_min_59: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_378, 0);  add_378 = None
        clamp_max_59: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_59, 6);  clamp_min_59 = None
        mul_449: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_673, clamp_max_59);  view_673 = clamp_max_59 = None
        div_87: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_449, 6);  mul_449 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_223: "f32[768, 384]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
        view_674: "f32[128, 768]" = torch.ops.aten.view.default(div_87, [128, 768]);  div_87 = None
        mm_111: "f32[128, 384]" = torch.ops.aten.mm.default(view_674, permute_223);  view_674 = permute_223 = None
        view_675: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_111, [8, 16, 384]);  mm_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_676: "f32[128, 384]" = torch.ops.aten.view.default(view_675, [128, 384]);  view_675 = None
        add_379: "f32[384]" = torch.ops.aten.add.Tensor(arg314_1, 1e-05);  arg314_1 = None
        sqrt_121: "f32[384]" = torch.ops.aten.sqrt.default(add_379);  add_379 = None
        reciprocal_121: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_450: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        sub_148: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_676, arg313_1);  view_676 = arg313_1 = None
        mul_451: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_148, mul_450);  sub_148 = mul_450 = None
        mul_452: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_451, arg315_1);  mul_451 = arg315_1 = None
        add_380: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_452, arg316_1);  mul_452 = arg316_1 = None
        view_677: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_380, [8, 16, 384]);  add_380 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_381: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_375, view_677);  add_375 = view_677 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_224: "f32[384, 768]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        view_678: "f32[128, 384]" = torch.ops.aten.view.default(add_381, [128, 384])
        mm_112: "f32[128, 768]" = torch.ops.aten.mm.default(view_678, permute_224);  view_678 = permute_224 = None
        view_679: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_112, [8, 16, 768]);  mm_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_680: "f32[128, 768]" = torch.ops.aten.view.default(view_679, [128, 768]);  view_679 = None
        add_382: "f32[768]" = torch.ops.aten.add.Tensor(arg319_1, 1e-05);  arg319_1 = None
        sqrt_122: "f32[768]" = torch.ops.aten.sqrt.default(add_382);  add_382 = None
        reciprocal_122: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_453: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        sub_149: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_680, arg318_1);  view_680 = arg318_1 = None
        mul_454: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_149, mul_453);  sub_149 = mul_453 = None
        mul_455: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_454, arg320_1);  mul_454 = arg320_1 = None
        add_383: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_455, arg321_1);  mul_455 = arg321_1 = None
        view_681: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_383, [8, 16, 768]);  add_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:231 in forward, code: q, k, v = self.qkv(x).view(
        view_682: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_681, [8, 16, 12, -1]);  view_681 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:232 in forward, code: B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(view_682, [16, 16, 32], 3);  view_682 = None
        getitem_77: "f32[8, 16, 12, 16]" = split_with_sizes_27[0]
        getitem_78: "f32[8, 16, 12, 16]" = split_with_sizes_27[1]
        getitem_79: "f32[8, 16, 12, 32]" = split_with_sizes_27[2];  split_with_sizes_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:233 in forward, code: q = q.permute(0, 2, 1, 3)
        permute_225: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_77, [0, 2, 1, 3]);  getitem_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:234 in forward, code: k = k.permute(0, 2, 3, 1)
        permute_226: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_78, [0, 2, 3, 1]);  getitem_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:235 in forward, code: v = v.permute(0, 2, 1, 3)
        permute_227: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3]);  getitem_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:237 in forward, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
        expand_108: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_225, [8, 12, 16, 16]);  permute_225 = None
        clone_159: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_683: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_159, [96, 16, 16]);  clone_159 = None
        expand_109: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_226, [8, 12, 16, 16]);  permute_226 = None
        clone_160: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_684: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_160, [96, 16, 16]);  clone_160 = None
        bmm_54: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_683, view_684);  view_683 = view_684 = None
        view_685: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_54, [8, 12, 16, 16]);  bmm_54 = None
        mul_456: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_685, 0.25);  view_685 = None
        add_384: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_456, index_13);  mul_456 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:238 in forward, code: attn = attn.softmax(dim=-1)
        amax_27: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_384, [-1], True)
        sub_150: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_384, amax_27);  add_384 = amax_27 = None
        exp_27: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_150);  sub_150 = None
        sum_28: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_88: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:240 in forward, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        expand_110: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_88, [8, 12, 16, 16]);  div_88 = None
        view_686: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_110, [96, 16, 16]);  expand_110 = None
        expand_111: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_227, [8, 12, 16, 32]);  permute_227 = None
        clone_161: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_687: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_161, [96, 16, 32]);  clone_161 = None
        bmm_55: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_686, view_687);  view_686 = view_687 = None
        view_688: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_55, [8, 12, 16, 32]);  bmm_55 = None
        permute_228: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None
        clone_162: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
        view_689: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_162, [8, 16, 384]);  clone_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:241 in forward, code: x = self.proj(x)
        add_385: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_689, 3)
        clamp_min_60: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_385, 0);  add_385 = None
        clamp_max_60: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_60, 6);  clamp_min_60 = None
        mul_457: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_689, clamp_max_60);  view_689 = clamp_max_60 = None
        div_89: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_457, 6);  mul_457 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_229: "f32[384, 384]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        view_690: "f32[128, 384]" = torch.ops.aten.view.default(div_89, [128, 384]);  div_89 = None
        mm_113: "f32[128, 384]" = torch.ops.aten.mm.default(view_690, permute_229);  view_690 = permute_229 = None
        view_691: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_113, [8, 16, 384]);  mm_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_692: "f32[128, 384]" = torch.ops.aten.view.default(view_691, [128, 384]);  view_691 = None
        add_386: "f32[384]" = torch.ops.aten.add.Tensor(arg326_1, 1e-05);  arg326_1 = None
        sqrt_123: "f32[384]" = torch.ops.aten.sqrt.default(add_386);  add_386 = None
        reciprocal_123: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_458: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        sub_151: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_692, arg325_1);  view_692 = arg325_1 = None
        mul_459: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_151, mul_458);  sub_151 = mul_458 = None
        mul_460: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_459, arg327_1);  mul_459 = arg327_1 = None
        add_387: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_460, arg328_1);  mul_460 = arg328_1 = None
        view_693: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_387, [8, 16, 384]);  add_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:458 in forward, code: x = x + self.drop_path1(self.attn(x))
        add_388: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_381, view_693);  add_381 = view_693 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_230: "f32[384, 768]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        view_694: "f32[128, 384]" = torch.ops.aten.view.default(add_388, [128, 384])
        mm_114: "f32[128, 768]" = torch.ops.aten.mm.default(view_694, permute_230);  view_694 = permute_230 = None
        view_695: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_114, [8, 16, 768]);  mm_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_696: "f32[128, 768]" = torch.ops.aten.view.default(view_695, [128, 768]);  view_695 = None
        add_389: "f32[768]" = torch.ops.aten.add.Tensor(arg331_1, 1e-05);  arg331_1 = None
        sqrt_124: "f32[768]" = torch.ops.aten.sqrt.default(add_389);  add_389 = None
        reciprocal_124: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_461: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        sub_152: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_696, arg330_1);  view_696 = arg330_1 = None
        mul_462: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_152, mul_461);  sub_152 = mul_461 = None
        mul_463: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_462, arg332_1);  mul_462 = arg332_1 = None
        add_390: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_463, arg333_1);  mul_463 = arg333_1 = None
        view_697: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_390, [8, 16, 768]);  add_390 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:370 in forward, code: x = self.act(x)
        add_391: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_697, 3)
        clamp_min_61: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_391, 0);  add_391 = None
        clamp_max_61: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_61, 6);  clamp_min_61 = None
        mul_464: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_697, clamp_max_61);  view_697 = clamp_max_61 = None
        div_90: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_464, 6);  mul_464 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:89 in forward, code: x = self.linear(x)
        permute_231: "f32[768, 384]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
        view_698: "f32[128, 768]" = torch.ops.aten.view.default(div_90, [128, 768]);  div_90 = None
        mm_115: "f32[128, 384]" = torch.ops.aten.mm.default(view_698, permute_231);  view_698 = permute_231 = None
        view_699: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_115, [8, 16, 384]);  mm_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:90 in forward, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
        view_700: "f32[128, 384]" = torch.ops.aten.view.default(view_699, [128, 384]);  view_699 = None
        add_392: "f32[384]" = torch.ops.aten.add.Tensor(arg336_1, 1e-05);  arg336_1 = None
        sqrt_125: "f32[384]" = torch.ops.aten.sqrt.default(add_392);  add_392 = None
        reciprocal_125: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_465: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        sub_153: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_700, arg335_1);  view_700 = arg335_1 = None
        mul_466: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_153, mul_465);  sub_153 = mul_465 = None
        mul_467: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_466, arg337_1);  mul_466 = arg337_1 = None
        add_393: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_467, arg338_1);  mul_467 = arg338_1 = None
        view_701: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_393, [8, 16, 384]);  add_393 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:459 in forward, code: x = x + self.drop_path2(self.mlp(x))
        add_394: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_388, view_701);  add_388 = view_701 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:747 in forward_head, code: x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
        mean_1: "f32[8, 384]" = torch.ops.aten.mean.dim(add_394, [1]);  add_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:120 in forward, code: return self.linear(self.drop(self.bn(x)))
        add_395: "f32[384]" = torch.ops.aten.add.Tensor(arg340_1, 1e-05);  arg340_1 = None
        sqrt_126: "f32[384]" = torch.ops.aten.sqrt.default(add_395);  add_395 = None
        reciprocal_126: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_468: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        sub_154: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean_1, arg339_1);  arg339_1 = None
        mul_469: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_154, mul_468);  sub_154 = mul_468 = None
        mul_470: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mul_469, arg341_1);  mul_469 = arg341_1 = None
        add_396: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_470, arg342_1);  mul_470 = arg342_1 = None
        permute_232: "f32[384, 1000]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_2: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg344_1, add_396, permute_232);  arg344_1 = add_396 = permute_232 = None
        add_397: "f32[384]" = torch.ops.aten.add.Tensor(arg346_1, 1e-05);  arg346_1 = None
        sqrt_127: "f32[384]" = torch.ops.aten.sqrt.default(add_397);  add_397 = None
        reciprocal_127: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_471: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        sub_155: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean_1, arg345_1);  mean_1 = arg345_1 = None
        mul_472: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_155, mul_471);  sub_155 = mul_471 = None
        mul_473: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mul_472, arg347_1);  mul_472 = arg347_1 = None
        add_398: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_473, arg348_1);  mul_473 = arg348_1 = None
        permute_233: "f32[384, 1000]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_3: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg350_1, add_398, permute_233);  arg350_1 = add_398 = permute_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/levit.py:756 in forward_head, code: return (x + x_dist) / 2
        add_399: "f32[8, 1000]" = torch.ops.aten.add.Tensor(addmm_2, addmm_3);  addmm_2 = addmm_3 = None
        div_91: "f32[8, 1000]" = torch.ops.aten.div.Tensor(add_399, 2);  add_399 = None
        return (div_91, index, index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, index_10, index_11, index_12, index_13)
        