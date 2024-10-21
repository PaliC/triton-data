class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[8, 3, 256, 256]", arg2_1: "f32[16]", arg3_1: "f32[16]", arg4_1: "f32[16]", arg5_1: "f32[16]", arg6_1: "f32[64, 16, 1, 1]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64, 1, 3, 3]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[32, 64, 1, 1]", arg17_1: "f32[32]", arg18_1: "f32[32]", arg19_1: "f32[32]", arg20_1: "f32[32]", arg21_1: "f32[128, 32, 1, 1]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[128, 1, 3, 3]", arg27_1: "f32[128]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[128]", arg31_1: "f32[64, 128, 1, 1]", arg32_1: "f32[64]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[256, 64, 1, 1]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[256]", arg40_1: "f32[256]", arg41_1: "f32[256, 1, 3, 3]", arg42_1: "f32[256]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[256]", arg46_1: "f32[64, 256, 1, 1]", arg47_1: "f32[64]", arg48_1: "f32[64]", arg49_1: "f32[64]", arg50_1: "f32[64]", arg51_1: "f32[256, 64, 1, 1]", arg52_1: "f32[256]", arg53_1: "f32[256]", arg54_1: "f32[256]", arg55_1: "f32[256]", arg56_1: "f32[256, 1, 3, 3]", arg57_1: "f32[256]", arg58_1: "f32[256]", arg59_1: "f32[256]", arg60_1: "f32[256]", arg61_1: "f32[64, 256, 1, 1]", arg62_1: "f32[64]", arg63_1: "f32[64]", arg64_1: "f32[64]", arg65_1: "f32[64]", arg66_1: "f32[256, 64, 1, 1]", arg67_1: "f32[256]", arg68_1: "f32[256]", arg69_1: "f32[256]", arg70_1: "f32[256]", arg71_1: "f32[256, 1, 3, 3]", arg72_1: "f32[256]", arg73_1: "f32[256]", arg74_1: "f32[256]", arg75_1: "f32[256]", arg76_1: "f32[96, 256, 1, 1]", arg77_1: "f32[96]", arg78_1: "f32[96]", arg79_1: "f32[96]", arg80_1: "f32[96]", arg81_1: "f32[96, 96, 3, 3]", arg82_1: "f32[96]", arg83_1: "f32[96]", arg84_1: "f32[96]", arg85_1: "f32[96]", arg86_1: "f32[144, 96, 1, 1]", arg87_1: "f32[144]", arg88_1: "f32[144]", arg89_1: "f32[432, 144]", arg90_1: "f32[432]", arg91_1: "f32[144, 144]", arg92_1: "f32[144]", arg93_1: "f32[144]", arg94_1: "f32[144]", arg95_1: "f32[288, 144]", arg96_1: "f32[288]", arg97_1: "f32[144, 288]", arg98_1: "f32[144]", arg99_1: "f32[144]", arg100_1: "f32[144]", arg101_1: "f32[432, 144]", arg102_1: "f32[432]", arg103_1: "f32[144, 144]", arg104_1: "f32[144]", arg105_1: "f32[144]", arg106_1: "f32[144]", arg107_1: "f32[288, 144]", arg108_1: "f32[288]", arg109_1: "f32[144, 288]", arg110_1: "f32[144]", arg111_1: "f32[144]", arg112_1: "f32[144]", arg113_1: "f32[96, 144, 1, 1]", arg114_1: "f32[96]", arg115_1: "f32[96]", arg116_1: "f32[96]", arg117_1: "f32[96]", arg118_1: "f32[96, 192, 3, 3]", arg119_1: "f32[96]", arg120_1: "f32[96]", arg121_1: "f32[96]", arg122_1: "f32[96]", arg123_1: "f32[384, 96, 1, 1]", arg124_1: "f32[384]", arg125_1: "f32[384]", arg126_1: "f32[384]", arg127_1: "f32[384]", arg128_1: "f32[384, 1, 3, 3]", arg129_1: "f32[384]", arg130_1: "f32[384]", arg131_1: "f32[384]", arg132_1: "f32[384]", arg133_1: "f32[128, 384, 1, 1]", arg134_1: "f32[128]", arg135_1: "f32[128]", arg136_1: "f32[128]", arg137_1: "f32[128]", arg138_1: "f32[128, 128, 3, 3]", arg139_1: "f32[128]", arg140_1: "f32[128]", arg141_1: "f32[128]", arg142_1: "f32[128]", arg143_1: "f32[192, 128, 1, 1]", arg144_1: "f32[192]", arg145_1: "f32[192]", arg146_1: "f32[576, 192]", arg147_1: "f32[576]", arg148_1: "f32[192, 192]", arg149_1: "f32[192]", arg150_1: "f32[192]", arg151_1: "f32[192]", arg152_1: "f32[384, 192]", arg153_1: "f32[384]", arg154_1: "f32[192, 384]", arg155_1: "f32[192]", arg156_1: "f32[192]", arg157_1: "f32[192]", arg158_1: "f32[576, 192]", arg159_1: "f32[576]", arg160_1: "f32[192, 192]", arg161_1: "f32[192]", arg162_1: "f32[192]", arg163_1: "f32[192]", arg164_1: "f32[384, 192]", arg165_1: "f32[384]", arg166_1: "f32[192, 384]", arg167_1: "f32[192]", arg168_1: "f32[192]", arg169_1: "f32[192]", arg170_1: "f32[576, 192]", arg171_1: "f32[576]", arg172_1: "f32[192, 192]", arg173_1: "f32[192]", arg174_1: "f32[192]", arg175_1: "f32[192]", arg176_1: "f32[384, 192]", arg177_1: "f32[384]", arg178_1: "f32[192, 384]", arg179_1: "f32[192]", arg180_1: "f32[192]", arg181_1: "f32[192]", arg182_1: "f32[576, 192]", arg183_1: "f32[576]", arg184_1: "f32[192, 192]", arg185_1: "f32[192]", arg186_1: "f32[192]", arg187_1: "f32[192]", arg188_1: "f32[384, 192]", arg189_1: "f32[384]", arg190_1: "f32[192, 384]", arg191_1: "f32[192]", arg192_1: "f32[192]", arg193_1: "f32[192]", arg194_1: "f32[128, 192, 1, 1]", arg195_1: "f32[128]", arg196_1: "f32[128]", arg197_1: "f32[128]", arg198_1: "f32[128]", arg199_1: "f32[128, 256, 3, 3]", arg200_1: "f32[128]", arg201_1: "f32[128]", arg202_1: "f32[128]", arg203_1: "f32[128]", arg204_1: "f32[512, 128, 1, 1]", arg205_1: "f32[512]", arg206_1: "f32[512]", arg207_1: "f32[512]", arg208_1: "f32[512]", arg209_1: "f32[512, 1, 3, 3]", arg210_1: "f32[512]", arg211_1: "f32[512]", arg212_1: "f32[512]", arg213_1: "f32[512]", arg214_1: "f32[160, 512, 1, 1]", arg215_1: "f32[160]", arg216_1: "f32[160]", arg217_1: "f32[160]", arg218_1: "f32[160]", arg219_1: "f32[160, 160, 3, 3]", arg220_1: "f32[160]", arg221_1: "f32[160]", arg222_1: "f32[160]", arg223_1: "f32[160]", arg224_1: "f32[240, 160, 1, 1]", arg225_1: "f32[240]", arg226_1: "f32[240]", arg227_1: "f32[720, 240]", arg228_1: "f32[720]", arg229_1: "f32[240, 240]", arg230_1: "f32[240]", arg231_1: "f32[240]", arg232_1: "f32[240]", arg233_1: "f32[480, 240]", arg234_1: "f32[480]", arg235_1: "f32[240, 480]", arg236_1: "f32[240]", arg237_1: "f32[240]", arg238_1: "f32[240]", arg239_1: "f32[720, 240]", arg240_1: "f32[720]", arg241_1: "f32[240, 240]", arg242_1: "f32[240]", arg243_1: "f32[240]", arg244_1: "f32[240]", arg245_1: "f32[480, 240]", arg246_1: "f32[480]", arg247_1: "f32[240, 480]", arg248_1: "f32[240]", arg249_1: "f32[240]", arg250_1: "f32[240]", arg251_1: "f32[720, 240]", arg252_1: "f32[720]", arg253_1: "f32[240, 240]", arg254_1: "f32[240]", arg255_1: "f32[240]", arg256_1: "f32[240]", arg257_1: "f32[480, 240]", arg258_1: "f32[480]", arg259_1: "f32[240, 480]", arg260_1: "f32[240]", arg261_1: "f32[240]", arg262_1: "f32[240]", arg263_1: "f32[160, 240, 1, 1]", arg264_1: "f32[160]", arg265_1: "f32[160]", arg266_1: "f32[160]", arg267_1: "f32[160]", arg268_1: "f32[160, 320, 3, 3]", arg269_1: "f32[160]", arg270_1: "f32[160]", arg271_1: "f32[160]", arg272_1: "f32[160]", arg273_1: "f32[640, 160, 1, 1]", arg274_1: "f32[640]", arg275_1: "f32[640]", arg276_1: "f32[640]", arg277_1: "f32[640]", arg278_1: "f32[1000, 640]", arg279_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_35: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_126: "f32[16]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_32: "f32[16]" = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_32: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_172: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_256: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_257: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_172, -1);  mul_172 = None
        unsqueeze_259: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_53: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_257);  convolution_35 = unsqueeze_257 = None
        mul_173: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_259);  sub_53 = unsqueeze_259 = None
        unsqueeze_260: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_261: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_174: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_261);  mul_173 = unsqueeze_261 = None
        unsqueeze_262: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_263: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_127: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_263);  mul_174 = unsqueeze_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_34: "f32[8, 16, 128, 128]" = torch.ops.aten.sigmoid.default(add_127)
        mul_175: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_127, sigmoid_34);  add_127 = sigmoid_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_36: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_175, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_175 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_128: "f32[64]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_33: "f32[64]" = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_33: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_176: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_264: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_265: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_176, -1);  mul_176 = None
        unsqueeze_267: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_54: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_265);  convolution_36 = unsqueeze_265 = None
        mul_177: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_267);  sub_54 = unsqueeze_267 = None
        unsqueeze_268: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_269: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_178: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_177, unsqueeze_269);  mul_177 = unsqueeze_269 = None
        unsqueeze_270: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_271: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_129: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_178, unsqueeze_271);  mul_178 = unsqueeze_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_35: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_129)
        mul_179: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_129, sigmoid_35);  add_129 = sigmoid_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_37: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_179, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  mul_179 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_130: "f32[64]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_34: "f32[64]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_34: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_180: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_272: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_273: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
        unsqueeze_275: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_55: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_273);  convolution_37 = unsqueeze_273 = None
        mul_181: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_275);  sub_55 = unsqueeze_275 = None
        unsqueeze_276: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_277: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_182: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_277);  mul_181 = unsqueeze_277 = None
        unsqueeze_278: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_279: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_131: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_279);  mul_182 = unsqueeze_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_36: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_131)
        mul_183: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_131, sigmoid_36);  add_131 = sigmoid_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_38: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_183, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_183 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_132: "f32[32]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_35: "f32[32]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_35: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_184: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_280: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_281: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_184, -1);  mul_184 = None
        unsqueeze_283: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_56: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_281);  convolution_38 = unsqueeze_281 = None
        mul_185: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_283);  sub_56 = unsqueeze_283 = None
        unsqueeze_284: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_285: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_186: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_185, unsqueeze_285);  mul_185 = unsqueeze_285 = None
        unsqueeze_286: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_287: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_133: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_186, unsqueeze_287);  mul_186 = unsqueeze_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_39: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(add_133, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_133 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_134: "f32[128]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_36: "f32[128]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_36: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_187: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_288: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_289: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_187, -1);  mul_187 = None
        unsqueeze_291: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_57: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_289);  convolution_39 = unsqueeze_289 = None
        mul_188: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_291);  sub_57 = unsqueeze_291 = None
        unsqueeze_292: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_293: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_189: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_293);  mul_188 = unsqueeze_293 = None
        unsqueeze_294: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_295: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_135: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_295);  mul_189 = unsqueeze_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_37: "f32[8, 128, 128, 128]" = torch.ops.aten.sigmoid.default(add_135)
        mul_190: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_135, sigmoid_37);  add_135 = sigmoid_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_40: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_190, arg26_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128);  mul_190 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_136: "f32[128]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_37: "f32[128]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_37: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_191: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_296: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_297: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_191, -1);  mul_191 = None
        unsqueeze_299: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_58: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_297);  convolution_40 = unsqueeze_297 = None
        mul_192: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_299);  sub_58 = unsqueeze_299 = None
        unsqueeze_300: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_301: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_193: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_301);  mul_192 = unsqueeze_301 = None
        unsqueeze_302: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_303: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_137: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_303);  mul_193 = unsqueeze_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_38: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_137)
        mul_194: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_38);  add_137 = sigmoid_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_41: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_194, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_194 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_138: "f32[64]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_38: "f32[64]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_38: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_195: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_305: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_307: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_59: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_305);  convolution_41 = unsqueeze_305 = None
        mul_196: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_307);  sub_59 = unsqueeze_307 = None
        unsqueeze_308: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_309: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_197: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_309);  mul_196 = unsqueeze_309 = None
        unsqueeze_310: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_311: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_139: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_311);  mul_197 = unsqueeze_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_42: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_139, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_140: "f32[256]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_39: "f32[256]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_39: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_313: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_315: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_60: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_313);  convolution_42 = unsqueeze_313 = None
        mul_199: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_315);  sub_60 = unsqueeze_315 = None
        unsqueeze_316: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_317: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_200: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_317);  mul_199 = unsqueeze_317 = None
        unsqueeze_318: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_319: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_141: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_319);  mul_200 = unsqueeze_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_39: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_141)
        mul_201: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_141, sigmoid_39);  add_141 = sigmoid_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_43: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_201, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  mul_201 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_142: "f32[256]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_40: "f32[256]" = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_40: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_202: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_202, -1);  mul_202 = None
        unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_61: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_321);  convolution_43 = unsqueeze_321 = None
        mul_203: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_323);  sub_61 = unsqueeze_323 = None
        unsqueeze_324: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_325: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_204: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_325);  mul_203 = unsqueeze_325 = None
        unsqueeze_326: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_327: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_143: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_327);  mul_204 = unsqueeze_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_40: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_143)
        mul_205: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_143, sigmoid_40);  add_143 = sigmoid_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_44: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_205, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_205 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_144: "f32[64]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_41: "f32[64]" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_41: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_206: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_329: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_206, -1);  mul_206 = None
        unsqueeze_331: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_62: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_329);  convolution_44 = unsqueeze_329 = None
        mul_207: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_331);  sub_62 = unsqueeze_331 = None
        unsqueeze_332: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_333: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_208: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_333);  mul_207 = unsqueeze_333 = None
        unsqueeze_334: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_335: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_145: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_208, unsqueeze_335);  mul_208 = unsqueeze_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_146: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_145, add_139);  add_145 = add_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_45: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_146, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_147: "f32[256]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_42: "f32[256]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_42: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_209: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_337: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_209, -1);  mul_209 = None
        unsqueeze_339: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_63: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_337);  convolution_45 = unsqueeze_337 = None
        mul_210: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_339);  sub_63 = unsqueeze_339 = None
        unsqueeze_340: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_341: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_211: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_341);  mul_210 = unsqueeze_341 = None
        unsqueeze_342: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_343: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_148: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_343);  mul_211 = unsqueeze_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_41: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_148)
        mul_212: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_148, sigmoid_41);  add_148 = sigmoid_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_46: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_212, arg56_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  mul_212 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_149: "f32[256]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_43: "f32[256]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_43: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_213: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_345: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
        unsqueeze_347: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_64: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_345);  convolution_46 = unsqueeze_345 = None
        mul_214: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_347);  sub_64 = unsqueeze_347 = None
        unsqueeze_348: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_349: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_215: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_349);  mul_214 = unsqueeze_349 = None
        unsqueeze_350: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_351: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_150: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_351);  mul_215 = unsqueeze_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_42: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_150)
        mul_216: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_150, sigmoid_42);  add_150 = sigmoid_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_47: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_216, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_216 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_151: "f32[64]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_44: "f32[64]" = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_44: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_217: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_353: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_217, -1);  mul_217 = None
        unsqueeze_355: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_65: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_353);  convolution_47 = unsqueeze_353 = None
        mul_218: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_355);  sub_65 = unsqueeze_355 = None
        unsqueeze_356: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_357: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_219: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_357);  mul_218 = unsqueeze_357 = None
        unsqueeze_358: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_359: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_152: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_359);  mul_219 = unsqueeze_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_153: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_152, add_146);  add_152 = add_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_48: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_153, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_153 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_154: "f32[256]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_45: "f32[256]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_45: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_361: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_220, -1);  mul_220 = None
        unsqueeze_363: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_66: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_361);  convolution_48 = unsqueeze_361 = None
        mul_221: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_363);  sub_66 = unsqueeze_363 = None
        unsqueeze_364: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_365: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_222: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_365);  mul_221 = unsqueeze_365 = None
        unsqueeze_366: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_367: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_155: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_367);  mul_222 = unsqueeze_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_43: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_155)
        mul_223: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_155, sigmoid_43);  add_155 = sigmoid_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_49: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_223, arg71_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 256);  mul_223 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_156: "f32[256]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_46: "f32[256]" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_46: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_224: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_369: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_224, -1);  mul_224 = None
        unsqueeze_371: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_67: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_369);  convolution_49 = unsqueeze_369 = None
        mul_225: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_371);  sub_67 = unsqueeze_371 = None
        unsqueeze_372: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_373: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_226: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_373);  mul_225 = unsqueeze_373 = None
        unsqueeze_374: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_375: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_157: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_375);  mul_226 = unsqueeze_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_44: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_157)
        mul_227: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_157, sigmoid_44);  add_157 = sigmoid_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_50: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(mul_227, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_227 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_158: "f32[96]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_47: "f32[96]" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_47: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_228: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_377: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_379: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_68: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_377);  convolution_50 = unsqueeze_377 = None
        mul_229: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_379);  sub_68 = unsqueeze_379 = None
        unsqueeze_380: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_381: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_230: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_381);  mul_229 = unsqueeze_381 = None
        unsqueeze_382: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_383: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_159: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_383);  mul_230 = unsqueeze_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_51: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(add_159, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_160: "f32[96]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_48: "f32[96]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_48: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_231: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_385: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
        unsqueeze_387: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_69: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_385);  convolution_51 = unsqueeze_385 = None
        mul_232: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_387);  sub_69 = unsqueeze_387 = None
        unsqueeze_388: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_389: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_233: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_389);  mul_232 = unsqueeze_389 = None
        unsqueeze_390: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_391: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_161: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_391);  mul_233 = unsqueeze_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_45: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_161)
        mul_234: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_161, sigmoid_45);  add_161 = sigmoid_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:234 in forward, code: x = self.conv_1x1(x)
        convolution_52: "f32[8, 144, 32, 32]" = torch.ops.aten.convolution.default(mul_234, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_234 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:249 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        view_109: "f32[18432, 2, 16, 2]" = torch.ops.aten.view.default(convolution_52, [18432, 2, 16, 2]);  convolution_52 = None
        permute_67: "f32[18432, 16, 2, 2]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:251 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_40: "f32[18432, 16, 2, 2]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_110: "f32[8, 144, 256, 4]" = torch.ops.aten.view.default(clone_40, [8, 144, 256, 4]);  clone_40 = None
        permute_68: "f32[8, 4, 256, 144]" = torch.ops.aten.permute.default(view_110, [0, 3, 2, 1]);  view_110 = None
        clone_41: "f32[8, 4, 256, 144]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_111: "f32[32, 256, 144]" = torch.ops.aten.view.default(clone_41, [32, 256, 144]);  clone_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_21 = torch.ops.aten.var_mean.correction(view_111, [2], correction = 0, keepdim = True)
        getitem_105: "f32[32, 256, 1]" = var_mean_21[0]
        getitem_106: "f32[32, 256, 1]" = var_mean_21[1];  var_mean_21 = None
        add_162: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_105, 1e-05);  getitem_105 = None
        rsqrt_21: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_70: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(view_111, getitem_106);  getitem_106 = None
        mul_235: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_21);  sub_70 = rsqrt_21 = None
        mul_236: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_235, arg87_1);  mul_235 = arg87_1 = None
        add_163: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_236, arg88_1);  mul_236 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_112: "f32[8192, 144]" = torch.ops.aten.view.default(add_163, [8192, 144]);  add_163 = None
        permute_69: "f32[144, 432]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_37: "f32[8192, 432]" = torch.ops.aten.addmm.default(arg90_1, view_112, permute_69);  arg90_1 = view_112 = permute_69 = None
        view_113: "f32[32, 256, 432]" = torch.ops.aten.view.default(addmm_37, [32, 256, 432]);  addmm_37 = None
        view_114: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.view.default(view_113, [32, 256, 3, 4, 36]);  view_113 = None
        permute_70: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.permute.default(view_114, [2, 0, 3, 1, 4]);  view_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_70);  permute_70 = None
        getitem_107: "f32[32, 4, 256, 36]" = unbind_9[0]
        getitem_108: "f32[32, 4, 256, 36]" = unbind_9[1]
        getitem_109: "f32[32, 4, 256, 36]" = unbind_9[2];  unbind_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_107, getitem_108, getitem_109, None, False);  getitem_107 = getitem_108 = getitem_109 = None
        getitem_110: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_71: "f32[32, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_110, [0, 2, 1, 3]);  getitem_110 = None
        view_115: "f32[32, 256, 144]" = torch.ops.aten.view.default(permute_71, [32, 256, 144]);  permute_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_116: "f32[8192, 144]" = torch.ops.aten.view.default(view_115, [8192, 144]);  view_115 = None
        permute_72: "f32[144, 144]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_38: "f32[8192, 144]" = torch.ops.aten.addmm.default(arg92_1, view_116, permute_72);  arg92_1 = view_116 = permute_72 = None
        view_117: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_38, [32, 256, 144]);  addmm_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_164: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(view_111, view_117);  view_111 = view_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_22 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_114: "f32[32, 256, 1]" = var_mean_22[0]
        getitem_115: "f32[32, 256, 1]" = var_mean_22[1];  var_mean_22 = None
        add_165: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_22: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_71: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_164, getitem_115);  getitem_115 = None
        mul_237: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_22);  sub_71 = rsqrt_22 = None
        mul_238: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_237, arg93_1);  mul_237 = arg93_1 = None
        add_166: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_238, arg94_1);  mul_238 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_118: "f32[8192, 144]" = torch.ops.aten.view.default(add_166, [8192, 144]);  add_166 = None
        permute_73: "f32[144, 288]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_39: "f32[8192, 288]" = torch.ops.aten.addmm.default(arg96_1, view_118, permute_73);  arg96_1 = view_118 = permute_73 = None
        view_119: "f32[32, 256, 288]" = torch.ops.aten.view.default(addmm_39, [32, 256, 288]);  addmm_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_46: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_119)
        mul_239: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_119, sigmoid_46);  view_119 = sigmoid_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_120: "f32[8192, 288]" = torch.ops.aten.view.default(mul_239, [8192, 288]);  mul_239 = None
        permute_74: "f32[288, 144]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_40: "f32[8192, 144]" = torch.ops.aten.addmm.default(arg98_1, view_120, permute_74);  arg98_1 = view_120 = permute_74 = None
        view_121: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_40, [32, 256, 144]);  addmm_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_167: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_164, view_121);  add_164 = view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_23 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
        getitem_116: "f32[32, 256, 1]" = var_mean_23[0]
        getitem_117: "f32[32, 256, 1]" = var_mean_23[1];  var_mean_23 = None
        add_168: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_23: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        sub_72: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_167, getitem_117);  getitem_117 = None
        mul_240: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_23);  sub_72 = rsqrt_23 = None
        mul_241: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_240, arg99_1);  mul_240 = arg99_1 = None
        add_169: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_241, arg100_1);  mul_241 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_122: "f32[8192, 144]" = torch.ops.aten.view.default(add_169, [8192, 144]);  add_169 = None
        permute_75: "f32[144, 432]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_41: "f32[8192, 432]" = torch.ops.aten.addmm.default(arg102_1, view_122, permute_75);  arg102_1 = view_122 = permute_75 = None
        view_123: "f32[32, 256, 432]" = torch.ops.aten.view.default(addmm_41, [32, 256, 432]);  addmm_41 = None
        view_124: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.view.default(view_123, [32, 256, 3, 4, 36]);  view_123 = None
        permute_76: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.permute.default(view_124, [2, 0, 3, 1, 4]);  view_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_76);  permute_76 = None
        getitem_118: "f32[32, 4, 256, 36]" = unbind_10[0]
        getitem_119: "f32[32, 4, 256, 36]" = unbind_10[1]
        getitem_120: "f32[32, 4, 256, 36]" = unbind_10[2];  unbind_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_118, getitem_119, getitem_120, None, False);  getitem_118 = getitem_119 = getitem_120 = None
        getitem_121: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_77: "f32[32, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3]);  getitem_121 = None
        view_125: "f32[32, 256, 144]" = torch.ops.aten.view.default(permute_77, [32, 256, 144]);  permute_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_126: "f32[8192, 144]" = torch.ops.aten.view.default(view_125, [8192, 144]);  view_125 = None
        permute_78: "f32[144, 144]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_42: "f32[8192, 144]" = torch.ops.aten.addmm.default(arg104_1, view_126, permute_78);  arg104_1 = view_126 = permute_78 = None
        view_127: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_42, [32, 256, 144]);  addmm_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_170: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_167, view_127);  add_167 = view_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_24 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
        getitem_125: "f32[32, 256, 1]" = var_mean_24[0]
        getitem_126: "f32[32, 256, 1]" = var_mean_24[1];  var_mean_24 = None
        add_171: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-05);  getitem_125 = None
        rsqrt_24: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_73: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_170, getitem_126);  getitem_126 = None
        mul_242: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_24);  sub_73 = rsqrt_24 = None
        mul_243: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_242, arg105_1);  mul_242 = arg105_1 = None
        add_172: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_243, arg106_1);  mul_243 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_128: "f32[8192, 144]" = torch.ops.aten.view.default(add_172, [8192, 144]);  add_172 = None
        permute_79: "f32[144, 288]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_43: "f32[8192, 288]" = torch.ops.aten.addmm.default(arg108_1, view_128, permute_79);  arg108_1 = view_128 = permute_79 = None
        view_129: "f32[32, 256, 288]" = torch.ops.aten.view.default(addmm_43, [32, 256, 288]);  addmm_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_47: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_129)
        mul_244: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_129, sigmoid_47);  view_129 = sigmoid_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_130: "f32[8192, 288]" = torch.ops.aten.view.default(mul_244, [8192, 288]);  mul_244 = None
        permute_80: "f32[288, 144]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_44: "f32[8192, 144]" = torch.ops.aten.addmm.default(arg110_1, view_130, permute_80);  arg110_1 = view_130 = permute_80 = None
        view_131: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_44, [32, 256, 144]);  addmm_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_173: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_170, view_131);  add_170 = view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:255 in forward, code: x = self.norm(x)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_127: "f32[32, 256, 1]" = var_mean_25[0]
        getitem_128: "f32[32, 256, 1]" = var_mean_25[1];  var_mean_25 = None
        add_174: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_127, 1e-05);  getitem_127 = None
        rsqrt_25: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_74: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_173, getitem_128);  add_173 = getitem_128 = None
        mul_245: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_25);  sub_74 = rsqrt_25 = None
        mul_246: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_245, arg111_1);  mul_245 = arg111_1 = None
        add_175: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_246, arg112_1);  mul_246 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:259 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_132: "f32[8, 4, 256, 144]" = torch.ops.aten.view.default(add_175, [8, 4, 256, -1]);  add_175 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_81: "f32[8, 144, 256, 4]" = torch.ops.aten.permute.default(view_132, [0, 3, 2, 1]);  view_132 = None
        clone_48: "f32[8, 144, 256, 4]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_133: "f32[18432, 16, 2, 2]" = torch.ops.aten.view.default(clone_48, [18432, 16, 2, 2]);  clone_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:262 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_82: "f32[18432, 2, 16, 2]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        clone_49: "f32[18432, 2, 16, 2]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_134: "f32[8, 144, 32, 32]" = torch.ops.aten.view.default(clone_49, [8, 144, 32, 32]);  clone_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_53: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(view_134, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_134 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_176: "f32[96]" = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_49: "f32[96]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
        reciprocal_49: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_247: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_393: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
        unsqueeze_395: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_75: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_393);  convolution_53 = unsqueeze_393 = None
        mul_248: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_395);  sub_75 = unsqueeze_395 = None
        unsqueeze_396: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_397: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_249: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_397);  mul_248 = unsqueeze_397 = None
        unsqueeze_398: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_399: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_177: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_249, unsqueeze_399);  mul_249 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_48: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_177)
        mul_250: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_177, sigmoid_48);  add_177 = sigmoid_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_3: "f32[8, 192, 32, 32]" = torch.ops.aten.cat.default([add_159, mul_250], 1);  add_159 = mul_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_54: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(cat_3, arg118_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_3 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_178: "f32[96]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_50: "f32[96]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_50: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_251: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_401: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_251, -1);  mul_251 = None
        unsqueeze_403: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_76: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_401);  convolution_54 = unsqueeze_401 = None
        mul_252: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_403);  sub_76 = unsqueeze_403 = None
        unsqueeze_404: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_405: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_253: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_405);  mul_252 = unsqueeze_405 = None
        unsqueeze_406: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_407: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_179: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_407);  mul_253 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_49: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_179)
        mul_254: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_179, sigmoid_49);  add_179 = sigmoid_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_55: "f32[8, 384, 32, 32]" = torch.ops.aten.convolution.default(mul_254, arg123_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_254 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_180: "f32[384]" = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_51: "f32[384]" = torch.ops.aten.sqrt.default(add_180);  add_180 = None
        reciprocal_51: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_255: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_409: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_411: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_77: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_409);  convolution_55 = unsqueeze_409 = None
        mul_256: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_411);  sub_77 = unsqueeze_411 = None
        unsqueeze_412: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_413: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_257: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_413);  mul_256 = unsqueeze_413 = None
        unsqueeze_414: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_415: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_181: "f32[8, 384, 32, 32]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_415);  mul_257 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_50: "f32[8, 384, 32, 32]" = torch.ops.aten.sigmoid.default(add_181)
        mul_258: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(add_181, sigmoid_50);  add_181 = sigmoid_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_56: "f32[8, 384, 16, 16]" = torch.ops.aten.convolution.default(mul_258, arg128_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 384);  mul_258 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_182: "f32[384]" = torch.ops.aten.add.Tensor(arg130_1, 1e-05);  arg130_1 = None
        sqrt_52: "f32[384]" = torch.ops.aten.sqrt.default(add_182);  add_182 = None
        reciprocal_52: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_259: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_417: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_259, -1);  mul_259 = None
        unsqueeze_419: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_78: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_417);  convolution_56 = unsqueeze_417 = None
        mul_260: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_419);  sub_78 = unsqueeze_419 = None
        unsqueeze_420: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_421: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_261: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(mul_260, unsqueeze_421);  mul_260 = unsqueeze_421 = None
        unsqueeze_422: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_423: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_183: "f32[8, 384, 16, 16]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_423);  mul_261 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_51: "f32[8, 384, 16, 16]" = torch.ops.aten.sigmoid.default(add_183)
        mul_262: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(add_183, sigmoid_51);  add_183 = sigmoid_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_57: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(mul_262, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_262 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_184: "f32[128]" = torch.ops.aten.add.Tensor(arg135_1, 1e-05);  arg135_1 = None
        sqrt_53: "f32[128]" = torch.ops.aten.sqrt.default(add_184);  add_184 = None
        reciprocal_53: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_263: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_425: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_263, -1);  mul_263 = None
        unsqueeze_427: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_79: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_425);  convolution_57 = unsqueeze_425 = None
        mul_264: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_427);  sub_79 = unsqueeze_427 = None
        unsqueeze_428: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_429: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_265: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_264, unsqueeze_429);  mul_264 = unsqueeze_429 = None
        unsqueeze_430: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_431: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_185: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_431);  mul_265 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_58: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(add_185, arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_186: "f32[128]" = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_54: "f32[128]" = torch.ops.aten.sqrt.default(add_186);  add_186 = None
        reciprocal_54: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_266: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_433: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_266, -1);  mul_266 = None
        unsqueeze_435: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_80: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_433);  convolution_58 = unsqueeze_433 = None
        mul_267: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_435);  sub_80 = unsqueeze_435 = None
        unsqueeze_436: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_437: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_268: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_267, unsqueeze_437);  mul_267 = unsqueeze_437 = None
        unsqueeze_438: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_439: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_187: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_268, unsqueeze_439);  mul_268 = unsqueeze_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_52: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_187)
        mul_269: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_187, sigmoid_52);  add_187 = sigmoid_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:234 in forward, code: x = self.conv_1x1(x)
        convolution_59: "f32[8, 192, 16, 16]" = torch.ops.aten.convolution.default(mul_269, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_269 = arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:249 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        view_135: "f32[12288, 2, 8, 2]" = torch.ops.aten.view.default(convolution_59, [12288, 2, 8, 2]);  convolution_59 = None
        permute_83: "f32[12288, 8, 2, 2]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:251 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_50: "f32[12288, 8, 2, 2]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_136: "f32[8, 192, 64, 4]" = torch.ops.aten.view.default(clone_50, [8, 192, 64, 4]);  clone_50 = None
        permute_84: "f32[8, 4, 64, 192]" = torch.ops.aten.permute.default(view_136, [0, 3, 2, 1]);  view_136 = None
        clone_51: "f32[8, 4, 64, 192]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_137: "f32[32, 64, 192]" = torch.ops.aten.view.default(clone_51, [32, 64, 192]);  clone_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_26 = torch.ops.aten.var_mean.correction(view_137, [2], correction = 0, keepdim = True)
        getitem_129: "f32[32, 64, 1]" = var_mean_26[0]
        getitem_130: "f32[32, 64, 1]" = var_mean_26[1];  var_mean_26 = None
        add_188: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_129, 1e-05);  getitem_129 = None
        rsqrt_26: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_81: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(view_137, getitem_130);  getitem_130 = None
        mul_270: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_26);  sub_81 = rsqrt_26 = None
        mul_271: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_270, arg144_1);  mul_270 = arg144_1 = None
        add_189: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_271, arg145_1);  mul_271 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_138: "f32[2048, 192]" = torch.ops.aten.view.default(add_189, [2048, 192]);  add_189 = None
        permute_85: "f32[192, 576]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_45: "f32[2048, 576]" = torch.ops.aten.addmm.default(arg147_1, view_138, permute_85);  arg147_1 = view_138 = permute_85 = None
        view_139: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_45, [32, 64, 576]);  addmm_45 = None
        view_140: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_139, [32, 64, 3, 4, 48]);  view_139 = None
        permute_86: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_140, [2, 0, 3, 1, 4]);  view_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_86);  permute_86 = None
        getitem_131: "f32[32, 4, 64, 48]" = unbind_11[0]
        getitem_132: "f32[32, 4, 64, 48]" = unbind_11[1]
        getitem_133: "f32[32, 4, 64, 48]" = unbind_11[2];  unbind_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_131, getitem_132, getitem_133, None, False);  getitem_131 = getitem_132 = getitem_133 = None
        getitem_134: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_87: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        view_141: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_87, [32, 64, 192]);  permute_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_142: "f32[2048, 192]" = torch.ops.aten.view.default(view_141, [2048, 192]);  view_141 = None
        permute_88: "f32[192, 192]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_46: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg149_1, view_142, permute_88);  arg149_1 = view_142 = permute_88 = None
        view_143: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_46, [32, 64, 192]);  addmm_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_190: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(view_137, view_143);  view_137 = view_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_27 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
        getitem_138: "f32[32, 64, 1]" = var_mean_27[0]
        getitem_139: "f32[32, 64, 1]" = var_mean_27[1];  var_mean_27 = None
        add_191: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
        rsqrt_27: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_82: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_190, getitem_139);  getitem_139 = None
        mul_272: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_27);  sub_82 = rsqrt_27 = None
        mul_273: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_272, arg150_1);  mul_272 = arg150_1 = None
        add_192: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_273, arg151_1);  mul_273 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_144: "f32[2048, 192]" = torch.ops.aten.view.default(add_192, [2048, 192]);  add_192 = None
        permute_89: "f32[192, 384]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_47: "f32[2048, 384]" = torch.ops.aten.addmm.default(arg153_1, view_144, permute_89);  arg153_1 = view_144 = permute_89 = None
        view_145: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_47, [32, 64, 384]);  addmm_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_53: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_145)
        mul_274: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_145, sigmoid_53);  view_145 = sigmoid_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_146: "f32[2048, 384]" = torch.ops.aten.view.default(mul_274, [2048, 384]);  mul_274 = None
        permute_90: "f32[384, 192]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_48: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg155_1, view_146, permute_90);  arg155_1 = view_146 = permute_90 = None
        view_147: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_48, [32, 64, 192]);  addmm_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_193: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_190, view_147);  add_190 = view_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_28 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
        getitem_140: "f32[32, 64, 1]" = var_mean_28[0]
        getitem_141: "f32[32, 64, 1]" = var_mean_28[1];  var_mean_28 = None
        add_194: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_28: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_83: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_193, getitem_141);  getitem_141 = None
        mul_275: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_28);  sub_83 = rsqrt_28 = None
        mul_276: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_275, arg156_1);  mul_275 = arg156_1 = None
        add_195: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_276, arg157_1);  mul_276 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_148: "f32[2048, 192]" = torch.ops.aten.view.default(add_195, [2048, 192]);  add_195 = None
        permute_91: "f32[192, 576]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_49: "f32[2048, 576]" = torch.ops.aten.addmm.default(arg159_1, view_148, permute_91);  arg159_1 = view_148 = permute_91 = None
        view_149: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_49, [32, 64, 576]);  addmm_49 = None
        view_150: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_149, [32, 64, 3, 4, 48]);  view_149 = None
        permute_92: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_150, [2, 0, 3, 1, 4]);  view_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_12 = torch.ops.aten.unbind.int(permute_92);  permute_92 = None
        getitem_142: "f32[32, 4, 64, 48]" = unbind_12[0]
        getitem_143: "f32[32, 4, 64, 48]" = unbind_12[1]
        getitem_144: "f32[32, 4, 64, 48]" = unbind_12[2];  unbind_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_142, getitem_143, getitem_144, None, False);  getitem_142 = getitem_143 = getitem_144 = None
        getitem_145: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_93: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None
        view_151: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_93, [32, 64, 192]);  permute_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_152: "f32[2048, 192]" = torch.ops.aten.view.default(view_151, [2048, 192]);  view_151 = None
        permute_94: "f32[192, 192]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_50: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg161_1, view_152, permute_94);  arg161_1 = view_152 = permute_94 = None
        view_153: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_50, [32, 64, 192]);  addmm_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_196: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_193, view_153);  add_193 = view_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_29 = torch.ops.aten.var_mean.correction(add_196, [2], correction = 0, keepdim = True)
        getitem_149: "f32[32, 64, 1]" = var_mean_29[0]
        getitem_150: "f32[32, 64, 1]" = var_mean_29[1];  var_mean_29 = None
        add_197: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_149, 1e-05);  getitem_149 = None
        rsqrt_29: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_84: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_196, getitem_150);  getitem_150 = None
        mul_277: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_29);  sub_84 = rsqrt_29 = None
        mul_278: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_277, arg162_1);  mul_277 = arg162_1 = None
        add_198: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_278, arg163_1);  mul_278 = arg163_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_154: "f32[2048, 192]" = torch.ops.aten.view.default(add_198, [2048, 192]);  add_198 = None
        permute_95: "f32[192, 384]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_51: "f32[2048, 384]" = torch.ops.aten.addmm.default(arg165_1, view_154, permute_95);  arg165_1 = view_154 = permute_95 = None
        view_155: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_51, [32, 64, 384]);  addmm_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_54: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_155)
        mul_279: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_155, sigmoid_54);  view_155 = sigmoid_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_156: "f32[2048, 384]" = torch.ops.aten.view.default(mul_279, [2048, 384]);  mul_279 = None
        permute_96: "f32[384, 192]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_52: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg167_1, view_156, permute_96);  arg167_1 = view_156 = permute_96 = None
        view_157: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_52, [32, 64, 192]);  addmm_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_199: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_196, view_157);  add_196 = view_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_30 = torch.ops.aten.var_mean.correction(add_199, [2], correction = 0, keepdim = True)
        getitem_151: "f32[32, 64, 1]" = var_mean_30[0]
        getitem_152: "f32[32, 64, 1]" = var_mean_30[1];  var_mean_30 = None
        add_200: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_151, 1e-05);  getitem_151 = None
        rsqrt_30: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        sub_85: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_199, getitem_152);  getitem_152 = None
        mul_280: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_30);  sub_85 = rsqrt_30 = None
        mul_281: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_280, arg168_1);  mul_280 = arg168_1 = None
        add_201: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_281, arg169_1);  mul_281 = arg169_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_158: "f32[2048, 192]" = torch.ops.aten.view.default(add_201, [2048, 192]);  add_201 = None
        permute_97: "f32[192, 576]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_53: "f32[2048, 576]" = torch.ops.aten.addmm.default(arg171_1, view_158, permute_97);  arg171_1 = view_158 = permute_97 = None
        view_159: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_53, [32, 64, 576]);  addmm_53 = None
        view_160: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_159, [32, 64, 3, 4, 48]);  view_159 = None
        permute_98: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_160, [2, 0, 3, 1, 4]);  view_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_98);  permute_98 = None
        getitem_153: "f32[32, 4, 64, 48]" = unbind_13[0]
        getitem_154: "f32[32, 4, 64, 48]" = unbind_13[1]
        getitem_155: "f32[32, 4, 64, 48]" = unbind_13[2];  unbind_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_153, getitem_154, getitem_155, None, False);  getitem_153 = getitem_154 = getitem_155 = None
        getitem_156: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_99: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
        view_161: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_99, [32, 64, 192]);  permute_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_162: "f32[2048, 192]" = torch.ops.aten.view.default(view_161, [2048, 192]);  view_161 = None
        permute_100: "f32[192, 192]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_54: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg173_1, view_162, permute_100);  arg173_1 = view_162 = permute_100 = None
        view_163: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_54, [32, 64, 192]);  addmm_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_202: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_199, view_163);  add_199 = view_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_31 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
        getitem_160: "f32[32, 64, 1]" = var_mean_31[0]
        getitem_161: "f32[32, 64, 1]" = var_mean_31[1];  var_mean_31 = None
        add_203: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_31: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_86: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_202, getitem_161);  getitem_161 = None
        mul_282: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_31);  sub_86 = rsqrt_31 = None
        mul_283: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_282, arg174_1);  mul_282 = arg174_1 = None
        add_204: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_283, arg175_1);  mul_283 = arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_164: "f32[2048, 192]" = torch.ops.aten.view.default(add_204, [2048, 192]);  add_204 = None
        permute_101: "f32[192, 384]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_55: "f32[2048, 384]" = torch.ops.aten.addmm.default(arg177_1, view_164, permute_101);  arg177_1 = view_164 = permute_101 = None
        view_165: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_55, [32, 64, 384]);  addmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_55: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_165)
        mul_284: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_165, sigmoid_55);  view_165 = sigmoid_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_166: "f32[2048, 384]" = torch.ops.aten.view.default(mul_284, [2048, 384]);  mul_284 = None
        permute_102: "f32[384, 192]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_56: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg179_1, view_166, permute_102);  arg179_1 = view_166 = permute_102 = None
        view_167: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_56, [32, 64, 192]);  addmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_205: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_202, view_167);  add_202 = view_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_32 = torch.ops.aten.var_mean.correction(add_205, [2], correction = 0, keepdim = True)
        getitem_162: "f32[32, 64, 1]" = var_mean_32[0]
        getitem_163: "f32[32, 64, 1]" = var_mean_32[1];  var_mean_32 = None
        add_206: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_32: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_87: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_205, getitem_163);  getitem_163 = None
        mul_285: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_32);  sub_87 = rsqrt_32 = None
        mul_286: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_285, arg180_1);  mul_285 = arg180_1 = None
        add_207: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_286, arg181_1);  mul_286 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_168: "f32[2048, 192]" = torch.ops.aten.view.default(add_207, [2048, 192]);  add_207 = None
        permute_103: "f32[192, 576]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_57: "f32[2048, 576]" = torch.ops.aten.addmm.default(arg183_1, view_168, permute_103);  arg183_1 = view_168 = permute_103 = None
        view_169: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_57, [32, 64, 576]);  addmm_57 = None
        view_170: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_169, [32, 64, 3, 4, 48]);  view_169 = None
        permute_104: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_170, [2, 0, 3, 1, 4]);  view_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_104);  permute_104 = None
        getitem_164: "f32[32, 4, 64, 48]" = unbind_14[0]
        getitem_165: "f32[32, 4, 64, 48]" = unbind_14[1]
        getitem_166: "f32[32, 4, 64, 48]" = unbind_14[2];  unbind_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_164, getitem_165, getitem_166, None, False);  getitem_164 = getitem_165 = getitem_166 = None
        getitem_167: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_105: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
        view_171: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_105, [32, 64, 192]);  permute_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_172: "f32[2048, 192]" = torch.ops.aten.view.default(view_171, [2048, 192]);  view_171 = None
        permute_106: "f32[192, 192]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_58: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg185_1, view_172, permute_106);  arg185_1 = view_172 = permute_106 = None
        view_173: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_58, [32, 64, 192]);  addmm_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_208: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_205, view_173);  add_205 = view_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_33 = torch.ops.aten.var_mean.correction(add_208, [2], correction = 0, keepdim = True)
        getitem_171: "f32[32, 64, 1]" = var_mean_33[0]
        getitem_172: "f32[32, 64, 1]" = var_mean_33[1];  var_mean_33 = None
        add_209: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_171, 1e-05);  getitem_171 = None
        rsqrt_33: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_88: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_208, getitem_172);  getitem_172 = None
        mul_287: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_33);  sub_88 = rsqrt_33 = None
        mul_288: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_287, arg186_1);  mul_287 = arg186_1 = None
        add_210: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_288, arg187_1);  mul_288 = arg187_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_174: "f32[2048, 192]" = torch.ops.aten.view.default(add_210, [2048, 192]);  add_210 = None
        permute_107: "f32[192, 384]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_59: "f32[2048, 384]" = torch.ops.aten.addmm.default(arg189_1, view_174, permute_107);  arg189_1 = view_174 = permute_107 = None
        view_175: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_59, [32, 64, 384]);  addmm_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_56: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_175)
        mul_289: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_175, sigmoid_56);  view_175 = sigmoid_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_176: "f32[2048, 384]" = torch.ops.aten.view.default(mul_289, [2048, 384]);  mul_289 = None
        permute_108: "f32[384, 192]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_60: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg191_1, view_176, permute_108);  arg191_1 = view_176 = permute_108 = None
        view_177: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_60, [32, 64, 192]);  addmm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_211: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_208, view_177);  add_208 = view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:255 in forward, code: x = self.norm(x)
        var_mean_34 = torch.ops.aten.var_mean.correction(add_211, [2], correction = 0, keepdim = True)
        getitem_173: "f32[32, 64, 1]" = var_mean_34[0]
        getitem_174: "f32[32, 64, 1]" = var_mean_34[1];  var_mean_34 = None
        add_212: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_173, 1e-05);  getitem_173 = None
        rsqrt_34: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_89: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_211, getitem_174);  add_211 = getitem_174 = None
        mul_290: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_34);  sub_89 = rsqrt_34 = None
        mul_291: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_290, arg192_1);  mul_290 = arg192_1 = None
        add_213: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_291, arg193_1);  mul_291 = arg193_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:259 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_178: "f32[8, 4, 64, 192]" = torch.ops.aten.view.default(add_213, [8, 4, 64, -1]);  add_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_109: "f32[8, 192, 64, 4]" = torch.ops.aten.permute.default(view_178, [0, 3, 2, 1]);  view_178 = None
        clone_64: "f32[8, 192, 64, 4]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
        view_179: "f32[12288, 8, 2, 2]" = torch.ops.aten.view.default(clone_64, [12288, 8, 2, 2]);  clone_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:262 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_110: "f32[12288, 2, 8, 2]" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None
        clone_65: "f32[12288, 2, 8, 2]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_180: "f32[8, 192, 16, 16]" = torch.ops.aten.view.default(clone_65, [8, 192, 16, 16]);  clone_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_60: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(view_180, arg194_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_180 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_214: "f32[128]" = torch.ops.aten.add.Tensor(arg196_1, 1e-05);  arg196_1 = None
        sqrt_55: "f32[128]" = torch.ops.aten.sqrt.default(add_214);  add_214 = None
        reciprocal_55: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_292: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_441: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_443: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_90: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_441);  convolution_60 = unsqueeze_441 = None
        mul_293: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_443);  sub_90 = unsqueeze_443 = None
        unsqueeze_444: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_445: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_294: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_445);  mul_293 = unsqueeze_445 = None
        unsqueeze_446: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_447: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_215: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_447);  mul_294 = unsqueeze_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_57: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_215)
        mul_295: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_215, sigmoid_57);  add_215 = sigmoid_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_4: "f32[8, 256, 16, 16]" = torch.ops.aten.cat.default([add_185, mul_295], 1);  add_185 = mul_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_61: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(cat_4, arg199_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_4 = arg199_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_216: "f32[128]" = torch.ops.aten.add.Tensor(arg201_1, 1e-05);  arg201_1 = None
        sqrt_56: "f32[128]" = torch.ops.aten.sqrt.default(add_216);  add_216 = None
        reciprocal_56: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_296: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_449: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_296, -1);  mul_296 = None
        unsqueeze_451: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_91: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_449);  convolution_61 = unsqueeze_449 = None
        mul_297: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_451);  sub_91 = unsqueeze_451 = None
        unsqueeze_452: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_453: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_298: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_297, unsqueeze_453);  mul_297 = unsqueeze_453 = None
        unsqueeze_454: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_455: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_217: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_455);  mul_298 = unsqueeze_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_58: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_217)
        mul_299: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_217, sigmoid_58);  add_217 = sigmoid_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_62: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_299, arg204_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_299 = arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_218: "f32[512]" = torch.ops.aten.add.Tensor(arg206_1, 1e-05);  arg206_1 = None
        sqrt_57: "f32[512]" = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_57: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_300: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_457: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_459: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_92: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_457);  convolution_62 = unsqueeze_457 = None
        mul_301: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_459);  sub_92 = unsqueeze_459 = None
        unsqueeze_460: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_461: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_302: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_461);  mul_301 = unsqueeze_461 = None
        unsqueeze_462: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_463: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_219: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_463);  mul_302 = unsqueeze_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_59: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_219)
        mul_303: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_219, sigmoid_59);  add_219 = sigmoid_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_63: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_303, arg209_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 512);  mul_303 = arg209_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_220: "f32[512]" = torch.ops.aten.add.Tensor(arg211_1, 1e-05);  arg211_1 = None
        sqrt_58: "f32[512]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_58: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_304: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_465: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_304, -1);  mul_304 = None
        unsqueeze_467: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_93: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_465);  convolution_63 = unsqueeze_465 = None
        mul_305: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_467);  sub_93 = unsqueeze_467 = None
        unsqueeze_468: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_469: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_306: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_305, unsqueeze_469);  mul_305 = unsqueeze_469 = None
        unsqueeze_470: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_471: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_221: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_306, unsqueeze_471);  mul_306 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_60: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_221)
        mul_307: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_221, sigmoid_60);  add_221 = sigmoid_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_64: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(mul_307, arg214_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_307 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_222: "f32[160]" = torch.ops.aten.add.Tensor(arg216_1, 1e-05);  arg216_1 = None
        sqrt_59: "f32[160]" = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_59: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_308: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_473: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_475: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_94: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_473);  convolution_64 = unsqueeze_473 = None
        mul_309: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_475);  sub_94 = unsqueeze_475 = None
        unsqueeze_476: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_477: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_310: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_477);  mul_309 = unsqueeze_477 = None
        unsqueeze_478: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_479: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_223: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_479);  mul_310 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_65: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(add_223, arg219_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg219_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_224: "f32[160]" = torch.ops.aten.add.Tensor(arg221_1, 1e-05);  arg221_1 = None
        sqrt_60: "f32[160]" = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_60: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_311: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_481: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_311, -1);  mul_311 = None
        unsqueeze_483: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_95: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_481);  convolution_65 = unsqueeze_481 = None
        mul_312: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_483);  sub_95 = unsqueeze_483 = None
        unsqueeze_484: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_485: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_313: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_312, unsqueeze_485);  mul_312 = unsqueeze_485 = None
        unsqueeze_486: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_487: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_225: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_313, unsqueeze_487);  mul_313 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_61: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_225)
        mul_314: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_225, sigmoid_61);  add_225 = sigmoid_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:234 in forward, code: x = self.conv_1x1(x)
        convolution_66: "f32[8, 240, 8, 8]" = torch.ops.aten.convolution.default(mul_314, arg224_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_314 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:249 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        view_181: "f32[7680, 2, 4, 2]" = torch.ops.aten.view.default(convolution_66, [7680, 2, 4, 2]);  convolution_66 = None
        permute_111: "f32[7680, 4, 2, 2]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:251 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_66: "f32[7680, 4, 2, 2]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_182: "f32[8, 240, 16, 4]" = torch.ops.aten.view.default(clone_66, [8, 240, 16, 4]);  clone_66 = None
        permute_112: "f32[8, 4, 16, 240]" = torch.ops.aten.permute.default(view_182, [0, 3, 2, 1]);  view_182 = None
        clone_67: "f32[8, 4, 16, 240]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_183: "f32[32, 16, 240]" = torch.ops.aten.view.default(clone_67, [32, 16, 240]);  clone_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_35 = torch.ops.aten.var_mean.correction(view_183, [2], correction = 0, keepdim = True)
        getitem_175: "f32[32, 16, 1]" = var_mean_35[0]
        getitem_176: "f32[32, 16, 1]" = var_mean_35[1];  var_mean_35 = None
        add_226: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_175, 1e-05);  getitem_175 = None
        rsqrt_35: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_96: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(view_183, getitem_176);  getitem_176 = None
        mul_315: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_35);  sub_96 = rsqrt_35 = None
        mul_316: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_315, arg225_1);  mul_315 = arg225_1 = None
        add_227: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_316, arg226_1);  mul_316 = arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_184: "f32[512, 240]" = torch.ops.aten.view.default(add_227, [512, 240]);  add_227 = None
        permute_113: "f32[240, 720]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_61: "f32[512, 720]" = torch.ops.aten.addmm.default(arg228_1, view_184, permute_113);  arg228_1 = view_184 = permute_113 = None
        view_185: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_61, [32, 16, 720]);  addmm_61 = None
        view_186: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_185, [32, 16, 3, 4, 60]);  view_185 = None
        permute_114: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_114);  permute_114 = None
        getitem_177: "f32[32, 4, 16, 60]" = unbind_15[0]
        getitem_178: "f32[32, 4, 16, 60]" = unbind_15[1]
        getitem_179: "f32[32, 4, 16, 60]" = unbind_15[2];  unbind_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_177, getitem_178, getitem_179, None, False);  getitem_177 = getitem_178 = getitem_179 = None
        getitem_180: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_115: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
        view_187: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_115, [32, 16, 240]);  permute_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_188: "f32[512, 240]" = torch.ops.aten.view.default(view_187, [512, 240]);  view_187 = None
        permute_116: "f32[240, 240]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_62: "f32[512, 240]" = torch.ops.aten.addmm.default(arg230_1, view_188, permute_116);  arg230_1 = view_188 = permute_116 = None
        view_189: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_62, [32, 16, 240]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_228: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(view_183, view_189);  view_183 = view_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_36 = torch.ops.aten.var_mean.correction(add_228, [2], correction = 0, keepdim = True)
        getitem_184: "f32[32, 16, 1]" = var_mean_36[0]
        getitem_185: "f32[32, 16, 1]" = var_mean_36[1];  var_mean_36 = None
        add_229: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_36: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_97: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_228, getitem_185);  getitem_185 = None
        mul_317: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_36);  sub_97 = rsqrt_36 = None
        mul_318: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_317, arg231_1);  mul_317 = arg231_1 = None
        add_230: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_318, arg232_1);  mul_318 = arg232_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_190: "f32[512, 240]" = torch.ops.aten.view.default(add_230, [512, 240]);  add_230 = None
        permute_117: "f32[240, 480]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_63: "f32[512, 480]" = torch.ops.aten.addmm.default(arg234_1, view_190, permute_117);  arg234_1 = view_190 = permute_117 = None
        view_191: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_63, [32, 16, 480]);  addmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_62: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_191)
        mul_319: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_191, sigmoid_62);  view_191 = sigmoid_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_192: "f32[512, 480]" = torch.ops.aten.view.default(mul_319, [512, 480]);  mul_319 = None
        permute_118: "f32[480, 240]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_64: "f32[512, 240]" = torch.ops.aten.addmm.default(arg236_1, view_192, permute_118);  arg236_1 = view_192 = permute_118 = None
        view_193: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_64, [32, 16, 240]);  addmm_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_231: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_228, view_193);  add_228 = view_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_37 = torch.ops.aten.var_mean.correction(add_231, [2], correction = 0, keepdim = True)
        getitem_186: "f32[32, 16, 1]" = var_mean_37[0]
        getitem_187: "f32[32, 16, 1]" = var_mean_37[1];  var_mean_37 = None
        add_232: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_37: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        sub_98: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_231, getitem_187);  getitem_187 = None
        mul_320: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_37);  sub_98 = rsqrt_37 = None
        mul_321: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_320, arg237_1);  mul_320 = arg237_1 = None
        add_233: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_321, arg238_1);  mul_321 = arg238_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_194: "f32[512, 240]" = torch.ops.aten.view.default(add_233, [512, 240]);  add_233 = None
        permute_119: "f32[240, 720]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_65: "f32[512, 720]" = torch.ops.aten.addmm.default(arg240_1, view_194, permute_119);  arg240_1 = view_194 = permute_119 = None
        view_195: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_65, [32, 16, 720]);  addmm_65 = None
        view_196: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_195, [32, 16, 3, 4, 60]);  view_195 = None
        permute_120: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_196, [2, 0, 3, 1, 4]);  view_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_120);  permute_120 = None
        getitem_188: "f32[32, 4, 16, 60]" = unbind_16[0]
        getitem_189: "f32[32, 4, 16, 60]" = unbind_16[1]
        getitem_190: "f32[32, 4, 16, 60]" = unbind_16[2];  unbind_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_188, getitem_189, getitem_190, None, False);  getitem_188 = getitem_189 = getitem_190 = None
        getitem_191: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_121: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_191, [0, 2, 1, 3]);  getitem_191 = None
        view_197: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_121, [32, 16, 240]);  permute_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_198: "f32[512, 240]" = torch.ops.aten.view.default(view_197, [512, 240]);  view_197 = None
        permute_122: "f32[240, 240]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_66: "f32[512, 240]" = torch.ops.aten.addmm.default(arg242_1, view_198, permute_122);  arg242_1 = view_198 = permute_122 = None
        view_199: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_66, [32, 16, 240]);  addmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_234: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_231, view_199);  add_231 = view_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_38 = torch.ops.aten.var_mean.correction(add_234, [2], correction = 0, keepdim = True)
        getitem_195: "f32[32, 16, 1]" = var_mean_38[0]
        getitem_196: "f32[32, 16, 1]" = var_mean_38[1];  var_mean_38 = None
        add_235: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_195, 1e-05);  getitem_195 = None
        rsqrt_38: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
        sub_99: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_234, getitem_196);  getitem_196 = None
        mul_322: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_38);  sub_99 = rsqrt_38 = None
        mul_323: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_322, arg243_1);  mul_322 = arg243_1 = None
        add_236: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_323, arg244_1);  mul_323 = arg244_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_200: "f32[512, 240]" = torch.ops.aten.view.default(add_236, [512, 240]);  add_236 = None
        permute_123: "f32[240, 480]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_67: "f32[512, 480]" = torch.ops.aten.addmm.default(arg246_1, view_200, permute_123);  arg246_1 = view_200 = permute_123 = None
        view_201: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_67, [32, 16, 480]);  addmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_63: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_201)
        mul_324: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_201, sigmoid_63);  view_201 = sigmoid_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_202: "f32[512, 480]" = torch.ops.aten.view.default(mul_324, [512, 480]);  mul_324 = None
        permute_124: "f32[480, 240]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_68: "f32[512, 240]" = torch.ops.aten.addmm.default(arg248_1, view_202, permute_124);  arg248_1 = view_202 = permute_124 = None
        view_203: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_68, [32, 16, 240]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_237: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_234, view_203);  add_234 = view_203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        var_mean_39 = torch.ops.aten.var_mean.correction(add_237, [2], correction = 0, keepdim = True)
        getitem_197: "f32[32, 16, 1]" = var_mean_39[0]
        getitem_198: "f32[32, 16, 1]" = var_mean_39[1];  var_mean_39 = None
        add_238: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_197, 1e-05);  getitem_197 = None
        rsqrt_39: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
        sub_100: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_237, getitem_198);  getitem_198 = None
        mul_325: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_39);  sub_100 = rsqrt_39 = None
        mul_326: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_325, arg249_1);  mul_325 = arg249_1 = None
        add_239: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_326, arg250_1);  mul_326 = arg250_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:87 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_204: "f32[512, 240]" = torch.ops.aten.view.default(add_239, [512, 240]);  add_239 = None
        permute_125: "f32[240, 720]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_69: "f32[512, 720]" = torch.ops.aten.addmm.default(arg252_1, view_204, permute_125);  arg252_1 = view_204 = permute_125 = None
        view_205: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_69, [32, 16, 720]);  addmm_69 = None
        view_206: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_205, [32, 16, 3, 4, 60]);  view_205 = None
        permute_126: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_206, [2, 0, 3, 1, 4]);  view_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:88 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_126);  permute_126 = None
        getitem_199: "f32[32, 4, 16, 60]" = unbind_17[0]
        getitem_200: "f32[32, 4, 16, 60]" = unbind_17[1]
        getitem_201: "f32[32, 4, 16, 60]" = unbind_17[2];  unbind_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:92 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_199, getitem_200, getitem_201, None, False);  getitem_199 = getitem_200 = getitem_201 = None
        getitem_202: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:103 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_127: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
        view_207: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_127, [32, 16, 240]);  permute_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:104 in forward, code: x = self.proj(x)
        view_208: "f32[512, 240]" = torch.ops.aten.view.default(view_207, [512, 240]);  view_207 = None
        permute_128: "f32[240, 240]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_70: "f32[512, 240]" = torch.ops.aten.addmm.default(arg254_1, view_208, permute_128);  arg254_1 = view_208 = permute_128 = None
        view_209: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_70, [32, 16, 240]);  addmm_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:165 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        add_240: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_237, view_209);  add_237 = view_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_40 = torch.ops.aten.var_mean.correction(add_240, [2], correction = 0, keepdim = True)
        getitem_206: "f32[32, 16, 1]" = var_mean_40[0]
        getitem_207: "f32[32, 16, 1]" = var_mean_40[1];  var_mean_40 = None
        add_241: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_40: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
        sub_101: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_240, getitem_207);  getitem_207 = None
        mul_327: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_40);  sub_101 = rsqrt_40 = None
        mul_328: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_327, arg255_1);  mul_327 = arg255_1 = None
        add_242: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_328, arg256_1);  mul_328 = arg256_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_210: "f32[512, 240]" = torch.ops.aten.view.default(add_242, [512, 240]);  add_242 = None
        permute_129: "f32[240, 480]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_71: "f32[512, 480]" = torch.ops.aten.addmm.default(arg258_1, view_210, permute_129);  arg258_1 = view_210 = permute_129 = None
        view_211: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_71, [32, 16, 480]);  addmm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:43 in forward, code: x = self.act(x)
        sigmoid_64: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_211)
        mul_329: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_211, sigmoid_64);  view_211 = sigmoid_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_212: "f32[512, 480]" = torch.ops.aten.view.default(mul_329, [512, 480]);  mul_329 = None
        permute_130: "f32[480, 240]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_72: "f32[512, 240]" = torch.ops.aten.addmm.default(arg260_1, view_212, permute_130);  arg260_1 = view_212 = permute_130 = None
        view_213: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_72, [32, 16, 240]);  addmm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/vision_transformer.py:166 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_243: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_240, view_213);  add_240 = view_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:255 in forward, code: x = self.norm(x)
        var_mean_41 = torch.ops.aten.var_mean.correction(add_243, [2], correction = 0, keepdim = True)
        getitem_208: "f32[32, 16, 1]" = var_mean_41[0]
        getitem_209: "f32[32, 16, 1]" = var_mean_41[1];  var_mean_41 = None
        add_244: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_41: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_102: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_243, getitem_209);  add_243 = getitem_209 = None
        mul_330: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_41);  sub_102 = rsqrt_41 = None
        mul_331: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_330, arg261_1);  mul_330 = arg261_1 = None
        add_245: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_331, arg262_1);  mul_331 = arg262_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:259 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_214: "f32[8, 4, 16, 240]" = torch.ops.aten.view.default(add_245, [8, 4, 16, -1]);  add_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_131: "f32[8, 240, 16, 4]" = torch.ops.aten.permute.default(view_214, [0, 3, 2, 1]);  view_214 = None
        clone_77: "f32[8, 240, 16, 4]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
        view_215: "f32[7680, 4, 2, 2]" = torch.ops.aten.view.default(clone_77, [7680, 4, 2, 2]);  clone_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:262 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_132: "f32[7680, 2, 4, 2]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        clone_78: "f32[7680, 2, 4, 2]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_216: "f32[8, 240, 8, 8]" = torch.ops.aten.view.default(clone_78, [8, 240, 8, 8]);  clone_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_67: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(view_216, arg263_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_216 = arg263_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_246: "f32[160]" = torch.ops.aten.add.Tensor(arg265_1, 1e-05);  arg265_1 = None
        sqrt_61: "f32[160]" = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_61: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_332: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_489: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_332, -1);  mul_332 = None
        unsqueeze_491: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_103: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_489);  convolution_67 = unsqueeze_489 = None
        mul_333: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_491);  sub_103 = unsqueeze_491 = None
        unsqueeze_492: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_493: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_334: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_333, unsqueeze_493);  mul_333 = unsqueeze_493 = None
        unsqueeze_494: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_495: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_247: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_334, unsqueeze_495);  mul_334 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_65: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_247)
        mul_335: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_247, sigmoid_65);  add_247 = sigmoid_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_5: "f32[8, 320, 8, 8]" = torch.ops.aten.cat.default([add_223, mul_335], 1);  add_223 = mul_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_68: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(cat_5, arg268_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_5 = arg268_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_248: "f32[160]" = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
        sqrt_62: "f32[160]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_62: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_336: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_497: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_499: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_104: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_497);  convolution_68 = unsqueeze_497 = None
        mul_337: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_499);  sub_104 = unsqueeze_499 = None
        unsqueeze_500: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_501: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_338: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_501);  mul_337 = unsqueeze_501 = None
        unsqueeze_502: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_503: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_249: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_503);  mul_338 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_66: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_249)
        mul_339: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_249, sigmoid_66);  add_249 = sigmoid_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_69: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(mul_339, arg273_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_339 = arg273_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_250: "f32[640]" = torch.ops.aten.add.Tensor(arg275_1, 1e-05);  arg275_1 = None
        sqrt_63: "f32[640]" = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_63: "f32[640]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_340: "f32[640]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_505: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(mul_340, -1);  mul_340 = None
        unsqueeze_507: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_105: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_505);  convolution_69 = unsqueeze_505 = None
        mul_341: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_507);  sub_105 = unsqueeze_507 = None
        unsqueeze_508: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_509: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_342: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_341, unsqueeze_509);  mul_341 = unsqueeze_509 = None
        unsqueeze_510: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_511: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_251: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_511);  mul_342 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_67: "f32[8, 640, 8, 8]" = torch.ops.aten.sigmoid.default(add_251)
        mul_343: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(add_251, sigmoid_67);  add_251 = sigmoid_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 640, 1, 1]" = torch.ops.aten.mean.dim(mul_343, [-1, -2], True);  mul_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_217: "f32[8, 640]" = torch.ops.aten.view.default(mean_1, [8, 640]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_133: "f32[640, 1000]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_73: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg279_1, view_217, permute_133);  arg279_1 = view_217 = permute_133 = None
        return (addmm_73,)
        