class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[32, 1, 3, 3]", arg7_1: "f32[32]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[8, 32, 1, 1]", arg12_1: "f32[8]", arg13_1: "f32[32, 8, 1, 1]", arg14_1: "f32[32]", arg15_1: "f32[16, 32, 1, 1]", arg16_1: "f32[16]", arg17_1: "f32[16]", arg18_1: "f32[16]", arg19_1: "f32[16]", arg20_1: "f32[96, 16, 1, 1]", arg21_1: "f32[96]", arg22_1: "f32[96]", arg23_1: "f32[96]", arg24_1: "f32[96]", arg25_1: "f32[96, 1, 3, 3]", arg26_1: "f32[96]", arg27_1: "f32[96]", arg28_1: "f32[96]", arg29_1: "f32[96]", arg30_1: "f32[4, 96, 1, 1]", arg31_1: "f32[4]", arg32_1: "f32[96, 4, 1, 1]", arg33_1: "f32[96]", arg34_1: "f32[24, 96, 1, 1]", arg35_1: "f32[24]", arg36_1: "f32[24]", arg37_1: "f32[24]", arg38_1: "f32[24]", arg39_1: "f32[144, 24, 1, 1]", arg40_1: "f32[144]", arg41_1: "f32[144]", arg42_1: "f32[144]", arg43_1: "f32[144]", arg44_1: "f32[144, 1, 3, 3]", arg45_1: "f32[144]", arg46_1: "f32[144]", arg47_1: "f32[144]", arg48_1: "f32[144]", arg49_1: "f32[6, 144, 1, 1]", arg50_1: "f32[6]", arg51_1: "f32[144, 6, 1, 1]", arg52_1: "f32[144]", arg53_1: "f32[24, 144, 1, 1]", arg54_1: "f32[24]", arg55_1: "f32[24]", arg56_1: "f32[24]", arg57_1: "f32[24]", arg58_1: "f32[144, 24, 1, 1]", arg59_1: "f32[144]", arg60_1: "f32[144]", arg61_1: "f32[144]", arg62_1: "f32[144]", arg63_1: "f32[144, 1, 5, 5]", arg64_1: "f32[144]", arg65_1: "f32[144]", arg66_1: "f32[144]", arg67_1: "f32[144]", arg68_1: "f32[6, 144, 1, 1]", arg69_1: "f32[6]", arg70_1: "f32[144, 6, 1, 1]", arg71_1: "f32[144]", arg72_1: "f32[40, 144, 1, 1]", arg73_1: "f32[40]", arg74_1: "f32[40]", arg75_1: "f32[40]", arg76_1: "f32[40]", arg77_1: "f32[240, 40, 1, 1]", arg78_1: "f32[240]", arg79_1: "f32[240]", arg80_1: "f32[240]", arg81_1: "f32[240]", arg82_1: "f32[240, 1, 5, 5]", arg83_1: "f32[240]", arg84_1: "f32[240]", arg85_1: "f32[240]", arg86_1: "f32[240]", arg87_1: "f32[10, 240, 1, 1]", arg88_1: "f32[10]", arg89_1: "f32[240, 10, 1, 1]", arg90_1: "f32[240]", arg91_1: "f32[40, 240, 1, 1]", arg92_1: "f32[40]", arg93_1: "f32[40]", arg94_1: "f32[40]", arg95_1: "f32[40]", arg96_1: "f32[240, 40, 1, 1]", arg97_1: "f32[240]", arg98_1: "f32[240]", arg99_1: "f32[240]", arg100_1: "f32[240]", arg101_1: "f32[240, 1, 3, 3]", arg102_1: "f32[240]", arg103_1: "f32[240]", arg104_1: "f32[240]", arg105_1: "f32[240]", arg106_1: "f32[10, 240, 1, 1]", arg107_1: "f32[10]", arg108_1: "f32[240, 10, 1, 1]", arg109_1: "f32[240]", arg110_1: "f32[80, 240, 1, 1]", arg111_1: "f32[80]", arg112_1: "f32[80]", arg113_1: "f32[80]", arg114_1: "f32[80]", arg115_1: "f32[480, 80, 1, 1]", arg116_1: "f32[480]", arg117_1: "f32[480]", arg118_1: "f32[480]", arg119_1: "f32[480]", arg120_1: "f32[480, 1, 3, 3]", arg121_1: "f32[480]", arg122_1: "f32[480]", arg123_1: "f32[480]", arg124_1: "f32[480]", arg125_1: "f32[20, 480, 1, 1]", arg126_1: "f32[20]", arg127_1: "f32[480, 20, 1, 1]", arg128_1: "f32[480]", arg129_1: "f32[80, 480, 1, 1]", arg130_1: "f32[80]", arg131_1: "f32[80]", arg132_1: "f32[80]", arg133_1: "f32[80]", arg134_1: "f32[480, 80, 1, 1]", arg135_1: "f32[480]", arg136_1: "f32[480]", arg137_1: "f32[480]", arg138_1: "f32[480]", arg139_1: "f32[480, 1, 3, 3]", arg140_1: "f32[480]", arg141_1: "f32[480]", arg142_1: "f32[480]", arg143_1: "f32[480]", arg144_1: "f32[20, 480, 1, 1]", arg145_1: "f32[20]", arg146_1: "f32[480, 20, 1, 1]", arg147_1: "f32[480]", arg148_1: "f32[80, 480, 1, 1]", arg149_1: "f32[80]", arg150_1: "f32[80]", arg151_1: "f32[80]", arg152_1: "f32[80]", arg153_1: "f32[480, 80, 1, 1]", arg154_1: "f32[480]", arg155_1: "f32[480]", arg156_1: "f32[480]", arg157_1: "f32[480]", arg158_1: "f32[480, 1, 5, 5]", arg159_1: "f32[480]", arg160_1: "f32[480]", arg161_1: "f32[480]", arg162_1: "f32[480]", arg163_1: "f32[20, 480, 1, 1]", arg164_1: "f32[20]", arg165_1: "f32[480, 20, 1, 1]", arg166_1: "f32[480]", arg167_1: "f32[112, 480, 1, 1]", arg168_1: "f32[112]", arg169_1: "f32[112]", arg170_1: "f32[112]", arg171_1: "f32[112]", arg172_1: "f32[672, 112, 1, 1]", arg173_1: "f32[672]", arg174_1: "f32[672]", arg175_1: "f32[672]", arg176_1: "f32[672]", arg177_1: "f32[672, 1, 5, 5]", arg178_1: "f32[672]", arg179_1: "f32[672]", arg180_1: "f32[672]", arg181_1: "f32[672]", arg182_1: "f32[28, 672, 1, 1]", arg183_1: "f32[28]", arg184_1: "f32[672, 28, 1, 1]", arg185_1: "f32[672]", arg186_1: "f32[112, 672, 1, 1]", arg187_1: "f32[112]", arg188_1: "f32[112]", arg189_1: "f32[112]", arg190_1: "f32[112]", arg191_1: "f32[672, 112, 1, 1]", arg192_1: "f32[672]", arg193_1: "f32[672]", arg194_1: "f32[672]", arg195_1: "f32[672]", arg196_1: "f32[672, 1, 5, 5]", arg197_1: "f32[672]", arg198_1: "f32[672]", arg199_1: "f32[672]", arg200_1: "f32[672]", arg201_1: "f32[28, 672, 1, 1]", arg202_1: "f32[28]", arg203_1: "f32[672, 28, 1, 1]", arg204_1: "f32[672]", arg205_1: "f32[112, 672, 1, 1]", arg206_1: "f32[112]", arg207_1: "f32[112]", arg208_1: "f32[112]", arg209_1: "f32[112]", arg210_1: "f32[672, 112, 1, 1]", arg211_1: "f32[672]", arg212_1: "f32[672]", arg213_1: "f32[672]", arg214_1: "f32[672]", arg215_1: "f32[672, 1, 5, 5]", arg216_1: "f32[672]", arg217_1: "f32[672]", arg218_1: "f32[672]", arg219_1: "f32[672]", arg220_1: "f32[28, 672, 1, 1]", arg221_1: "f32[28]", arg222_1: "f32[672, 28, 1, 1]", arg223_1: "f32[672]", arg224_1: "f32[192, 672, 1, 1]", arg225_1: "f32[192]", arg226_1: "f32[192]", arg227_1: "f32[192]", arg228_1: "f32[192]", arg229_1: "f32[1152, 192, 1, 1]", arg230_1: "f32[1152]", arg231_1: "f32[1152]", arg232_1: "f32[1152]", arg233_1: "f32[1152]", arg234_1: "f32[1152, 1, 5, 5]", arg235_1: "f32[1152]", arg236_1: "f32[1152]", arg237_1: "f32[1152]", arg238_1: "f32[1152]", arg239_1: "f32[48, 1152, 1, 1]", arg240_1: "f32[48]", arg241_1: "f32[1152, 48, 1, 1]", arg242_1: "f32[1152]", arg243_1: "f32[192, 1152, 1, 1]", arg244_1: "f32[192]", arg245_1: "f32[192]", arg246_1: "f32[192]", arg247_1: "f32[192]", arg248_1: "f32[1152, 192, 1, 1]", arg249_1: "f32[1152]", arg250_1: "f32[1152]", arg251_1: "f32[1152]", arg252_1: "f32[1152]", arg253_1: "f32[1152, 1, 5, 5]", arg254_1: "f32[1152]", arg255_1: "f32[1152]", arg256_1: "f32[1152]", arg257_1: "f32[1152]", arg258_1: "f32[48, 1152, 1, 1]", arg259_1: "f32[48]", arg260_1: "f32[1152, 48, 1, 1]", arg261_1: "f32[1152]", arg262_1: "f32[192, 1152, 1, 1]", arg263_1: "f32[192]", arg264_1: "f32[192]", arg265_1: "f32[192]", arg266_1: "f32[192]", arg267_1: "f32[1152, 192, 1, 1]", arg268_1: "f32[1152]", arg269_1: "f32[1152]", arg270_1: "f32[1152]", arg271_1: "f32[1152]", arg272_1: "f32[1152, 1, 5, 5]", arg273_1: "f32[1152]", arg274_1: "f32[1152]", arg275_1: "f32[1152]", arg276_1: "f32[1152]", arg277_1: "f32[48, 1152, 1, 1]", arg278_1: "f32[48]", arg279_1: "f32[1152, 48, 1, 1]", arg280_1: "f32[1152]", arg281_1: "f32[192, 1152, 1, 1]", arg282_1: "f32[192]", arg283_1: "f32[192]", arg284_1: "f32[192]", arg285_1: "f32[192]", arg286_1: "f32[1152, 192, 1, 1]", arg287_1: "f32[1152]", arg288_1: "f32[1152]", arg289_1: "f32[1152]", arg290_1: "f32[1152]", arg291_1: "f32[1152, 1, 3, 3]", arg292_1: "f32[1152]", arg293_1: "f32[1152]", arg294_1: "f32[1152]", arg295_1: "f32[1152]", arg296_1: "f32[48, 1152, 1, 1]", arg297_1: "f32[48]", arg298_1: "f32[1152, 48, 1, 1]", arg299_1: "f32[1152]", arg300_1: "f32[320, 1152, 1, 1]", arg301_1: "f32[320]", arg302_1: "f32[320]", arg303_1: "f32[320]", arg304_1: "f32[320]", arg305_1: "f32[1280, 320, 1, 1]", arg306_1: "f32[1280]", arg307_1: "f32[1280]", arg308_1: "f32[1280]", arg309_1: "f32[1280]", arg310_1: "f32[1000, 1280]", arg311_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_5: "f32[8, 3, 225, 225]" = torch.ops.aten.constant_pad_nd.default(arg1_1, [0, 1, 0, 1], 0.0);  arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv2d_same.py:27 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_81: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(constant_pad_nd_5, arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd_5 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_107: "f32[32]" = torch.ops.aten.add.Tensor(arg3_1, 0.001);  arg3_1 = None
        sqrt_49: "f32[32]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
        reciprocal_49: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_212: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_393: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_212, -1);  mul_212 = None
        unsqueeze_395: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_49: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_393);  convolution_81 = unsqueeze_393 = None
        mul_213: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_397: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_214: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_397);  mul_213 = unsqueeze_397 = None
        unsqueeze_398: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_399: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_108: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_399);  mul_214 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_65: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_108)
        mul_215: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_108, sigmoid_65);  add_108 = sigmoid_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_82: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(mul_215, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  mul_215 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_109: "f32[32]" = torch.ops.aten.add.Tensor(arg8_1, 0.001);  arg8_1 = None
        sqrt_50: "f32[32]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_50: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_216: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_401: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_403: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_50: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_401);  convolution_82 = unsqueeze_401 = None
        mul_217: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_405: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_218: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_405);  mul_217 = unsqueeze_405 = None
        unsqueeze_406: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_407: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_110: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_407);  mul_218 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_66: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_110)
        mul_219: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_110, sigmoid_66);  add_110 = sigmoid_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_17: "f32[8, 32, 1, 1]" = torch.ops.aten.mean.dim(mul_219, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_83: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_17, arg11_1, arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg11_1 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_67: "f32[8, 8, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_83)
        mul_220: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_83, sigmoid_67);  convolution_83 = sigmoid_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_84: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mul_220, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_220 = arg13_1 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_68: "f32[8, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_84);  convolution_84 = None
        mul_221: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_219, sigmoid_68);  mul_219 = sigmoid_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_85: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(mul_221, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_221 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_111: "f32[16]" = torch.ops.aten.add.Tensor(arg17_1, 0.001);  arg17_1 = None
        sqrt_51: "f32[16]" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_51: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_222: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_409: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
        unsqueeze_411: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_51: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_409);  convolution_85 = unsqueeze_409 = None
        mul_223: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_413: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_224: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_413);  mul_223 = unsqueeze_413 = None
        unsqueeze_414: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_415: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_112: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_415);  mul_224 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_86: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(add_112, arg20_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_112 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_113: "f32[96]" = torch.ops.aten.add.Tensor(arg22_1, 0.001);  arg22_1 = None
        sqrt_52: "f32[96]" = torch.ops.aten.sqrt.default(add_113);  add_113 = None
        reciprocal_52: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_225: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_417: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
        unsqueeze_419: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_52: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_417);  convolution_86 = unsqueeze_417 = None
        mul_226: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_421: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_227: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_421);  mul_226 = unsqueeze_421 = None
        unsqueeze_422: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_423: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_114: "f32[8, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_423);  mul_227 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_69: "f32[8, 96, 112, 112]" = torch.ops.aten.sigmoid.default(add_114)
        mul_228: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(add_114, sigmoid_69);  add_114 = sigmoid_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "f32[8, 96, 113, 113]" = torch.ops.aten.constant_pad_nd.default(mul_228, [0, 1, 0, 1], 0.0);  mul_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv2d_same.py:27 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_87: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_6, arg25_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 96);  constant_pad_nd_6 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_115: "f32[96]" = torch.ops.aten.add.Tensor(arg27_1, 0.001);  arg27_1 = None
        sqrt_53: "f32[96]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
        reciprocal_53: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_229: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_425: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_229, -1);  mul_229 = None
        unsqueeze_427: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_53: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_425);  convolution_87 = unsqueeze_425 = None
        mul_230: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_429: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_231: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_429);  mul_230 = unsqueeze_429 = None
        unsqueeze_430: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_431: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_116: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_231, unsqueeze_431);  mul_231 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_70: "f32[8, 96, 56, 56]" = torch.ops.aten.sigmoid.default(add_116)
        mul_232: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_116, sigmoid_70);  add_116 = sigmoid_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_18: "f32[8, 96, 1, 1]" = torch.ops.aten.mean.dim(mul_232, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_88: "f32[8, 4, 1, 1]" = torch.ops.aten.convolution.default(mean_18, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg30_1 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_71: "f32[8, 4, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_88)
        mul_233: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_88, sigmoid_71);  convolution_88 = sigmoid_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_89: "f32[8, 96, 1, 1]" = torch.ops.aten.convolution.default(mul_233, arg32_1, arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_233 = arg32_1 = arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_72: "f32[8, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_89);  convolution_89 = None
        mul_234: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_232, sigmoid_72);  mul_232 = sigmoid_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_90: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_234, arg34_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_234 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_117: "f32[24]" = torch.ops.aten.add.Tensor(arg36_1, 0.001);  arg36_1 = None
        sqrt_54: "f32[24]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_54: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_235: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_433: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_235, -1);  mul_235 = None
        unsqueeze_435: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_54: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_433);  convolution_90 = unsqueeze_433 = None
        mul_236: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
        unsqueeze_436: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_437: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_237: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_236, unsqueeze_437);  mul_236 = unsqueeze_437 = None
        unsqueeze_438: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_439: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_118: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_439);  mul_237 = unsqueeze_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_91: "f32[8, 144, 56, 56]" = torch.ops.aten.convolution.default(add_118, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_119: "f32[144]" = torch.ops.aten.add.Tensor(arg41_1, 0.001);  arg41_1 = None
        sqrt_55: "f32[144]" = torch.ops.aten.sqrt.default(add_119);  add_119 = None
        reciprocal_55: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_238: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_441: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_238, -1);  mul_238 = None
        unsqueeze_443: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_55: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_441);  convolution_91 = unsqueeze_441 = None
        mul_239: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
        unsqueeze_444: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_445: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_240: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_445);  mul_239 = unsqueeze_445 = None
        unsqueeze_446: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_447: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_120: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_240, unsqueeze_447);  mul_240 = unsqueeze_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_73: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_120)
        mul_241: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_120, sigmoid_73);  add_120 = sigmoid_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_92: "f32[8, 144, 56, 56]" = torch.ops.aten.convolution.default(mul_241, arg44_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144);  mul_241 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_121: "f32[144]" = torch.ops.aten.add.Tensor(arg46_1, 0.001);  arg46_1 = None
        sqrt_56: "f32[144]" = torch.ops.aten.sqrt.default(add_121);  add_121 = None
        reciprocal_56: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_242: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_449: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_242, -1);  mul_242 = None
        unsqueeze_451: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_56: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_449);  convolution_92 = unsqueeze_449 = None
        mul_243: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
        unsqueeze_452: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_453: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_244: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_243, unsqueeze_453);  mul_243 = unsqueeze_453 = None
        unsqueeze_454: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_455: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_122: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_455);  mul_244 = unsqueeze_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_74: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_122)
        mul_245: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_122, sigmoid_74);  add_122 = sigmoid_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_19: "f32[8, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_245, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_93: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_19, arg49_1, arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg49_1 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_75: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_93)
        mul_246: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_93, sigmoid_75);  convolution_93 = sigmoid_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_94: "f32[8, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_246, arg51_1, arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_246 = arg51_1 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_76: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_94);  convolution_94 = None
        mul_247: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_245, sigmoid_76);  mul_245 = sigmoid_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_95: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_247, arg53_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_247 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_123: "f32[24]" = torch.ops.aten.add.Tensor(arg55_1, 0.001);  arg55_1 = None
        sqrt_57: "f32[24]" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_57: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_248: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_457: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_248, -1);  mul_248 = None
        unsqueeze_459: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_57: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_457);  convolution_95 = unsqueeze_457 = None
        mul_249: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
        unsqueeze_460: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_461: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_250: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_249, unsqueeze_461);  mul_249 = unsqueeze_461 = None
        unsqueeze_462: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_463: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_124: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_250, unsqueeze_463);  mul_250 = unsqueeze_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_125: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_124, add_118);  add_124 = add_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_96: "f32[8, 144, 56, 56]" = torch.ops.aten.convolution.default(add_125, arg58_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_125 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_126: "f32[144]" = torch.ops.aten.add.Tensor(arg60_1, 0.001);  arg60_1 = None
        sqrt_58: "f32[144]" = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_58: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_251: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_465: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_251, -1);  mul_251 = None
        unsqueeze_467: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_58: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_465);  convolution_96 = unsqueeze_465 = None
        mul_252: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_469: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_253: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_469);  mul_252 = unsqueeze_469 = None
        unsqueeze_470: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_471: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_127: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_471);  mul_253 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_77: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_127)
        mul_254: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_127, sigmoid_77);  add_127 = sigmoid_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "f32[8, 144, 59, 59]" = torch.ops.aten.constant_pad_nd.default(mul_254, [1, 2, 1, 2], 0.0);  mul_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv2d_same.py:27 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_97: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_7, arg63_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 144);  constant_pad_nd_7 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_128: "f32[144]" = torch.ops.aten.add.Tensor(arg65_1, 0.001);  arg65_1 = None
        sqrt_59: "f32[144]" = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_59: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_255: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_473: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_475: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_59: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_473);  convolution_97 = unsqueeze_473 = None
        mul_256: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_477: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_257: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_477);  mul_256 = unsqueeze_477 = None
        unsqueeze_478: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_479: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_129: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_479);  mul_257 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_78: "f32[8, 144, 28, 28]" = torch.ops.aten.sigmoid.default(add_129)
        mul_258: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_129, sigmoid_78);  add_129 = sigmoid_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_20: "f32[8, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_258, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_98: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_20, arg68_1, arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg68_1 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_79: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_98)
        mul_259: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_98, sigmoid_79);  convolution_98 = sigmoid_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_99: "f32[8, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_259, arg70_1, arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_259 = arg70_1 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_80: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_99);  convolution_99 = None
        mul_260: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_258, sigmoid_80);  mul_258 = sigmoid_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_100: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_260, arg72_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_260 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_130: "f32[40]" = torch.ops.aten.add.Tensor(arg74_1, 0.001);  arg74_1 = None
        sqrt_60: "f32[40]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_60: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_261: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_481: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_483: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_60: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_481);  convolution_100 = unsqueeze_481 = None
        mul_262: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_485: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_263: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_485);  mul_262 = unsqueeze_485 = None
        unsqueeze_486: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_487: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_131: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_487);  mul_263 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_101: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(add_131, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_132: "f32[240]" = torch.ops.aten.add.Tensor(arg79_1, 0.001);  arg79_1 = None
        sqrt_61: "f32[240]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_61: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_264: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_489: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_491: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_61: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_489);  convolution_101 = unsqueeze_489 = None
        mul_265: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_493: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_266: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_493);  mul_265 = unsqueeze_493 = None
        unsqueeze_494: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_495: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_133: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_495);  mul_266 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_81: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_133)
        mul_267: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_133, sigmoid_81);  add_133 = sigmoid_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_102: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(mul_267, arg82_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240);  mul_267 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_134: "f32[240]" = torch.ops.aten.add.Tensor(arg84_1, 0.001);  arg84_1 = None
        sqrt_62: "f32[240]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_62: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_268: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_497: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_268, -1);  mul_268 = None
        unsqueeze_499: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_497);  convolution_102 = unsqueeze_497 = None
        mul_269: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_501: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_270: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_501);  mul_269 = unsqueeze_501 = None
        unsqueeze_502: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_503: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_135: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_503);  mul_270 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_82: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_135)
        mul_271: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_135, sigmoid_82);  add_135 = sigmoid_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_21: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_271, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_103: "f32[8, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_21, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg87_1 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_83: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_103)
        mul_272: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_103, sigmoid_83);  convolution_103 = sigmoid_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_104: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_272, arg89_1, arg90_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_272 = arg89_1 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_84: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_104);  convolution_104 = None
        mul_273: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_271, sigmoid_84);  mul_271 = sigmoid_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_105: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_273, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_273 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_136: "f32[40]" = torch.ops.aten.add.Tensor(arg93_1, 0.001);  arg93_1 = None
        sqrt_63: "f32[40]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_63: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_274: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_505: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_274, -1);  mul_274 = None
        unsqueeze_507: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_505);  convolution_105 = unsqueeze_505 = None
        mul_275: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_509: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_276: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_509);  mul_275 = unsqueeze_509 = None
        unsqueeze_510: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_511: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_137: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_511);  mul_276 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_138: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_137, add_131);  add_137 = add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_106: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(add_138, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_138 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_139: "f32[240]" = torch.ops.aten.add.Tensor(arg98_1, 0.001);  arg98_1 = None
        sqrt_64: "f32[240]" = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_64: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_277: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_513: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_515: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_513);  convolution_106 = unsqueeze_513 = None
        mul_278: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_517: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_279: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_517);  mul_278 = unsqueeze_517 = None
        unsqueeze_518: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_519: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_140: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_519);  mul_279 = unsqueeze_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_85: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_140)
        mul_280: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_140, sigmoid_85);  add_140 = sigmoid_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "f32[8, 240, 29, 29]" = torch.ops.aten.constant_pad_nd.default(mul_280, [0, 1, 0, 1], 0.0);  mul_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv2d_same.py:27 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_107: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_8, arg101_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240);  constant_pad_nd_8 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_141: "f32[240]" = torch.ops.aten.add.Tensor(arg103_1, 0.001);  arg103_1 = None
        sqrt_65: "f32[240]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_65: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_281: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_521: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_281, -1);  mul_281 = None
        unsqueeze_523: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_521);  convolution_107 = unsqueeze_521 = None
        mul_282: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_525: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_283: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_282, unsqueeze_525);  mul_282 = unsqueeze_525 = None
        unsqueeze_526: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_527: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_142: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_527);  mul_283 = unsqueeze_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_86: "f32[8, 240, 14, 14]" = torch.ops.aten.sigmoid.default(add_142)
        mul_284: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_142, sigmoid_86);  add_142 = sigmoid_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_22: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_284, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_108: "f32[8, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_22, arg106_1, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg106_1 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_87: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_108)
        mul_285: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_108, sigmoid_87);  convolution_108 = sigmoid_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_109: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_285, arg108_1, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_285 = arg108_1 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_88: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_109);  convolution_109 = None
        mul_286: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_284, sigmoid_88);  mul_284 = sigmoid_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_110: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_286, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_286 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_143: "f32[80]" = torch.ops.aten.add.Tensor(arg112_1, 0.001);  arg112_1 = None
        sqrt_66: "f32[80]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_66: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_287: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_529: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_287, -1);  mul_287 = None
        unsqueeze_531: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_529);  convolution_110 = unsqueeze_529 = None
        mul_288: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_533: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_289: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_533);  mul_288 = unsqueeze_533 = None
        unsqueeze_534: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_535: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_144: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_535);  mul_289 = unsqueeze_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_111: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_144, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_145: "f32[480]" = torch.ops.aten.add.Tensor(arg117_1, 0.001);  arg117_1 = None
        sqrt_67: "f32[480]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_67: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_290: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_537: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_290, -1);  mul_290 = None
        unsqueeze_539: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_537);  convolution_111 = unsqueeze_537 = None
        mul_291: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_541: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_292: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_291, unsqueeze_541);  mul_291 = unsqueeze_541 = None
        unsqueeze_542: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_543: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_146: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_543);  mul_292 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_89: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_146)
        mul_293: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_146, sigmoid_89);  add_146 = sigmoid_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_112: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_293, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  mul_293 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_147: "f32[480]" = torch.ops.aten.add.Tensor(arg122_1, 0.001);  arg122_1 = None
        sqrt_68: "f32[480]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_68: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_294: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_545: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
        unsqueeze_547: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_545);  convolution_112 = unsqueeze_545 = None
        mul_295: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_549: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_296: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_549);  mul_295 = unsqueeze_549 = None
        unsqueeze_550: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_551: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_148: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_551);  mul_296 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_90: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_148)
        mul_297: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_148, sigmoid_90);  add_148 = sigmoid_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_23: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_297, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_113: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_23, arg125_1, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg125_1 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_91: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_113)
        mul_298: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_113, sigmoid_91);  convolution_113 = sigmoid_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_114: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_298, arg127_1, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_298 = arg127_1 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_92: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_114);  convolution_114 = None
        mul_299: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_297, sigmoid_92);  mul_297 = sigmoid_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_115: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_299, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_299 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_149: "f32[80]" = torch.ops.aten.add.Tensor(arg131_1, 0.001);  arg131_1 = None
        sqrt_69: "f32[80]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_69: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_300: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_553: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_555: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_553);  convolution_115 = unsqueeze_553 = None
        mul_301: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_557: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_302: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_557);  mul_301 = unsqueeze_557 = None
        unsqueeze_558: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_559: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_150: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_559);  mul_302 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_151: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_150, add_144);  add_150 = add_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_116: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_151, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_152: "f32[480]" = torch.ops.aten.add.Tensor(arg136_1, 0.001);  arg136_1 = None
        sqrt_70: "f32[480]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_70: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_303: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_561: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_563: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_561);  convolution_116 = unsqueeze_561 = None
        mul_304: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_565: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_305: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_565);  mul_304 = unsqueeze_565 = None
        unsqueeze_566: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_567: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_153: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_567);  mul_305 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_93: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_153)
        mul_306: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_153, sigmoid_93);  add_153 = sigmoid_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_117: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_306, arg139_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  mul_306 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_154: "f32[480]" = torch.ops.aten.add.Tensor(arg141_1, 0.001);  arg141_1 = None
        sqrt_71: "f32[480]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_71: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_307: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_569: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_307, -1);  mul_307 = None
        unsqueeze_571: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_569);  convolution_117 = unsqueeze_569 = None
        mul_308: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_573: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_309: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_573);  mul_308 = unsqueeze_573 = None
        unsqueeze_574: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_575: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_155: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_309, unsqueeze_575);  mul_309 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_94: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_155)
        mul_310: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_155, sigmoid_94);  add_155 = sigmoid_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_24: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_310, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_118: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_24, arg144_1, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg144_1 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_95: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_118)
        mul_311: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_118, sigmoid_95);  convolution_118 = sigmoid_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_119: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_311, arg146_1, arg147_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_311 = arg146_1 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_96: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_119);  convolution_119 = None
        mul_312: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_310, sigmoid_96);  mul_310 = sigmoid_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_120: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_312, arg148_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_312 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_156: "f32[80]" = torch.ops.aten.add.Tensor(arg150_1, 0.001);  arg150_1 = None
        sqrt_72: "f32[80]" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_72: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_313: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_577: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_313, -1);  mul_313 = None
        unsqueeze_579: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_577);  convolution_120 = unsqueeze_577 = None
        mul_314: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_581: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_315: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_581);  mul_314 = unsqueeze_581 = None
        unsqueeze_582: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_583: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_157: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_315, unsqueeze_583);  mul_315 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_158: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_157, add_151);  add_157 = add_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_121: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_158, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_158 = arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_159: "f32[480]" = torch.ops.aten.add.Tensor(arg155_1, 0.001);  arg155_1 = None
        sqrt_73: "f32[480]" = torch.ops.aten.sqrt.default(add_159);  add_159 = None
        reciprocal_73: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_316: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_585: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_316, -1);  mul_316 = None
        unsqueeze_587: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_585);  convolution_121 = unsqueeze_585 = None
        mul_317: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_589: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_318: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_589);  mul_317 = unsqueeze_589 = None
        unsqueeze_590: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_591: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_160: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_318, unsqueeze_591);  mul_318 = unsqueeze_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_97: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_160)
        mul_319: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_160, sigmoid_97);  add_160 = sigmoid_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_122: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_319, arg158_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480);  mul_319 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_161: "f32[480]" = torch.ops.aten.add.Tensor(arg160_1, 0.001);  arg160_1 = None
        sqrt_74: "f32[480]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_74: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_320: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_593: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_595: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_593);  convolution_122 = unsqueeze_593 = None
        mul_321: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_597: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_322: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_597);  mul_321 = unsqueeze_597 = None
        unsqueeze_598: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_599: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_162: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_322, unsqueeze_599);  mul_322 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_98: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_162)
        mul_323: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_162, sigmoid_98);  add_162 = sigmoid_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_25: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_323, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_123: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_25, arg163_1, arg164_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg163_1 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_99: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_123)
        mul_324: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_123, sigmoid_99);  convolution_123 = sigmoid_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_124: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_324, arg165_1, arg166_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_324 = arg165_1 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_100: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_124);  convolution_124 = None
        mul_325: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_323, sigmoid_100);  mul_323 = sigmoid_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_125: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_325, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_325 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_163: "f32[112]" = torch.ops.aten.add.Tensor(arg169_1, 0.001);  arg169_1 = None
        sqrt_75: "f32[112]" = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_75: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_326: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_601: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_326, -1);  mul_326 = None
        unsqueeze_603: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_601);  convolution_125 = unsqueeze_601 = None
        mul_327: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_605: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_328: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_327, unsqueeze_605);  mul_327 = unsqueeze_605 = None
        unsqueeze_606: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_607: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_164: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_607);  mul_328 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_126: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_164, arg172_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg172_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_165: "f32[672]" = torch.ops.aten.add.Tensor(arg174_1, 0.001);  arg174_1 = None
        sqrt_76: "f32[672]" = torch.ops.aten.sqrt.default(add_165);  add_165 = None
        reciprocal_76: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_329: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_609: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_329, -1);  mul_329 = None
        unsqueeze_611: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_609);  convolution_126 = unsqueeze_609 = None
        mul_330: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_613: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_331: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_330, unsqueeze_613);  mul_330 = unsqueeze_613 = None
        unsqueeze_614: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_615: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_166: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_331, unsqueeze_615);  mul_331 = unsqueeze_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_101: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_166)
        mul_332: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_101);  add_166 = sigmoid_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_127: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(mul_332, arg177_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  mul_332 = arg177_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_167: "f32[672]" = torch.ops.aten.add.Tensor(arg179_1, 0.001);  arg179_1 = None
        sqrt_77: "f32[672]" = torch.ops.aten.sqrt.default(add_167);  add_167 = None
        reciprocal_77: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_333: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_617: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_619: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_617);  convolution_127 = unsqueeze_617 = None
        mul_334: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_621: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_335: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_621);  mul_334 = unsqueeze_621 = None
        unsqueeze_622: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_623: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_168: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_623);  mul_335 = unsqueeze_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_102: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_168)
        mul_336: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_168, sigmoid_102);  add_168 = sigmoid_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_26: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_336, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_128: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_26, arg182_1, arg183_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg182_1 = arg183_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_103: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_128)
        mul_337: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_128, sigmoid_103);  convolution_128 = sigmoid_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_129: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_337, arg184_1, arg185_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_337 = arg184_1 = arg185_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_104: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_129);  convolution_129 = None
        mul_338: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_336, sigmoid_104);  mul_336 = sigmoid_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_130: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_338, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_338 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_169: "f32[112]" = torch.ops.aten.add.Tensor(arg188_1, 0.001);  arg188_1 = None
        sqrt_78: "f32[112]" = torch.ops.aten.sqrt.default(add_169);  add_169 = None
        reciprocal_78: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_339: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_625: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_627: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_625);  convolution_130 = unsqueeze_625 = None
        mul_340: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_629: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_341: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_629);  mul_340 = unsqueeze_629 = None
        unsqueeze_630: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_631: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_170: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_631);  mul_341 = unsqueeze_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_171: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_170, add_164);  add_170 = add_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_131: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_171, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_172: "f32[672]" = torch.ops.aten.add.Tensor(arg193_1, 0.001);  arg193_1 = None
        sqrt_79: "f32[672]" = torch.ops.aten.sqrt.default(add_172);  add_172 = None
        reciprocal_79: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_342: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_633: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_635: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_633);  convolution_131 = unsqueeze_633 = None
        mul_343: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_637: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_344: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_637);  mul_343 = unsqueeze_637 = None
        unsqueeze_638: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_639: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_173: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_639);  mul_344 = unsqueeze_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_105: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_173)
        mul_345: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_173, sigmoid_105);  add_173 = sigmoid_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_132: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(mul_345, arg196_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  mul_345 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_174: "f32[672]" = torch.ops.aten.add.Tensor(arg198_1, 0.001);  arg198_1 = None
        sqrt_80: "f32[672]" = torch.ops.aten.sqrt.default(add_174);  add_174 = None
        reciprocal_80: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_346: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_641: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_346, -1);  mul_346 = None
        unsqueeze_643: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_641);  convolution_132 = unsqueeze_641 = None
        mul_347: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_645: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_348: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_347, unsqueeze_645);  mul_347 = unsqueeze_645 = None
        unsqueeze_646: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_647: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_175: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_348, unsqueeze_647);  mul_348 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_106: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_175)
        mul_349: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_175, sigmoid_106);  add_175 = sigmoid_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_27: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_349, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_133: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_27, arg201_1, arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_27 = arg201_1 = arg202_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_107: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_133)
        mul_350: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_133, sigmoid_107);  convolution_133 = sigmoid_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_134: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_350, arg203_1, arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_350 = arg203_1 = arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_108: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_134);  convolution_134 = None
        mul_351: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_349, sigmoid_108);  mul_349 = sigmoid_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_135: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_351, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_351 = arg205_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_176: "f32[112]" = torch.ops.aten.add.Tensor(arg207_1, 0.001);  arg207_1 = None
        sqrt_81: "f32[112]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
        reciprocal_81: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_352: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_649: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_352, -1);  mul_352 = None
        unsqueeze_651: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_649);  convolution_135 = unsqueeze_649 = None
        mul_353: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_653: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_354: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_353, unsqueeze_653);  mul_353 = unsqueeze_653 = None
        unsqueeze_654: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_655: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_177: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_354, unsqueeze_655);  mul_354 = unsqueeze_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_178: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_177, add_171);  add_177 = add_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_136: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_178, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_178 = arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_179: "f32[672]" = torch.ops.aten.add.Tensor(arg212_1, 0.001);  arg212_1 = None
        sqrt_82: "f32[672]" = torch.ops.aten.sqrt.default(add_179);  add_179 = None
        reciprocal_82: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_355: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_657: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_355, -1);  mul_355 = None
        unsqueeze_659: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_657);  convolution_136 = unsqueeze_657 = None
        mul_356: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_661: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_357: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_356, unsqueeze_661);  mul_356 = unsqueeze_661 = None
        unsqueeze_662: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_663: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_180: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_357, unsqueeze_663);  mul_357 = unsqueeze_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_109: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_180)
        mul_358: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, sigmoid_109);  add_180 = sigmoid_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_9: "f32[8, 672, 17, 17]" = torch.ops.aten.constant_pad_nd.default(mul_358, [1, 2, 1, 2], 0.0);  mul_358 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv2d_same.py:27 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_137: "f32[8, 672, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_9, arg215_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 672);  constant_pad_nd_9 = arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_181: "f32[672]" = torch.ops.aten.add.Tensor(arg217_1, 0.001);  arg217_1 = None
        sqrt_83: "f32[672]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_83: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_359: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_665: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_359, -1);  mul_359 = None
        unsqueeze_667: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_665);  convolution_137 = unsqueeze_665 = None
        mul_360: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_669: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_361: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_669);  mul_360 = unsqueeze_669 = None
        unsqueeze_670: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_671: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_182: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_361, unsqueeze_671);  mul_361 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_110: "f32[8, 672, 7, 7]" = torch.ops.aten.sigmoid.default(add_182)
        mul_362: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_182, sigmoid_110);  add_182 = sigmoid_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_28: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_362, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_138: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_28, arg220_1, arg221_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_28 = arg220_1 = arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_111: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_138)
        mul_363: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_138, sigmoid_111);  convolution_138 = sigmoid_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_139: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_363, arg222_1, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_363 = arg222_1 = arg223_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_112: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_139);  convolution_139 = None
        mul_364: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_362, sigmoid_112);  mul_362 = sigmoid_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_140: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_364, arg224_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_364 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_183: "f32[192]" = torch.ops.aten.add.Tensor(arg226_1, 0.001);  arg226_1 = None
        sqrt_84: "f32[192]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_84: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_365: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_673: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_365, -1);  mul_365 = None
        unsqueeze_675: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_673);  convolution_140 = unsqueeze_673 = None
        mul_366: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_677: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_367: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_366, unsqueeze_677);  mul_366 = unsqueeze_677 = None
        unsqueeze_678: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_679: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_184: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_367, unsqueeze_679);  mul_367 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_141: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_184, arg229_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg229_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_185: "f32[1152]" = torch.ops.aten.add.Tensor(arg231_1, 0.001);  arg231_1 = None
        sqrt_85: "f32[1152]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_85: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_368: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_681: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_368, -1);  mul_368 = None
        unsqueeze_683: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_681);  convolution_141 = unsqueeze_681 = None
        mul_369: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_685: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_370: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_369, unsqueeze_685);  mul_369 = unsqueeze_685 = None
        unsqueeze_686: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_687: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_186: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_687);  mul_370 = unsqueeze_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_113: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_186)
        mul_371: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_186, sigmoid_113);  add_186 = sigmoid_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_142: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_371, arg234_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_371 = arg234_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_187: "f32[1152]" = torch.ops.aten.add.Tensor(arg236_1, 0.001);  arg236_1 = None
        sqrt_86: "f32[1152]" = torch.ops.aten.sqrt.default(add_187);  add_187 = None
        reciprocal_86: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_372: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_689: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_691: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_689);  convolution_142 = unsqueeze_689 = None
        mul_373: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_693: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_374: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_693);  mul_373 = unsqueeze_693 = None
        unsqueeze_694: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_695: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_188: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_695);  mul_374 = unsqueeze_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_114: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_188)
        mul_375: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_188, sigmoid_114);  add_188 = sigmoid_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_29: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_375, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_143: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_29, arg239_1, arg240_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_29 = arg239_1 = arg240_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_115: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_143)
        mul_376: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_143, sigmoid_115);  convolution_143 = sigmoid_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_144: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_376, arg241_1, arg242_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_376 = arg241_1 = arg242_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_116: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_144);  convolution_144 = None
        mul_377: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_375, sigmoid_116);  mul_375 = sigmoid_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_145: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_377, arg243_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_377 = arg243_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_189: "f32[192]" = torch.ops.aten.add.Tensor(arg245_1, 0.001);  arg245_1 = None
        sqrt_87: "f32[192]" = torch.ops.aten.sqrt.default(add_189);  add_189 = None
        reciprocal_87: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_378: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_697: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_699: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_697);  convolution_145 = unsqueeze_697 = None
        mul_379: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_701: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_380: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_701);  mul_379 = unsqueeze_701 = None
        unsqueeze_702: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_703: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_190: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_703);  mul_380 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_191: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_190, add_184);  add_190 = add_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_146: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_191, arg248_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_192: "f32[1152]" = torch.ops.aten.add.Tensor(arg250_1, 0.001);  arg250_1 = None
        sqrt_88: "f32[1152]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_88: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_381: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_705: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_707: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_705);  convolution_146 = unsqueeze_705 = None
        mul_382: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_709: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_383: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_709);  mul_382 = unsqueeze_709 = None
        unsqueeze_710: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_711: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_193: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_383, unsqueeze_711);  mul_383 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_117: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_193)
        mul_384: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_193, sigmoid_117);  add_193 = sigmoid_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_147: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_384, arg253_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_384 = arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_194: "f32[1152]" = torch.ops.aten.add.Tensor(arg255_1, 0.001);  arg255_1 = None
        sqrt_89: "f32[1152]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_89: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_385: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_713: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_385, -1);  mul_385 = None
        unsqueeze_715: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_713);  convolution_147 = unsqueeze_713 = None
        mul_386: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_717: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_387: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_386, unsqueeze_717);  mul_386 = unsqueeze_717 = None
        unsqueeze_718: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_719: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_195: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_387, unsqueeze_719);  mul_387 = unsqueeze_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_118: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_195)
        mul_388: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_195, sigmoid_118);  add_195 = sigmoid_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_30: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_388, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_148: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_30, arg258_1, arg259_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_30 = arg258_1 = arg259_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_119: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_148)
        mul_389: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_148, sigmoid_119);  convolution_148 = sigmoid_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_149: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_389, arg260_1, arg261_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_389 = arg260_1 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_120: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_149);  convolution_149 = None
        mul_390: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_388, sigmoid_120);  mul_388 = sigmoid_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_150: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_390, arg262_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_390 = arg262_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_196: "f32[192]" = torch.ops.aten.add.Tensor(arg264_1, 0.001);  arg264_1 = None
        sqrt_90: "f32[192]" = torch.ops.aten.sqrt.default(add_196);  add_196 = None
        reciprocal_90: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_391: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_721: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_391, -1);  mul_391 = None
        unsqueeze_723: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_721);  convolution_150 = unsqueeze_721 = None
        mul_392: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_725: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_393: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_725);  mul_392 = unsqueeze_725 = None
        unsqueeze_726: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_727: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_197: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_393, unsqueeze_727);  mul_393 = unsqueeze_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_198: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_197, add_191);  add_197 = add_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_151: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_198, arg267_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg267_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_199: "f32[1152]" = torch.ops.aten.add.Tensor(arg269_1, 0.001);  arg269_1 = None
        sqrt_91: "f32[1152]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_91: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_394: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_729: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_394, -1);  mul_394 = None
        unsqueeze_731: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_729);  convolution_151 = unsqueeze_729 = None
        mul_395: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_733: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_396: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_395, unsqueeze_733);  mul_395 = unsqueeze_733 = None
        unsqueeze_734: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_735: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_200: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_396, unsqueeze_735);  mul_396 = unsqueeze_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_121: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_200)
        mul_397: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_200, sigmoid_121);  add_200 = sigmoid_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_152: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_397, arg272_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_397 = arg272_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_201: "f32[1152]" = torch.ops.aten.add.Tensor(arg274_1, 0.001);  arg274_1 = None
        sqrt_92: "f32[1152]" = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_92: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_398: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_737: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_398, -1);  mul_398 = None
        unsqueeze_739: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_737);  convolution_152 = unsqueeze_737 = None
        mul_399: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_741: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_400: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_741);  mul_399 = unsqueeze_741 = None
        unsqueeze_742: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_743: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_202: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_400, unsqueeze_743);  mul_400 = unsqueeze_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_122: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_202)
        mul_401: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_202, sigmoid_122);  add_202 = sigmoid_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_31: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_401, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_153: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_31, arg277_1, arg278_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_31 = arg277_1 = arg278_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_123: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_153)
        mul_402: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_153, sigmoid_123);  convolution_153 = sigmoid_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_154: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_402, arg279_1, arg280_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_402 = arg279_1 = arg280_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_124: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_154);  convolution_154 = None
        mul_403: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_401, sigmoid_124);  mul_401 = sigmoid_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_155: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_403, arg281_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_403 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_203: "f32[192]" = torch.ops.aten.add.Tensor(arg283_1, 0.001);  arg283_1 = None
        sqrt_93: "f32[192]" = torch.ops.aten.sqrt.default(add_203);  add_203 = None
        reciprocal_93: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_404: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_745: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_404, -1);  mul_404 = None
        unsqueeze_747: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_745);  convolution_155 = unsqueeze_745 = None
        mul_405: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_749: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_406: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_749);  mul_405 = unsqueeze_749 = None
        unsqueeze_750: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_751: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_204: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_751);  mul_406 = unsqueeze_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_205: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_204, add_198);  add_204 = add_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_156: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_205, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_205 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_206: "f32[1152]" = torch.ops.aten.add.Tensor(arg288_1, 0.001);  arg288_1 = None
        sqrt_94: "f32[1152]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_94: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_407: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_753: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_755: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_753);  convolution_156 = unsqueeze_753 = None
        mul_408: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_757: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_409: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_757);  mul_408 = unsqueeze_757 = None
        unsqueeze_758: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_759: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_207: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_759);  mul_409 = unsqueeze_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_125: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_207)
        mul_410: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_207, sigmoid_125);  add_207 = sigmoid_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_157: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_410, arg291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152);  mul_410 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_208: "f32[1152]" = torch.ops.aten.add.Tensor(arg293_1, 0.001);  arg293_1 = None
        sqrt_95: "f32[1152]" = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_95: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_411: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_761: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_763: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_761);  convolution_157 = unsqueeze_761 = None
        mul_412: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_765: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_413: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_765);  mul_412 = unsqueeze_765 = None
        unsqueeze_766: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_767: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_209: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_413, unsqueeze_767);  mul_413 = unsqueeze_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_126: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_209)
        mul_414: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_209, sigmoid_126);  add_209 = sigmoid_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_32: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_414, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_158: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_32, arg296_1, arg297_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_32 = arg296_1 = arg297_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_127: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_158)
        mul_415: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_158, sigmoid_127);  convolution_158 = sigmoid_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_159: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_415, arg298_1, arg299_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_415 = arg298_1 = arg299_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_128: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_159);  convolution_159 = None
        mul_416: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_414, sigmoid_128);  mul_414 = sigmoid_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_160: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(mul_416, arg300_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_416 = arg300_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_210: "f32[320]" = torch.ops.aten.add.Tensor(arg302_1, 0.001);  arg302_1 = None
        sqrt_96: "f32[320]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_96: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_417: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
        unsqueeze_769: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
        unsqueeze_771: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_769);  convolution_160 = unsqueeze_769 = None
        mul_418: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_773: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_419: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_773);  mul_418 = unsqueeze_773 = None
        unsqueeze_774: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_775: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_211: "f32[8, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_775);  mul_419 = unsqueeze_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/efficientnet.py:258 in forward_features, code: x = self.conv_head(x)
        convolution_161: "f32[8, 1280, 7, 7]" = torch.ops.aten.convolution.default(add_211, arg305_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_211 = arg305_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_212: "f32[1280]" = torch.ops.aten.add.Tensor(arg307_1, 0.001);  arg307_1 = None
        sqrt_97: "f32[1280]" = torch.ops.aten.sqrt.default(add_212);  add_212 = None
        reciprocal_97: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_420: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_777: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
        unsqueeze_779: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_777);  convolution_161 = unsqueeze_777 = None
        mul_421: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_781: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_422: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_781);  mul_421 = unsqueeze_781 = None
        unsqueeze_782: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_783: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_213: "f32[8, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_783);  mul_422 = unsqueeze_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_129: "f32[8, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(add_213)
        mul_423: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_213, sigmoid_129);  add_213 = sigmoid_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_33: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_423, [-1, -2], True);  mul_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 1280]" = torch.ops.aten.view.default(mean_33, [8, 1280]);  mean_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/efficientnet.py:266 in forward_head, code: return x if pre_logits else self.classifier(x)
        permute_1: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg311_1, view_1, permute_1);  arg311_1 = view_1 = permute_1 = None
        return (addmm_1,)
        