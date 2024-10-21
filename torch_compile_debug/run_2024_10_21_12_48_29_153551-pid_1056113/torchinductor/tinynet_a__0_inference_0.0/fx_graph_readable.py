class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[8, 3, 192, 192]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[32, 1, 3, 3]", arg7_1: "f32[32]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[8, 32, 1, 1]", arg12_1: "f32[8]", arg13_1: "f32[32, 8, 1, 1]", arg14_1: "f32[32]", arg15_1: "f32[16, 32, 1, 1]", arg16_1: "f32[16]", arg17_1: "f32[16]", arg18_1: "f32[16]", arg19_1: "f32[16]", arg20_1: "f32[96, 16, 1, 1]", arg21_1: "f32[96]", arg22_1: "f32[96]", arg23_1: "f32[96]", arg24_1: "f32[96]", arg25_1: "f32[96, 1, 3, 3]", arg26_1: "f32[96]", arg27_1: "f32[96]", arg28_1: "f32[96]", arg29_1: "f32[96]", arg30_1: "f32[4, 96, 1, 1]", arg31_1: "f32[4]", arg32_1: "f32[96, 4, 1, 1]", arg33_1: "f32[96]", arg34_1: "f32[24, 96, 1, 1]", arg35_1: "f32[24]", arg36_1: "f32[24]", arg37_1: "f32[24]", arg38_1: "f32[24]", arg39_1: "f32[144, 24, 1, 1]", arg40_1: "f32[144]", arg41_1: "f32[144]", arg42_1: "f32[144]", arg43_1: "f32[144]", arg44_1: "f32[144, 1, 3, 3]", arg45_1: "f32[144]", arg46_1: "f32[144]", arg47_1: "f32[144]", arg48_1: "f32[144]", arg49_1: "f32[6, 144, 1, 1]", arg50_1: "f32[6]", arg51_1: "f32[144, 6, 1, 1]", arg52_1: "f32[144]", arg53_1: "f32[24, 144, 1, 1]", arg54_1: "f32[24]", arg55_1: "f32[24]", arg56_1: "f32[24]", arg57_1: "f32[24]", arg58_1: "f32[144, 24, 1, 1]", arg59_1: "f32[144]", arg60_1: "f32[144]", arg61_1: "f32[144]", arg62_1: "f32[144]", arg63_1: "f32[144, 1, 5, 5]", arg64_1: "f32[144]", arg65_1: "f32[144]", arg66_1: "f32[144]", arg67_1: "f32[144]", arg68_1: "f32[6, 144, 1, 1]", arg69_1: "f32[6]", arg70_1: "f32[144, 6, 1, 1]", arg71_1: "f32[144]", arg72_1: "f32[40, 144, 1, 1]", arg73_1: "f32[40]", arg74_1: "f32[40]", arg75_1: "f32[40]", arg76_1: "f32[40]", arg77_1: "f32[240, 40, 1, 1]", arg78_1: "f32[240]", arg79_1: "f32[240]", arg80_1: "f32[240]", arg81_1: "f32[240]", arg82_1: "f32[240, 1, 5, 5]", arg83_1: "f32[240]", arg84_1: "f32[240]", arg85_1: "f32[240]", arg86_1: "f32[240]", arg87_1: "f32[10, 240, 1, 1]", arg88_1: "f32[10]", arg89_1: "f32[240, 10, 1, 1]", arg90_1: "f32[240]", arg91_1: "f32[40, 240, 1, 1]", arg92_1: "f32[40]", arg93_1: "f32[40]", arg94_1: "f32[40]", arg95_1: "f32[40]", arg96_1: "f32[240, 40, 1, 1]", arg97_1: "f32[240]", arg98_1: "f32[240]", arg99_1: "f32[240]", arg100_1: "f32[240]", arg101_1: "f32[240, 1, 3, 3]", arg102_1: "f32[240]", arg103_1: "f32[240]", arg104_1: "f32[240]", arg105_1: "f32[240]", arg106_1: "f32[10, 240, 1, 1]", arg107_1: "f32[10]", arg108_1: "f32[240, 10, 1, 1]", arg109_1: "f32[240]", arg110_1: "f32[80, 240, 1, 1]", arg111_1: "f32[80]", arg112_1: "f32[80]", arg113_1: "f32[80]", arg114_1: "f32[80]", arg115_1: "f32[480, 80, 1, 1]", arg116_1: "f32[480]", arg117_1: "f32[480]", arg118_1: "f32[480]", arg119_1: "f32[480]", arg120_1: "f32[480, 1, 3, 3]", arg121_1: "f32[480]", arg122_1: "f32[480]", arg123_1: "f32[480]", arg124_1: "f32[480]", arg125_1: "f32[20, 480, 1, 1]", arg126_1: "f32[20]", arg127_1: "f32[480, 20, 1, 1]", arg128_1: "f32[480]", arg129_1: "f32[80, 480, 1, 1]", arg130_1: "f32[80]", arg131_1: "f32[80]", arg132_1: "f32[80]", arg133_1: "f32[80]", arg134_1: "f32[480, 80, 1, 1]", arg135_1: "f32[480]", arg136_1: "f32[480]", arg137_1: "f32[480]", arg138_1: "f32[480]", arg139_1: "f32[480, 1, 3, 3]", arg140_1: "f32[480]", arg141_1: "f32[480]", arg142_1: "f32[480]", arg143_1: "f32[480]", arg144_1: "f32[20, 480, 1, 1]", arg145_1: "f32[20]", arg146_1: "f32[480, 20, 1, 1]", arg147_1: "f32[480]", arg148_1: "f32[80, 480, 1, 1]", arg149_1: "f32[80]", arg150_1: "f32[80]", arg151_1: "f32[80]", arg152_1: "f32[80]", arg153_1: "f32[480, 80, 1, 1]", arg154_1: "f32[480]", arg155_1: "f32[480]", arg156_1: "f32[480]", arg157_1: "f32[480]", arg158_1: "f32[480, 1, 3, 3]", arg159_1: "f32[480]", arg160_1: "f32[480]", arg161_1: "f32[480]", arg162_1: "f32[480]", arg163_1: "f32[20, 480, 1, 1]", arg164_1: "f32[20]", arg165_1: "f32[480, 20, 1, 1]", arg166_1: "f32[480]", arg167_1: "f32[80, 480, 1, 1]", arg168_1: "f32[80]", arg169_1: "f32[80]", arg170_1: "f32[80]", arg171_1: "f32[80]", arg172_1: "f32[480, 80, 1, 1]", arg173_1: "f32[480]", arg174_1: "f32[480]", arg175_1: "f32[480]", arg176_1: "f32[480]", arg177_1: "f32[480, 1, 5, 5]", arg178_1: "f32[480]", arg179_1: "f32[480]", arg180_1: "f32[480]", arg181_1: "f32[480]", arg182_1: "f32[20, 480, 1, 1]", arg183_1: "f32[20]", arg184_1: "f32[480, 20, 1, 1]", arg185_1: "f32[480]", arg186_1: "f32[112, 480, 1, 1]", arg187_1: "f32[112]", arg188_1: "f32[112]", arg189_1: "f32[112]", arg190_1: "f32[112]", arg191_1: "f32[672, 112, 1, 1]", arg192_1: "f32[672]", arg193_1: "f32[672]", arg194_1: "f32[672]", arg195_1: "f32[672]", arg196_1: "f32[672, 1, 5, 5]", arg197_1: "f32[672]", arg198_1: "f32[672]", arg199_1: "f32[672]", arg200_1: "f32[672]", arg201_1: "f32[28, 672, 1, 1]", arg202_1: "f32[28]", arg203_1: "f32[672, 28, 1, 1]", arg204_1: "f32[672]", arg205_1: "f32[112, 672, 1, 1]", arg206_1: "f32[112]", arg207_1: "f32[112]", arg208_1: "f32[112]", arg209_1: "f32[112]", arg210_1: "f32[672, 112, 1, 1]", arg211_1: "f32[672]", arg212_1: "f32[672]", arg213_1: "f32[672]", arg214_1: "f32[672]", arg215_1: "f32[672, 1, 5, 5]", arg216_1: "f32[672]", arg217_1: "f32[672]", arg218_1: "f32[672]", arg219_1: "f32[672]", arg220_1: "f32[28, 672, 1, 1]", arg221_1: "f32[28]", arg222_1: "f32[672, 28, 1, 1]", arg223_1: "f32[672]", arg224_1: "f32[112, 672, 1, 1]", arg225_1: "f32[112]", arg226_1: "f32[112]", arg227_1: "f32[112]", arg228_1: "f32[112]", arg229_1: "f32[672, 112, 1, 1]", arg230_1: "f32[672]", arg231_1: "f32[672]", arg232_1: "f32[672]", arg233_1: "f32[672]", arg234_1: "f32[672, 1, 5, 5]", arg235_1: "f32[672]", arg236_1: "f32[672]", arg237_1: "f32[672]", arg238_1: "f32[672]", arg239_1: "f32[28, 672, 1, 1]", arg240_1: "f32[28]", arg241_1: "f32[672, 28, 1, 1]", arg242_1: "f32[672]", arg243_1: "f32[112, 672, 1, 1]", arg244_1: "f32[112]", arg245_1: "f32[112]", arg246_1: "f32[112]", arg247_1: "f32[112]", arg248_1: "f32[672, 112, 1, 1]", arg249_1: "f32[672]", arg250_1: "f32[672]", arg251_1: "f32[672]", arg252_1: "f32[672]", arg253_1: "f32[672, 1, 5, 5]", arg254_1: "f32[672]", arg255_1: "f32[672]", arg256_1: "f32[672]", arg257_1: "f32[672]", arg258_1: "f32[28, 672, 1, 1]", arg259_1: "f32[28]", arg260_1: "f32[672, 28, 1, 1]", arg261_1: "f32[672]", arg262_1: "f32[192, 672, 1, 1]", arg263_1: "f32[192]", arg264_1: "f32[192]", arg265_1: "f32[192]", arg266_1: "f32[192]", arg267_1: "f32[1152, 192, 1, 1]", arg268_1: "f32[1152]", arg269_1: "f32[1152]", arg270_1: "f32[1152]", arg271_1: "f32[1152]", arg272_1: "f32[1152, 1, 5, 5]", arg273_1: "f32[1152]", arg274_1: "f32[1152]", arg275_1: "f32[1152]", arg276_1: "f32[1152]", arg277_1: "f32[48, 1152, 1, 1]", arg278_1: "f32[48]", arg279_1: "f32[1152, 48, 1, 1]", arg280_1: "f32[1152]", arg281_1: "f32[192, 1152, 1, 1]", arg282_1: "f32[192]", arg283_1: "f32[192]", arg284_1: "f32[192]", arg285_1: "f32[192]", arg286_1: "f32[1152, 192, 1, 1]", arg287_1: "f32[1152]", arg288_1: "f32[1152]", arg289_1: "f32[1152]", arg290_1: "f32[1152]", arg291_1: "f32[1152, 1, 5, 5]", arg292_1: "f32[1152]", arg293_1: "f32[1152]", arg294_1: "f32[1152]", arg295_1: "f32[1152]", arg296_1: "f32[48, 1152, 1, 1]", arg297_1: "f32[48]", arg298_1: "f32[1152, 48, 1, 1]", arg299_1: "f32[1152]", arg300_1: "f32[192, 1152, 1, 1]", arg301_1: "f32[192]", arg302_1: "f32[192]", arg303_1: "f32[192]", arg304_1: "f32[192]", arg305_1: "f32[1152, 192, 1, 1]", arg306_1: "f32[1152]", arg307_1: "f32[1152]", arg308_1: "f32[1152]", arg309_1: "f32[1152]", arg310_1: "f32[1152, 1, 5, 5]", arg311_1: "f32[1152]", arg312_1: "f32[1152]", arg313_1: "f32[1152]", arg314_1: "f32[1152]", arg315_1: "f32[48, 1152, 1, 1]", arg316_1: "f32[48]", arg317_1: "f32[1152, 48, 1, 1]", arg318_1: "f32[1152]", arg319_1: "f32[192, 1152, 1, 1]", arg320_1: "f32[192]", arg321_1: "f32[192]", arg322_1: "f32[192]", arg323_1: "f32[192]", arg324_1: "f32[1152, 192, 1, 1]", arg325_1: "f32[1152]", arg326_1: "f32[1152]", arg327_1: "f32[1152]", arg328_1: "f32[1152]", arg329_1: "f32[1152, 1, 5, 5]", arg330_1: "f32[1152]", arg331_1: "f32[1152]", arg332_1: "f32[1152]", arg333_1: "f32[1152]", arg334_1: "f32[48, 1152, 1, 1]", arg335_1: "f32[48]", arg336_1: "f32[1152, 48, 1, 1]", arg337_1: "f32[1152]", arg338_1: "f32[192, 1152, 1, 1]", arg339_1: "f32[192]", arg340_1: "f32[192]", arg341_1: "f32[192]", arg342_1: "f32[192]", arg343_1: "f32[1152, 192, 1, 1]", arg344_1: "f32[1152]", arg345_1: "f32[1152]", arg346_1: "f32[1152]", arg347_1: "f32[1152]", arg348_1: "f32[1152, 1, 3, 3]", arg349_1: "f32[1152]", arg350_1: "f32[1152]", arg351_1: "f32[1152]", arg352_1: "f32[1152]", arg353_1: "f32[48, 1152, 1, 1]", arg354_1: "f32[48]", arg355_1: "f32[1152, 48, 1, 1]", arg356_1: "f32[1152]", arg357_1: "f32[320, 1152, 1, 1]", arg358_1: "f32[320]", arg359_1: "f32[320]", arg360_1: "f32[320]", arg361_1: "f32[320]", arg362_1: "f32[1280, 320, 1, 1]", arg363_1: "f32[1280]", arg364_1: "f32[1280]", arg365_1: "f32[1280]", arg366_1: "f32[1280]", arg367_1: "f32[1000, 1280]", arg368_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/efficientnet.py:252 in forward_features, code: x = self.conv_stem(x)
        convolution_96: "f32[8, 32, 96, 96]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_128: "f32[32]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_58: "f32[32]" = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_58: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_251: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_465: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_251, -1);  mul_251 = None
        unsqueeze_467: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_58: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_465);  convolution_96 = unsqueeze_465 = None
        mul_252: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_469: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_253: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_469);  mul_252 = unsqueeze_469 = None
        unsqueeze_470: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_471: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_129: "f32[8, 32, 96, 96]" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_471);  mul_253 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_77: "f32[8, 32, 96, 96]" = torch.ops.aten.sigmoid.default(add_129)
        mul_254: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(add_129, sigmoid_77);  add_129 = sigmoid_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:190 in forward, code: x = self.conv_dw(x)
        convolution_97: "f32[8, 32, 96, 96]" = torch.ops.aten.convolution.default(mul_254, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  mul_254 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_130: "f32[32]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_59: "f32[32]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_59: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_255: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_473: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_475: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_59: "f32[8, 32, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_473);  convolution_97 = unsqueeze_473 = None
        mul_256: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_477: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_257: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_477);  mul_256 = unsqueeze_477 = None
        unsqueeze_478: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_479: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_131: "f32[8, 32, 96, 96]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_479);  mul_257 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_78: "f32[8, 32, 96, 96]" = torch.ops.aten.sigmoid.default(add_131)
        mul_258: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(add_131, sigmoid_78);  add_131 = sigmoid_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_20: "f32[8, 32, 1, 1]" = torch.ops.aten.mean.dim(mul_258, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_98: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_20, arg11_1, arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg11_1 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_79: "f32[8, 8, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_98)
        mul_259: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_98, sigmoid_79);  convolution_98 = sigmoid_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_99: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mul_259, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_259 = arg13_1 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_80: "f32[8, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_99);  convolution_99 = None
        mul_260: "f32[8, 32, 96, 96]" = torch.ops.aten.mul.Tensor(mul_258, sigmoid_80);  mul_258 = sigmoid_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:194 in forward, code: x = self.conv_pw(x)
        convolution_100: "f32[8, 16, 96, 96]" = torch.ops.aten.convolution.default(mul_260, arg15_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_260 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_132: "f32[16]" = torch.ops.aten.add.Tensor(arg17_1, 1e-05);  arg17_1 = None
        sqrt_60: "f32[16]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_60: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_261: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_481: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_483: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_60: "f32[8, 16, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_481);  convolution_100 = unsqueeze_481 = None
        mul_262: "f32[8, 16, 96, 96]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_485: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_263: "f32[8, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_485);  mul_262 = unsqueeze_485 = None
        unsqueeze_486: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_487: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_133: "f32[8, 16, 96, 96]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_487);  mul_263 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_101: "f32[8, 96, 96, 96]" = torch.ops.aten.convolution.default(add_133, arg20_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_133 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_134: "f32[96]" = torch.ops.aten.add.Tensor(arg22_1, 1e-05);  arg22_1 = None
        sqrt_61: "f32[96]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_61: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_264: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_489: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_491: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_61: "f32[8, 96, 96, 96]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_489);  convolution_101 = unsqueeze_489 = None
        mul_265: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_493: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_266: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_493);  mul_265 = unsqueeze_493 = None
        unsqueeze_494: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_495: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_135: "f32[8, 96, 96, 96]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_495);  mul_266 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_81: "f32[8, 96, 96, 96]" = torch.ops.aten.sigmoid.default(add_135)
        mul_267: "f32[8, 96, 96, 96]" = torch.ops.aten.mul.Tensor(add_135, sigmoid_81);  add_135 = sigmoid_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_102: "f32[8, 96, 48, 48]" = torch.ops.aten.convolution.default(mul_267, arg25_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96);  mul_267 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_136: "f32[96]" = torch.ops.aten.add.Tensor(arg27_1, 1e-05);  arg27_1 = None
        sqrt_62: "f32[96]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_62: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_268: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_497: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_268, -1);  mul_268 = None
        unsqueeze_499: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62: "f32[8, 96, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_497);  convolution_102 = unsqueeze_497 = None
        mul_269: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_501: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_270: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_501);  mul_269 = unsqueeze_501 = None
        unsqueeze_502: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_503: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_137: "f32[8, 96, 48, 48]" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_503);  mul_270 = unsqueeze_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_82: "f32[8, 96, 48, 48]" = torch.ops.aten.sigmoid.default(add_137)
        mul_271: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_82);  add_137 = sigmoid_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_21: "f32[8, 96, 1, 1]" = torch.ops.aten.mean.dim(mul_271, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_103: "f32[8, 4, 1, 1]" = torch.ops.aten.convolution.default(mean_21, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg30_1 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_83: "f32[8, 4, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_103)
        mul_272: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_103, sigmoid_83);  convolution_103 = sigmoid_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_104: "f32[8, 96, 1, 1]" = torch.ops.aten.convolution.default(mul_272, arg32_1, arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_272 = arg32_1 = arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_84: "f32[8, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_104);  convolution_104 = None
        mul_273: "f32[8, 96, 48, 48]" = torch.ops.aten.mul.Tensor(mul_271, sigmoid_84);  mul_271 = sigmoid_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_105: "f32[8, 24, 48, 48]" = torch.ops.aten.convolution.default(mul_273, arg34_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_273 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_138: "f32[24]" = torch.ops.aten.add.Tensor(arg36_1, 1e-05);  arg36_1 = None
        sqrt_63: "f32[24]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_63: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_274: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_505: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_274, -1);  mul_274 = None
        unsqueeze_507: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_505);  convolution_105 = unsqueeze_505 = None
        mul_275: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_509: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_276: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_509);  mul_275 = unsqueeze_509 = None
        unsqueeze_510: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_511: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_139: "f32[8, 24, 48, 48]" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_511);  mul_276 = unsqueeze_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_106: "f32[8, 144, 48, 48]" = torch.ops.aten.convolution.default(add_139, arg39_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_140: "f32[144]" = torch.ops.aten.add.Tensor(arg41_1, 1e-05);  arg41_1 = None
        sqrt_64: "f32[144]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_64: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_277: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_513: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_515: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_513);  convolution_106 = unsqueeze_513 = None
        mul_278: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_517: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_279: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_517);  mul_278 = unsqueeze_517 = None
        unsqueeze_518: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_519: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_141: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_519);  mul_279 = unsqueeze_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_85: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(add_141)
        mul_280: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(add_141, sigmoid_85);  add_141 = sigmoid_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_107: "f32[8, 144, 48, 48]" = torch.ops.aten.convolution.default(mul_280, arg44_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144);  mul_280 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_142: "f32[144]" = torch.ops.aten.add.Tensor(arg46_1, 1e-05);  arg46_1 = None
        sqrt_65: "f32[144]" = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_65: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_281: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_521: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_281, -1);  mul_281 = None
        unsqueeze_523: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_521);  convolution_107 = unsqueeze_521 = None
        mul_282: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_525: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_283: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_282, unsqueeze_525);  mul_282 = unsqueeze_525 = None
        unsqueeze_526: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_527: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_143: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_527);  mul_283 = unsqueeze_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_86: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(add_143)
        mul_284: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(add_143, sigmoid_86);  add_143 = sigmoid_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_22: "f32[8, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_284, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_108: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_22, arg49_1, arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg49_1 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_87: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_108)
        mul_285: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_108, sigmoid_87);  convolution_108 = sigmoid_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_109: "f32[8, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_285, arg51_1, arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_285 = arg51_1 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_88: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_109);  convolution_109 = None
        mul_286: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_284, sigmoid_88);  mul_284 = sigmoid_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_110: "f32[8, 24, 48, 48]" = torch.ops.aten.convolution.default(mul_286, arg53_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_286 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_144: "f32[24]" = torch.ops.aten.add.Tensor(arg55_1, 1e-05);  arg55_1 = None
        sqrt_66: "f32[24]" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_66: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_287: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_529: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_287, -1);  mul_287 = None
        unsqueeze_531: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66: "f32[8, 24, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_529);  convolution_110 = unsqueeze_529 = None
        mul_288: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_533: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_289: "f32[8, 24, 48, 48]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_533);  mul_288 = unsqueeze_533 = None
        unsqueeze_534: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_535: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_145: "f32[8, 24, 48, 48]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_535);  mul_289 = unsqueeze_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_146: "f32[8, 24, 48, 48]" = torch.ops.aten.add.Tensor(add_145, add_139);  add_145 = add_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_111: "f32[8, 144, 48, 48]" = torch.ops.aten.convolution.default(add_146, arg58_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_146 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_147: "f32[144]" = torch.ops.aten.add.Tensor(arg60_1, 1e-05);  arg60_1 = None
        sqrt_67: "f32[144]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_67: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_290: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_537: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_290, -1);  mul_290 = None
        unsqueeze_539: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67: "f32[8, 144, 48, 48]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_537);  convolution_111 = unsqueeze_537 = None
        mul_291: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_541: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_292: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(mul_291, unsqueeze_541);  mul_291 = unsqueeze_541 = None
        unsqueeze_542: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_543: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_148: "f32[8, 144, 48, 48]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_543);  mul_292 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_89: "f32[8, 144, 48, 48]" = torch.ops.aten.sigmoid.default(add_148)
        mul_293: "f32[8, 144, 48, 48]" = torch.ops.aten.mul.Tensor(add_148, sigmoid_89);  add_148 = sigmoid_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_112: "f32[8, 144, 24, 24]" = torch.ops.aten.convolution.default(mul_293, arg63_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 144);  mul_293 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_149: "f32[144]" = torch.ops.aten.add.Tensor(arg65_1, 1e-05);  arg65_1 = None
        sqrt_68: "f32[144]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_68: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_294: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_545: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
        unsqueeze_547: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68: "f32[8, 144, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_545);  convolution_112 = unsqueeze_545 = None
        mul_295: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_549: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_296: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_549);  mul_295 = unsqueeze_549 = None
        unsqueeze_550: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_551: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_150: "f32[8, 144, 24, 24]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_551);  mul_296 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_90: "f32[8, 144, 24, 24]" = torch.ops.aten.sigmoid.default(add_150)
        mul_297: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(add_150, sigmoid_90);  add_150 = sigmoid_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_23: "f32[8, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_297, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_113: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_23, arg68_1, arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg68_1 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_91: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_113)
        mul_298: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_113, sigmoid_91);  convolution_113 = sigmoid_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_114: "f32[8, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_298, arg70_1, arg71_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_298 = arg70_1 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_92: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_114);  convolution_114 = None
        mul_299: "f32[8, 144, 24, 24]" = torch.ops.aten.mul.Tensor(mul_297, sigmoid_92);  mul_297 = sigmoid_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_115: "f32[8, 40, 24, 24]" = torch.ops.aten.convolution.default(mul_299, arg72_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_299 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_151: "f32[40]" = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_69: "f32[40]" = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_69: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_300: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_553: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_555: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_553);  convolution_115 = unsqueeze_553 = None
        mul_301: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_557: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_302: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_557);  mul_301 = unsqueeze_557 = None
        unsqueeze_558: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_559: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_152: "f32[8, 40, 24, 24]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_559);  mul_302 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_116: "f32[8, 240, 24, 24]" = torch.ops.aten.convolution.default(add_152, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_153: "f32[240]" = torch.ops.aten.add.Tensor(arg79_1, 1e-05);  arg79_1 = None
        sqrt_70: "f32[240]" = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_70: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_303: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_561: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_563: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_561);  convolution_116 = unsqueeze_561 = None
        mul_304: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_565: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_305: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_565);  mul_304 = unsqueeze_565 = None
        unsqueeze_566: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_567: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_154: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_567);  mul_305 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_93: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(add_154)
        mul_306: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(add_154, sigmoid_93);  add_154 = sigmoid_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_117: "f32[8, 240, 24, 24]" = torch.ops.aten.convolution.default(mul_306, arg82_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240);  mul_306 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_155: "f32[240]" = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_71: "f32[240]" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
        reciprocal_71: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_307: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_569: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_307, -1);  mul_307 = None
        unsqueeze_571: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_569);  convolution_117 = unsqueeze_569 = None
        mul_308: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_573: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_309: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_573);  mul_308 = unsqueeze_573 = None
        unsqueeze_574: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_575: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_156: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Tensor(mul_309, unsqueeze_575);  mul_309 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_94: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(add_156)
        mul_310: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(add_156, sigmoid_94);  add_156 = sigmoid_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_24: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_310, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_118: "f32[8, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_24, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg87_1 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_95: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_118)
        mul_311: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_118, sigmoid_95);  convolution_118 = sigmoid_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_119: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_311, arg89_1, arg90_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_311 = arg89_1 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_96: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_119);  convolution_119 = None
        mul_312: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_310, sigmoid_96);  mul_310 = sigmoid_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_120: "f32[8, 40, 24, 24]" = torch.ops.aten.convolution.default(mul_312, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_312 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_157: "f32[40]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_72: "f32[40]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_72: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_313: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_577: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_313, -1);  mul_313 = None
        unsqueeze_579: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72: "f32[8, 40, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_577);  convolution_120 = unsqueeze_577 = None
        mul_314: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_581: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_315: "f32[8, 40, 24, 24]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_581);  mul_314 = unsqueeze_581 = None
        unsqueeze_582: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_583: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_158: "f32[8, 40, 24, 24]" = torch.ops.aten.add.Tensor(mul_315, unsqueeze_583);  mul_315 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_159: "f32[8, 40, 24, 24]" = torch.ops.aten.add.Tensor(add_158, add_152);  add_158 = add_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_121: "f32[8, 240, 24, 24]" = torch.ops.aten.convolution.default(add_159, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_159 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_160: "f32[240]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_73: "f32[240]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_73: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_316: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_585: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_316, -1);  mul_316 = None
        unsqueeze_587: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73: "f32[8, 240, 24, 24]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_585);  convolution_121 = unsqueeze_585 = None
        mul_317: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_589: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_318: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_589);  mul_317 = unsqueeze_589 = None
        unsqueeze_590: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_591: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_161: "f32[8, 240, 24, 24]" = torch.ops.aten.add.Tensor(mul_318, unsqueeze_591);  mul_318 = unsqueeze_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_97: "f32[8, 240, 24, 24]" = torch.ops.aten.sigmoid.default(add_161)
        mul_319: "f32[8, 240, 24, 24]" = torch.ops.aten.mul.Tensor(add_161, sigmoid_97);  add_161 = sigmoid_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_122: "f32[8, 240, 12, 12]" = torch.ops.aten.convolution.default(mul_319, arg101_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240);  mul_319 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_162: "f32[240]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_74: "f32[240]" = torch.ops.aten.sqrt.default(add_162);  add_162 = None
        reciprocal_74: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_320: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_593: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_595: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74: "f32[8, 240, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_593);  convolution_122 = unsqueeze_593 = None
        mul_321: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_597: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_322: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_597);  mul_321 = unsqueeze_597 = None
        unsqueeze_598: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_599: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_163: "f32[8, 240, 12, 12]" = torch.ops.aten.add.Tensor(mul_322, unsqueeze_599);  mul_322 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_98: "f32[8, 240, 12, 12]" = torch.ops.aten.sigmoid.default(add_163)
        mul_323: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(add_163, sigmoid_98);  add_163 = sigmoid_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_25: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_323, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_123: "f32[8, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_25, arg106_1, arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg106_1 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_99: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_123)
        mul_324: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_123, sigmoid_99);  convolution_123 = sigmoid_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_124: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_324, arg108_1, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_324 = arg108_1 = arg109_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_100: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_124);  convolution_124 = None
        mul_325: "f32[8, 240, 12, 12]" = torch.ops.aten.mul.Tensor(mul_323, sigmoid_100);  mul_323 = sigmoid_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_125: "f32[8, 80, 12, 12]" = torch.ops.aten.convolution.default(mul_325, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_325 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_164: "f32[80]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_75: "f32[80]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_75: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_326: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_601: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_326, -1);  mul_326 = None
        unsqueeze_603: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_601);  convolution_125 = unsqueeze_601 = None
        mul_327: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_605: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_328: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(mul_327, unsqueeze_605);  mul_327 = unsqueeze_605 = None
        unsqueeze_606: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_607: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_165: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_607);  mul_328 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_126: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(add_165, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_166: "f32[480]" = torch.ops.aten.add.Tensor(arg117_1, 1e-05);  arg117_1 = None
        sqrt_76: "f32[480]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_76: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_329: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_609: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_329, -1);  mul_329 = None
        unsqueeze_611: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_609);  convolution_126 = unsqueeze_609 = None
        mul_330: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_613: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_331: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_330, unsqueeze_613);  mul_330 = unsqueeze_613 = None
        unsqueeze_614: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_615: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_167: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_331, unsqueeze_615);  mul_331 = unsqueeze_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_101: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_167)
        mul_332: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_167, sigmoid_101);  add_167 = sigmoid_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_127: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(mul_332, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  mul_332 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_168: "f32[480]" = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_77: "f32[480]" = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_77: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_333: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_617: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_619: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_617);  convolution_127 = unsqueeze_617 = None
        mul_334: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_621: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_335: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_621);  mul_334 = unsqueeze_621 = None
        unsqueeze_622: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_623: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_169: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_623);  mul_335 = unsqueeze_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_102: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_169)
        mul_336: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_169, sigmoid_102);  add_169 = sigmoid_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_26: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_336, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_128: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_26, arg125_1, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg125_1 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_103: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_128)
        mul_337: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_128, sigmoid_103);  convolution_128 = sigmoid_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_129: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_337, arg127_1, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_337 = arg127_1 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_104: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_129);  convolution_129 = None
        mul_338: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_336, sigmoid_104);  mul_336 = sigmoid_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_130: "f32[8, 80, 12, 12]" = torch.ops.aten.convolution.default(mul_338, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_338 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_170: "f32[80]" = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_78: "f32[80]" = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_78: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_339: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_625: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_627: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_625);  convolution_130 = unsqueeze_625 = None
        mul_340: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_629: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_341: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_629);  mul_340 = unsqueeze_629 = None
        unsqueeze_630: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_631: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_171: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_631);  mul_341 = unsqueeze_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_172: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(add_171, add_165);  add_171 = add_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_131: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(add_172, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_173: "f32[480]" = torch.ops.aten.add.Tensor(arg136_1, 1e-05);  arg136_1 = None
        sqrt_79: "f32[480]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_79: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_342: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_633: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_635: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_633);  convolution_131 = unsqueeze_633 = None
        mul_343: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_637: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_344: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_637);  mul_343 = unsqueeze_637 = None
        unsqueeze_638: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_639: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_174: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_639);  mul_344 = unsqueeze_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_105: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_174)
        mul_345: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_174, sigmoid_105);  add_174 = sigmoid_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_132: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(mul_345, arg139_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  mul_345 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_175: "f32[480]" = torch.ops.aten.add.Tensor(arg141_1, 1e-05);  arg141_1 = None
        sqrt_80: "f32[480]" = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_80: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_346: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_641: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_346, -1);  mul_346 = None
        unsqueeze_643: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_641);  convolution_132 = unsqueeze_641 = None
        mul_347: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_645: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_348: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_347, unsqueeze_645);  mul_347 = unsqueeze_645 = None
        unsqueeze_646: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_647: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_176: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_348, unsqueeze_647);  mul_348 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_106: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_176)
        mul_349: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_176, sigmoid_106);  add_176 = sigmoid_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_27: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_349, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_133: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_27, arg144_1, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_27 = arg144_1 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_107: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_133)
        mul_350: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_133, sigmoid_107);  convolution_133 = sigmoid_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_134: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_350, arg146_1, arg147_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_350 = arg146_1 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_108: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_134);  convolution_134 = None
        mul_351: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_349, sigmoid_108);  mul_349 = sigmoid_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_135: "f32[8, 80, 12, 12]" = torch.ops.aten.convolution.default(mul_351, arg148_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_351 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_177: "f32[80]" = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_81: "f32[80]" = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_81: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_352: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_649: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_352, -1);  mul_352 = None
        unsqueeze_651: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_649);  convolution_135 = unsqueeze_649 = None
        mul_353: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_653: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_354: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(mul_353, unsqueeze_653);  mul_353 = unsqueeze_653 = None
        unsqueeze_654: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_655: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_178: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(mul_354, unsqueeze_655);  mul_354 = unsqueeze_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_179: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(add_178, add_172);  add_178 = add_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_136: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(add_179, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_180: "f32[480]" = torch.ops.aten.add.Tensor(arg155_1, 1e-05);  arg155_1 = None
        sqrt_82: "f32[480]" = torch.ops.aten.sqrt.default(add_180);  add_180 = None
        reciprocal_82: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_355: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_657: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_355, -1);  mul_355 = None
        unsqueeze_659: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_657);  convolution_136 = unsqueeze_657 = None
        mul_356: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_661: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_357: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_356, unsqueeze_661);  mul_356 = unsqueeze_661 = None
        unsqueeze_662: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_663: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_181: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_357, unsqueeze_663);  mul_357 = unsqueeze_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_109: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_181)
        mul_358: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_181, sigmoid_109);  add_181 = sigmoid_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_137: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(mul_358, arg158_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480);  mul_358 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_182: "f32[480]" = torch.ops.aten.add.Tensor(arg160_1, 1e-05);  arg160_1 = None
        sqrt_83: "f32[480]" = torch.ops.aten.sqrt.default(add_182);  add_182 = None
        reciprocal_83: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_359: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_665: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_359, -1);  mul_359 = None
        unsqueeze_667: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_665);  convolution_137 = unsqueeze_665 = None
        mul_360: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_669: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_361: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_669);  mul_360 = unsqueeze_669 = None
        unsqueeze_670: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_671: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_183: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_361, unsqueeze_671);  mul_361 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_110: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_183)
        mul_362: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_183, sigmoid_110);  add_183 = sigmoid_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_28: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_362, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_138: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_28, arg163_1, arg164_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_28 = arg163_1 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_111: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_138)
        mul_363: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_138, sigmoid_111);  convolution_138 = sigmoid_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_139: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_363, arg165_1, arg166_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_363 = arg165_1 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_112: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_139);  convolution_139 = None
        mul_364: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_362, sigmoid_112);  mul_362 = sigmoid_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_140: "f32[8, 80, 12, 12]" = torch.ops.aten.convolution.default(mul_364, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_364 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_184: "f32[80]" = torch.ops.aten.add.Tensor(arg169_1, 1e-05);  arg169_1 = None
        sqrt_84: "f32[80]" = torch.ops.aten.sqrt.default(add_184);  add_184 = None
        reciprocal_84: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_365: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_673: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_365, -1);  mul_365 = None
        unsqueeze_675: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84: "f32[8, 80, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_673);  convolution_140 = unsqueeze_673 = None
        mul_366: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_677: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_367: "f32[8, 80, 12, 12]" = torch.ops.aten.mul.Tensor(mul_366, unsqueeze_677);  mul_366 = unsqueeze_677 = None
        unsqueeze_678: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_679: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_185: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(mul_367, unsqueeze_679);  mul_367 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_186: "f32[8, 80, 12, 12]" = torch.ops.aten.add.Tensor(add_185, add_179);  add_185 = add_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_141: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(add_186, arg172_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_186 = arg172_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_187: "f32[480]" = torch.ops.aten.add.Tensor(arg174_1, 1e-05);  arg174_1 = None
        sqrt_85: "f32[480]" = torch.ops.aten.sqrt.default(add_187);  add_187 = None
        reciprocal_85: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_368: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_681: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_368, -1);  mul_368 = None
        unsqueeze_683: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_681);  convolution_141 = unsqueeze_681 = None
        mul_369: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_685: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_370: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_369, unsqueeze_685);  mul_369 = unsqueeze_685 = None
        unsqueeze_686: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_687: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_188: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_687);  mul_370 = unsqueeze_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_113: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_188)
        mul_371: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_188, sigmoid_113);  add_188 = sigmoid_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_142: "f32[8, 480, 12, 12]" = torch.ops.aten.convolution.default(mul_371, arg177_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480);  mul_371 = arg177_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_189: "f32[480]" = torch.ops.aten.add.Tensor(arg179_1, 1e-05);  arg179_1 = None
        sqrt_86: "f32[480]" = torch.ops.aten.sqrt.default(add_189);  add_189 = None
        reciprocal_86: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_372: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_689: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_691: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86: "f32[8, 480, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_689);  convolution_142 = unsqueeze_689 = None
        mul_373: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_693: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_374: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_693);  mul_373 = unsqueeze_693 = None
        unsqueeze_694: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_695: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_190: "f32[8, 480, 12, 12]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_695);  mul_374 = unsqueeze_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_114: "f32[8, 480, 12, 12]" = torch.ops.aten.sigmoid.default(add_190)
        mul_375: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(add_190, sigmoid_114);  add_190 = sigmoid_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_29: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_375, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_143: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_29, arg182_1, arg183_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_29 = arg182_1 = arg183_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_115: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_143)
        mul_376: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_143, sigmoid_115);  convolution_143 = sigmoid_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_144: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_376, arg184_1, arg185_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_376 = arg184_1 = arg185_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_116: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_144);  convolution_144 = None
        mul_377: "f32[8, 480, 12, 12]" = torch.ops.aten.mul.Tensor(mul_375, sigmoid_116);  mul_375 = sigmoid_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_145: "f32[8, 112, 12, 12]" = torch.ops.aten.convolution.default(mul_377, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_377 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_191: "f32[112]" = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_87: "f32[112]" = torch.ops.aten.sqrt.default(add_191);  add_191 = None
        reciprocal_87: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_378: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_697: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_699: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_697);  convolution_145 = unsqueeze_697 = None
        mul_379: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_701: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_380: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_701);  mul_379 = unsqueeze_701 = None
        unsqueeze_702: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_703: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_192: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_703);  mul_380 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_146: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(add_192, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_193: "f32[672]" = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_88: "f32[672]" = torch.ops.aten.sqrt.default(add_193);  add_193 = None
        reciprocal_88: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_381: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_705: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_707: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_705);  convolution_146 = unsqueeze_705 = None
        mul_382: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_709: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_383: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_709);  mul_382 = unsqueeze_709 = None
        unsqueeze_710: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_711: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_194: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_383, unsqueeze_711);  mul_383 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_117: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_194)
        mul_384: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_194, sigmoid_117);  add_194 = sigmoid_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_147: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(mul_384, arg196_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  mul_384 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_195: "f32[672]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_89: "f32[672]" = torch.ops.aten.sqrt.default(add_195);  add_195 = None
        reciprocal_89: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_385: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_713: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_385, -1);  mul_385 = None
        unsqueeze_715: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_713);  convolution_147 = unsqueeze_713 = None
        mul_386: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_717: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_387: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_386, unsqueeze_717);  mul_386 = unsqueeze_717 = None
        unsqueeze_718: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_719: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_196: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_387, unsqueeze_719);  mul_387 = unsqueeze_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_118: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_196)
        mul_388: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_196, sigmoid_118);  add_196 = sigmoid_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_30: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_388, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_148: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_30, arg201_1, arg202_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_30 = arg201_1 = arg202_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_119: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_148)
        mul_389: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_148, sigmoid_119);  convolution_148 = sigmoid_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_149: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_389, arg203_1, arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_389 = arg203_1 = arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_120: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_149);  convolution_149 = None
        mul_390: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_388, sigmoid_120);  mul_388 = sigmoid_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_150: "f32[8, 112, 12, 12]" = torch.ops.aten.convolution.default(mul_390, arg205_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_390 = arg205_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_197: "f32[112]" = torch.ops.aten.add.Tensor(arg207_1, 1e-05);  arg207_1 = None
        sqrt_90: "f32[112]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_90: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_391: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_721: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_391, -1);  mul_391 = None
        unsqueeze_723: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_721);  convolution_150 = unsqueeze_721 = None
        mul_392: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_725: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_393: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_725);  mul_392 = unsqueeze_725 = None
        unsqueeze_726: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_727: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_198: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(mul_393, unsqueeze_727);  mul_393 = unsqueeze_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_199: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(add_198, add_192);  add_198 = add_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_151: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(add_199, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_200: "f32[672]" = torch.ops.aten.add.Tensor(arg212_1, 1e-05);  arg212_1 = None
        sqrt_91: "f32[672]" = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_91: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_394: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_729: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_394, -1);  mul_394 = None
        unsqueeze_731: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_729);  convolution_151 = unsqueeze_729 = None
        mul_395: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_733: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_396: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_395, unsqueeze_733);  mul_395 = unsqueeze_733 = None
        unsqueeze_734: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_735: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_201: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_396, unsqueeze_735);  mul_396 = unsqueeze_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_121: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_201)
        mul_397: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_201, sigmoid_121);  add_201 = sigmoid_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_152: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(mul_397, arg215_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  mul_397 = arg215_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_202: "f32[672]" = torch.ops.aten.add.Tensor(arg217_1, 1e-05);  arg217_1 = None
        sqrt_92: "f32[672]" = torch.ops.aten.sqrt.default(add_202);  add_202 = None
        reciprocal_92: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_398: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_737: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_398, -1);  mul_398 = None
        unsqueeze_739: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_737);  convolution_152 = unsqueeze_737 = None
        mul_399: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_741: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_400: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_741);  mul_399 = unsqueeze_741 = None
        unsqueeze_742: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_743: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_203: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_400, unsqueeze_743);  mul_400 = unsqueeze_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_122: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_203)
        mul_401: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_203, sigmoid_122);  add_203 = sigmoid_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_31: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_401, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_153: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_31, arg220_1, arg221_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_31 = arg220_1 = arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_123: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_153)
        mul_402: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_153, sigmoid_123);  convolution_153 = sigmoid_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_154: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_402, arg222_1, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_402 = arg222_1 = arg223_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_124: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_154);  convolution_154 = None
        mul_403: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_401, sigmoid_124);  mul_401 = sigmoid_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_155: "f32[8, 112, 12, 12]" = torch.ops.aten.convolution.default(mul_403, arg224_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_403 = arg224_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_204: "f32[112]" = torch.ops.aten.add.Tensor(arg226_1, 1e-05);  arg226_1 = None
        sqrt_93: "f32[112]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_93: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_404: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_745: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_404, -1);  mul_404 = None
        unsqueeze_747: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_155, unsqueeze_745);  convolution_155 = unsqueeze_745 = None
        mul_405: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_749: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_406: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_749);  mul_405 = unsqueeze_749 = None
        unsqueeze_750: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_751: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_205: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_751);  mul_406 = unsqueeze_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_206: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(add_205, add_199);  add_205 = add_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_156: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(add_206, arg229_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg229_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_207: "f32[672]" = torch.ops.aten.add.Tensor(arg231_1, 1e-05);  arg231_1 = None
        sqrt_94: "f32[672]" = torch.ops.aten.sqrt.default(add_207);  add_207 = None
        reciprocal_94: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_407: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_753: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_755: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_753);  convolution_156 = unsqueeze_753 = None
        mul_408: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_757: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_409: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_757);  mul_408 = unsqueeze_757 = None
        unsqueeze_758: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_759: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_208: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_759);  mul_409 = unsqueeze_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_125: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_208)
        mul_410: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_208, sigmoid_125);  add_208 = sigmoid_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_157: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(mul_410, arg234_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672);  mul_410 = arg234_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_209: "f32[672]" = torch.ops.aten.add.Tensor(arg236_1, 1e-05);  arg236_1 = None
        sqrt_95: "f32[672]" = torch.ops.aten.sqrt.default(add_209);  add_209 = None
        reciprocal_95: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_411: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_761: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_763: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_157, unsqueeze_761);  convolution_157 = unsqueeze_761 = None
        mul_412: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_765: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_413: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_765);  mul_412 = unsqueeze_765 = None
        unsqueeze_766: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_767: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_210: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_413, unsqueeze_767);  mul_413 = unsqueeze_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_126: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_210)
        mul_414: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_210, sigmoid_126);  add_210 = sigmoid_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_32: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_414, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_158: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_32, arg239_1, arg240_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_32 = arg239_1 = arg240_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_127: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_158)
        mul_415: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_158, sigmoid_127);  convolution_158 = sigmoid_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_159: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_415, arg241_1, arg242_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_415 = arg241_1 = arg242_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_128: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_159);  convolution_159 = None
        mul_416: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_414, sigmoid_128);  mul_414 = sigmoid_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_160: "f32[8, 112, 12, 12]" = torch.ops.aten.convolution.default(mul_416, arg243_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_416 = arg243_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_211: "f32[112]" = torch.ops.aten.add.Tensor(arg245_1, 1e-05);  arg245_1 = None
        sqrt_96: "f32[112]" = torch.ops.aten.sqrt.default(add_211);  add_211 = None
        reciprocal_96: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_417: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_769: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
        unsqueeze_771: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96: "f32[8, 112, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_769);  convolution_160 = unsqueeze_769 = None
        mul_418: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_773: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_419: "f32[8, 112, 12, 12]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_773);  mul_418 = unsqueeze_773 = None
        unsqueeze_774: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_775: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_212: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_775);  mul_419 = unsqueeze_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_213: "f32[8, 112, 12, 12]" = torch.ops.aten.add.Tensor(add_212, add_206);  add_212 = add_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_161: "f32[8, 672, 12, 12]" = torch.ops.aten.convolution.default(add_213, arg248_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_213 = arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_214: "f32[672]" = torch.ops.aten.add.Tensor(arg250_1, 1e-05);  arg250_1 = None
        sqrt_97: "f32[672]" = torch.ops.aten.sqrt.default(add_214);  add_214 = None
        reciprocal_97: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_420: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_777: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
        unsqueeze_779: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97: "f32[8, 672, 12, 12]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_777);  convolution_161 = unsqueeze_777 = None
        mul_421: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_781: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_422: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_781);  mul_421 = unsqueeze_781 = None
        unsqueeze_782: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_783: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_215: "f32[8, 672, 12, 12]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_783);  mul_422 = unsqueeze_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_129: "f32[8, 672, 12, 12]" = torch.ops.aten.sigmoid.default(add_215)
        mul_423: "f32[8, 672, 12, 12]" = torch.ops.aten.mul.Tensor(add_215, sigmoid_129);  add_215 = sigmoid_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_162: "f32[8, 672, 6, 6]" = torch.ops.aten.convolution.default(mul_423, arg253_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672);  mul_423 = arg253_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_216: "f32[672]" = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_98: "f32[672]" = torch.ops.aten.sqrt.default(add_216);  add_216 = None
        reciprocal_98: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_424: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_785: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_424, -1);  mul_424 = None
        unsqueeze_787: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98: "f32[8, 672, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_785);  convolution_162 = unsqueeze_785 = None
        mul_425: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_789: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_426: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(mul_425, unsqueeze_789);  mul_425 = unsqueeze_789 = None
        unsqueeze_790: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_791: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_217: "f32[8, 672, 6, 6]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_791);  mul_426 = unsqueeze_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_130: "f32[8, 672, 6, 6]" = torch.ops.aten.sigmoid.default(add_217)
        mul_427: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(add_217, sigmoid_130);  add_217 = sigmoid_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_33: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_427, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_163: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_33, arg258_1, arg259_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_33 = arg258_1 = arg259_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_131: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_163)
        mul_428: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_163, sigmoid_131);  convolution_163 = sigmoid_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_164: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_428, arg260_1, arg261_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_428 = arg260_1 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_132: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_164);  convolution_164 = None
        mul_429: "f32[8, 672, 6, 6]" = torch.ops.aten.mul.Tensor(mul_427, sigmoid_132);  mul_427 = sigmoid_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_165: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_429, arg262_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_429 = arg262_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_218: "f32[192]" = torch.ops.aten.add.Tensor(arg264_1, 1e-05);  arg264_1 = None
        sqrt_99: "f32[192]" = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_99: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_430: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_793: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_430, -1);  mul_430 = None
        unsqueeze_795: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_165, unsqueeze_793);  convolution_165 = unsqueeze_793 = None
        mul_431: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_797: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_432: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_431, unsqueeze_797);  mul_431 = unsqueeze_797 = None
        unsqueeze_798: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_799: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_219: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_432, unsqueeze_799);  mul_432 = unsqueeze_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_166: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_219, arg267_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg267_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_220: "f32[1152]" = torch.ops.aten.add.Tensor(arg269_1, 1e-05);  arg269_1 = None
        sqrt_100: "f32[1152]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_100: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_433: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_801: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_433, -1);  mul_433 = None
        unsqueeze_803: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_801);  convolution_166 = unsqueeze_801 = None
        mul_434: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_805: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_435: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_805);  mul_434 = unsqueeze_805 = None
        unsqueeze_806: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_807: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_221: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_435, unsqueeze_807);  mul_435 = unsqueeze_807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_133: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_221)
        mul_436: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_221, sigmoid_133);  add_221 = sigmoid_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_167: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_436, arg272_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_436 = arg272_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_222: "f32[1152]" = torch.ops.aten.add.Tensor(arg274_1, 1e-05);  arg274_1 = None
        sqrt_101: "f32[1152]" = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_101: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_437: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
        unsqueeze_809: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_437, -1);  mul_437 = None
        unsqueeze_811: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_167, unsqueeze_809);  convolution_167 = unsqueeze_809 = None
        mul_438: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_813: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_439: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_438, unsqueeze_813);  mul_438 = unsqueeze_813 = None
        unsqueeze_814: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_815: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_223: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_439, unsqueeze_815);  mul_439 = unsqueeze_815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_134: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_223)
        mul_440: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_223, sigmoid_134);  add_223 = sigmoid_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_34: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_440, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_168: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_34, arg277_1, arg278_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_34 = arg277_1 = arg278_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_135: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_168)
        mul_441: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_168, sigmoid_135);  convolution_168 = sigmoid_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_169: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_441, arg279_1, arg280_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_441 = arg279_1 = arg280_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_136: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_169);  convolution_169 = None
        mul_442: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_440, sigmoid_136);  mul_440 = sigmoid_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_170: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_442, arg281_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_442 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_224: "f32[192]" = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_102: "f32[192]" = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_102: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_443: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_817: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_443, -1);  mul_443 = None
        unsqueeze_819: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_817);  convolution_170 = unsqueeze_817 = None
        mul_444: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_821: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_445: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_444, unsqueeze_821);  mul_444 = unsqueeze_821 = None
        unsqueeze_822: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_823: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_225: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_445, unsqueeze_823);  mul_445 = unsqueeze_823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_226: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_225, add_219);  add_225 = add_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_171: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_226, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_227: "f32[1152]" = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_103: "f32[1152]" = torch.ops.aten.sqrt.default(add_227);  add_227 = None
        reciprocal_103: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_446: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_825: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_446, -1);  mul_446 = None
        unsqueeze_827: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_171, unsqueeze_825);  convolution_171 = unsqueeze_825 = None
        mul_447: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_829: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_448: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_447, unsqueeze_829);  mul_447 = unsqueeze_829 = None
        unsqueeze_830: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_831: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_228: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_448, unsqueeze_831);  mul_448 = unsqueeze_831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_137: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_228)
        mul_449: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_228, sigmoid_137);  add_228 = sigmoid_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_172: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_449, arg291_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_449 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_229: "f32[1152]" = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_104: "f32[1152]" = torch.ops.aten.sqrt.default(add_229);  add_229 = None
        reciprocal_104: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_450: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_833: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_835: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_833);  convolution_172 = unsqueeze_833 = None
        mul_451: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_837: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_452: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_837);  mul_451 = unsqueeze_837 = None
        unsqueeze_838: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_839: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_230: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_839);  mul_452 = unsqueeze_839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_138: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_230)
        mul_453: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_230, sigmoid_138);  add_230 = sigmoid_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_35: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_453, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_173: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_35, arg296_1, arg297_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_35 = arg296_1 = arg297_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_139: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_173)
        mul_454: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_173, sigmoid_139);  convolution_173 = sigmoid_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_174: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_454, arg298_1, arg299_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_454 = arg298_1 = arg299_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_140: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_174);  convolution_174 = None
        mul_455: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_453, sigmoid_140);  mul_453 = sigmoid_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_175: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_455, arg300_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_455 = arg300_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_231: "f32[192]" = torch.ops.aten.add.Tensor(arg302_1, 1e-05);  arg302_1 = None
        sqrt_105: "f32[192]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_105: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_456: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg301_1, -1);  arg301_1 = None
        unsqueeze_841: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_843: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_175, unsqueeze_841);  convolution_175 = unsqueeze_841 = None
        mul_457: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_845: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_458: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_845);  mul_457 = unsqueeze_845 = None
        unsqueeze_846: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_847: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_232: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_458, unsqueeze_847);  mul_458 = unsqueeze_847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_233: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_232, add_226);  add_232 = add_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_176: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_233, arg305_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg305_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_234: "f32[1152]" = torch.ops.aten.add.Tensor(arg307_1, 1e-05);  arg307_1 = None
        sqrt_106: "f32[1152]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_106: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_459: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_849: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_851: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_849);  convolution_176 = unsqueeze_849 = None
        mul_460: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_853: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_461: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_853);  mul_460 = unsqueeze_853 = None
        unsqueeze_854: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_855: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_235: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_855);  mul_461 = unsqueeze_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_141: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_235)
        mul_462: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_235, sigmoid_141);  add_235 = sigmoid_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_177: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_462, arg310_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_462 = arg310_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_236: "f32[1152]" = torch.ops.aten.add.Tensor(arg312_1, 1e-05);  arg312_1 = None
        sqrt_107: "f32[1152]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_107: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_463: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_857: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_463, -1);  mul_463 = None
        unsqueeze_859: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_177, unsqueeze_857);  convolution_177 = unsqueeze_857 = None
        mul_464: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_861: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_465: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_464, unsqueeze_861);  mul_464 = unsqueeze_861 = None
        unsqueeze_862: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_863: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_237: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_465, unsqueeze_863);  mul_465 = unsqueeze_863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_142: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_237)
        mul_466: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_237, sigmoid_142);  add_237 = sigmoid_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_36: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_466, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_178: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_36, arg315_1, arg316_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_36 = arg315_1 = arg316_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_143: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_178)
        mul_467: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_178, sigmoid_143);  convolution_178 = sigmoid_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_179: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_467, arg317_1, arg318_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_467 = arg317_1 = arg318_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_144: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_179);  convolution_179 = None
        mul_468: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_466, sigmoid_144);  mul_466 = sigmoid_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_180: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_468, arg319_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_468 = arg319_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_238: "f32[192]" = torch.ops.aten.add.Tensor(arg321_1, 1e-05);  arg321_1 = None
        sqrt_108: "f32[192]" = torch.ops.aten.sqrt.default(add_238);  add_238 = None
        reciprocal_108: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_469: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_865: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_469, -1);  mul_469 = None
        unsqueeze_867: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_865);  convolution_180 = unsqueeze_865 = None
        mul_470: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_869: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_471: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_869);  mul_470 = unsqueeze_869 = None
        unsqueeze_870: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_871: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_239: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_471, unsqueeze_871);  mul_471 = unsqueeze_871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_240: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_239, add_233);  add_239 = add_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_181: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_240, arg324_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg324_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_241: "f32[1152]" = torch.ops.aten.add.Tensor(arg326_1, 1e-05);  arg326_1 = None
        sqrt_109: "f32[1152]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_109: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_472: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_873: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_472, -1);  mul_472 = None
        unsqueeze_875: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_181, unsqueeze_873);  convolution_181 = unsqueeze_873 = None
        mul_473: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_877: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_474: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_877);  mul_473 = unsqueeze_877 = None
        unsqueeze_878: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_879: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_242: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_474, unsqueeze_879);  mul_474 = unsqueeze_879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_145: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_242)
        mul_475: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_242, sigmoid_145);  add_242 = sigmoid_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_182: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_475, arg329_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  mul_475 = arg329_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_243: "f32[1152]" = torch.ops.aten.add.Tensor(arg331_1, 1e-05);  arg331_1 = None
        sqrt_110: "f32[1152]" = torch.ops.aten.sqrt.default(add_243);  add_243 = None
        reciprocal_110: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_476: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_881: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_476, -1);  mul_476 = None
        unsqueeze_883: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_881);  convolution_182 = unsqueeze_881 = None
        mul_477: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_885: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_478: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_477, unsqueeze_885);  mul_477 = unsqueeze_885 = None
        unsqueeze_886: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg333_1, -1);  arg333_1 = None
        unsqueeze_887: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_244: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_478, unsqueeze_887);  mul_478 = unsqueeze_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_146: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_244)
        mul_479: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_244, sigmoid_146);  add_244 = sigmoid_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_37: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_479, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_183: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_37, arg334_1, arg335_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_37 = arg334_1 = arg335_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_147: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_183)
        mul_480: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_183, sigmoid_147);  convolution_183 = sigmoid_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_184: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_480, arg336_1, arg337_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_480 = arg336_1 = arg337_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_148: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_184);  convolution_184 = None
        mul_481: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_479, sigmoid_148);  mul_479 = sigmoid_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_185: "f32[8, 192, 6, 6]" = torch.ops.aten.convolution.default(mul_481, arg338_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_481 = arg338_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_245: "f32[192]" = torch.ops.aten.add.Tensor(arg340_1, 1e-05);  arg340_1 = None
        sqrt_111: "f32[192]" = torch.ops.aten.sqrt.default(add_245);  add_245 = None
        reciprocal_111: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_482: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_889: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_482, -1);  mul_482 = None
        unsqueeze_891: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111: "f32[8, 192, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_185, unsqueeze_889);  convolution_185 = unsqueeze_889 = None
        mul_483: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
        unsqueeze_893: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_484: "f32[8, 192, 6, 6]" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_893);  mul_483 = unsqueeze_893 = None
        unsqueeze_894: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_895: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_246: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(mul_484, unsqueeze_895);  mul_484 = unsqueeze_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:295 in forward, code: x = self.drop_path(x) + shortcut
        add_247: "f32[8, 192, 6, 6]" = torch.ops.aten.add.Tensor(add_246, add_240);  add_246 = add_240 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:286 in forward, code: x = self.conv_pw(x)
        convolution_186: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(add_247, arg343_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_247 = arg343_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_248: "f32[1152]" = torch.ops.aten.add.Tensor(arg345_1, 1e-05);  arg345_1 = None
        sqrt_112: "f32[1152]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_112: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_485: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_897: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_485, -1);  mul_485 = None
        unsqueeze_899: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_897);  convolution_186 = unsqueeze_897 = None
        mul_486: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
        unsqueeze_901: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_487: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_901);  mul_486 = unsqueeze_901 = None
        unsqueeze_902: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, -1);  arg347_1 = None
        unsqueeze_903: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_249: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_487, unsqueeze_903);  mul_487 = unsqueeze_903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_149: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_249)
        mul_488: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_249, sigmoid_149);  add_249 = sigmoid_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:288 in forward, code: x = self.conv_dw(x)
        convolution_187: "f32[8, 1152, 6, 6]" = torch.ops.aten.convolution.default(mul_488, arg348_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152);  mul_488 = arg348_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_250: "f32[1152]" = torch.ops.aten.add.Tensor(arg350_1, 1e-05);  arg350_1 = None
        sqrt_113: "f32[1152]" = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_113: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_489: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_905: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_489, -1);  mul_489 = None
        unsqueeze_907: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113: "f32[8, 1152, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_905);  convolution_187 = unsqueeze_905 = None
        mul_490: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg351_1, -1);  arg351_1 = None
        unsqueeze_909: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_491: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_909);  mul_490 = unsqueeze_909 = None
        unsqueeze_910: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, -1);  arg352_1 = None
        unsqueeze_911: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_251: "f32[8, 1152, 6, 6]" = torch.ops.aten.add.Tensor(mul_491, unsqueeze_911);  mul_491 = unsqueeze_911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_150: "f32[8, 1152, 6, 6]" = torch.ops.aten.sigmoid.default(add_251)
        mul_492: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(add_251, sigmoid_150);  add_251 = sigmoid_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:64 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_38: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_492, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:65 in forward, code: x_se = self.conv_reduce(x_se)
        convolution_188: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_38, arg353_1, arg354_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_38 = arg353_1 = arg354_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:66 in forward, code: x_se = self.act1(x_se)
        sigmoid_151: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_188)
        mul_493: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_188, sigmoid_151);  convolution_188 = sigmoid_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:67 in forward, code: x_se = self.conv_expand(x_se)
        convolution_189: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_493, arg355_1, arg356_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_493 = arg355_1 = arg356_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:68 in forward, code: return x * self.gate(x_se)
        sigmoid_152: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_189);  convolution_189 = None
        mul_494: "f32[8, 1152, 6, 6]" = torch.ops.aten.mul.Tensor(mul_492, sigmoid_152);  mul_492 = sigmoid_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:292 in forward, code: x = self.conv_pwl(x)
        convolution_190: "f32[8, 320, 6, 6]" = torch.ops.aten.convolution.default(mul_494, arg357_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_494 = arg357_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_252: "f32[320]" = torch.ops.aten.add.Tensor(arg359_1, 1e-05);  arg359_1 = None
        sqrt_114: "f32[320]" = torch.ops.aten.sqrt.default(add_252);  add_252 = None
        reciprocal_114: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_495: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg358_1, -1);  arg358_1 = None
        unsqueeze_913: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_495, -1);  mul_495 = None
        unsqueeze_915: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114: "f32[8, 320, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_913);  convolution_190 = unsqueeze_913 = None
        mul_496: "f32[8, 320, 6, 6]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_917: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_497: "f32[8, 320, 6, 6]" = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_917);  mul_496 = unsqueeze_917 = None
        unsqueeze_918: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg361_1, -1);  arg361_1 = None
        unsqueeze_919: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_253: "f32[8, 320, 6, 6]" = torch.ops.aten.add.Tensor(mul_497, unsqueeze_919);  mul_497 = unsqueeze_919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/efficientnet.py:258 in forward_features, code: x = self.conv_head(x)
        convolution_191: "f32[8, 1280, 6, 6]" = torch.ops.aten.convolution.default(add_253, arg362_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_253 = arg362_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_254: "f32[1280]" = torch.ops.aten.add.Tensor(arg364_1, 1e-05);  arg364_1 = None
        sqrt_115: "f32[1280]" = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_115: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_498: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg363_1, -1);  arg363_1 = None
        unsqueeze_921: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_498, -1);  mul_498 = None
        unsqueeze_923: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115: "f32[8, 1280, 6, 6]" = torch.ops.aten.sub.Tensor(convolution_191, unsqueeze_921);  convolution_191 = unsqueeze_921 = None
        mul_499: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_925: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_500: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_925);  mul_499 = unsqueeze_925 = None
        unsqueeze_926: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg366_1, -1);  arg366_1 = None
        unsqueeze_927: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_255: "f32[8, 1280, 6, 6]" = torch.ops.aten.add.Tensor(mul_500, unsqueeze_927);  mul_500 = unsqueeze_927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_153: "f32[8, 1280, 6, 6]" = torch.ops.aten.sigmoid.default(add_255)
        mul_501: "f32[8, 1280, 6, 6]" = torch.ops.aten.mul.Tensor(add_255, sigmoid_153);  add_255 = sigmoid_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_39: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_501, [-1, -2], True);  mul_501 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 1280]" = torch.ops.aten.view.default(mean_39, [8, 1280]);  mean_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/efficientnet.py:266 in forward_head, code: return x if pre_logits else self.classifier(x)
        permute_1: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg368_1, view_1, permute_1);  arg368_1 = view_1 = permute_1 = None
        return (addmm_1,)
        