class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[24, 3, 3, 3]", arg1_1: "f32[8, 3, 256, 256]", arg2_1: "f32[24]", arg3_1: "f32[24]", arg4_1: "f32[24]", arg5_1: "f32[24]", arg6_1: "f32[32, 24, 3, 3]", arg7_1: "f32[32]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[64, 32, 3, 3]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[64, 64, 1, 1]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[64, 16, 3, 3]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[1, 1, 3]", arg27_1: "f32[256, 64, 1, 1]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[256]", arg32_1: "f32[256, 64, 1, 1]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[64, 256, 1, 1]", arg38_1: "f32[64]", arg39_1: "f32[64]", arg40_1: "f32[64]", arg41_1: "f32[64]", arg42_1: "f32[64, 16, 3, 3]", arg43_1: "f32[64]", arg44_1: "f32[64]", arg45_1: "f32[64]", arg46_1: "f32[64]", arg47_1: "f32[1, 1, 3]", arg48_1: "f32[256, 64, 1, 1]", arg49_1: "f32[256]", arg50_1: "f32[256]", arg51_1: "f32[256]", arg52_1: "f32[256]", arg53_1: "f32[128, 256, 1, 1]", arg54_1: "f32[128]", arg55_1: "f32[128]", arg56_1: "f32[128]", arg57_1: "f32[128]", arg58_1: "f32[128, 16, 3, 3]", arg59_1: "f32[128]", arg60_1: "f32[128]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[1, 1, 5]", arg64_1: "f32[512, 128, 1, 1]", arg65_1: "f32[512]", arg66_1: "f32[512]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[512, 256, 1, 1]", arg70_1: "f32[512]", arg71_1: "f32[512]", arg72_1: "f32[512]", arg73_1: "f32[512]", arg74_1: "f32[128, 512, 1, 1]", arg75_1: "f32[128]", arg76_1: "f32[128]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[128, 16, 3, 3]", arg80_1: "f32[128]", arg81_1: "f32[128]", arg82_1: "f32[128]", arg83_1: "f32[128]", arg84_1: "f32[1, 1, 5]", arg85_1: "f32[512, 128, 1, 1]", arg86_1: "f32[512]", arg87_1: "f32[512]", arg88_1: "f32[512]", arg89_1: "f32[512]", arg90_1: "f32[256, 512, 1, 1]", arg91_1: "f32[256]", arg92_1: "f32[256]", arg93_1: "f32[256]", arg94_1: "f32[256]", arg95_1: "f32[256, 16, 3, 3]", arg96_1: "f32[256]", arg97_1: "f32[256]", arg98_1: "f32[256]", arg99_1: "f32[256]", arg100_1: "f32[1, 1, 5]", arg101_1: "f32[1024, 256, 1, 1]", arg102_1: "f32[1024]", arg103_1: "f32[1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024]", arg106_1: "f32[1024, 512, 1, 1]", arg107_1: "f32[1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024]", arg110_1: "f32[1024]", arg111_1: "f32[256, 1024, 1, 1]", arg112_1: "f32[256]", arg113_1: "f32[256]", arg114_1: "f32[256]", arg115_1: "f32[256]", arg116_1: "f32[128, 256, 1, 1]", arg117_1: "f32[384, 256, 1, 1]", arg118_1: "f32[23, 16]", arg119_1: "f32[23, 16]", arg120_1: "f32[256]", arg121_1: "f32[256]", arg122_1: "f32[256]", arg123_1: "f32[256]", arg124_1: "f32[1024, 256, 1, 1]", arg125_1: "f32[1024]", arg126_1: "f32[1024]", arg127_1: "f32[1024]", arg128_1: "f32[1024]", arg129_1: "f32[512, 1024, 1, 1]", arg130_1: "f32[512]", arg131_1: "f32[512]", arg132_1: "f32[512]", arg133_1: "f32[512]", arg134_1: "f32[128, 512, 1, 1]", arg135_1: "f32[640, 512, 1, 1]", arg136_1: "f32[23, 16]", arg137_1: "f32[23, 16]", arg138_1: "f32[512]", arg139_1: "f32[512]", arg140_1: "f32[512]", arg141_1: "f32[512]", arg142_1: "f32[2048, 512, 1, 1]", arg143_1: "f32[2048]", arg144_1: "f32[2048]", arg145_1: "f32[2048]", arg146_1: "f32[2048]", arg147_1: "f32[2048, 1024, 1, 1]", arg148_1: "f32[2048]", arg149_1: "f32[2048]", arg150_1: "f32[2048]", arg151_1: "f32[2048]", arg152_1: "f32[512, 2048, 1, 1]", arg153_1: "f32[512]", arg154_1: "f32[512]", arg155_1: "f32[512]", arg156_1: "f32[512]", arg157_1: "f32[128, 512, 1, 1]", arg158_1: "f32[640, 512, 1, 1]", arg159_1: "f32[23, 16]", arg160_1: "f32[23, 16]", arg161_1: "f32[512]", arg162_1: "f32[512]", arg163_1: "f32[512]", arg164_1: "f32[512]", arg165_1: "f32[2048, 512, 1, 1]", arg166_1: "f32[2048]", arg167_1: "f32[2048]", arg168_1: "f32[2048]", arg169_1: "f32[2048]", arg170_1: "f32[1000, 2048]", arg171_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_39: "f32[8, 24, 128, 128]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_248: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_249: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_34: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_249);  convolution_39 = unsqueeze_249 = None
        add_76: "f32[24]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_31: "f32[24]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
        reciprocal_31: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_128: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_128, -1);  mul_128 = None
        unsqueeze_251: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_129: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_251);  sub_34 = unsqueeze_251 = None
        unsqueeze_252: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_253: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_130: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_253);  mul_129 = unsqueeze_253 = None
        unsqueeze_254: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_255: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_77: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Tensor(mul_130, unsqueeze_255);  mul_130 = unsqueeze_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_32: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(add_77)
        mul_131: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(add_77, sigmoid_32);  add_77 = sigmoid_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_40: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_131, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_131 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_256: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_257: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_35: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_257);  convolution_40 = unsqueeze_257 = None
        add_78: "f32[32]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_32: "f32[32]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        reciprocal_32: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_132: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
        unsqueeze_259: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_133: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_259);  sub_35 = unsqueeze_259 = None
        unsqueeze_260: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_261: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_134: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_261);  mul_133 = unsqueeze_261 = None
        unsqueeze_262: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_263: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_79: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_263);  mul_134 = unsqueeze_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_33: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(add_79)
        mul_135: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_79, sigmoid_33);  add_79 = sigmoid_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_41: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_135, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_135 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_264: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_265: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_36: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_265);  convolution_41 = unsqueeze_265 = None
        add_80: "f32[64]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_33: "f32[64]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_33: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_136: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_136, -1);  mul_136 = None
        unsqueeze_267: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_137: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_267);  sub_36 = unsqueeze_267 = None
        unsqueeze_268: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_269: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_138: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_137, unsqueeze_269);  mul_137 = unsqueeze_269 = None
        unsqueeze_270: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_271: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_81: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_138, unsqueeze_271);  mul_138 = unsqueeze_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_34: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_81)
        mul_139: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_81, sigmoid_34);  add_81 = sigmoid_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:1421 in forward_features, code: x = self.stem(x)
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(mul_139, [3, 3], [2, 2], [1, 1], [1, 1], False);  mul_139 = None
        getitem_8: "f32[8, 64, 64, 64]" = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_42: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_8, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_272: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_273: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_37: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_273);  convolution_42 = unsqueeze_273 = None
        add_82: "f32[64]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_34: "f32[64]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        reciprocal_34: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_140: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_140, -1);  mul_140 = None
        unsqueeze_275: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_141: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_275);  sub_37 = unsqueeze_275 = None
        unsqueeze_276: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_277: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_142: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_277);  mul_141 = unsqueeze_277 = None
        unsqueeze_278: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_279: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_83: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_279);  mul_142 = unsqueeze_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_35: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_83)
        mul_143: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_83, sigmoid_35);  add_83 = sigmoid_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_43: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_143, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4);  mul_143 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_280: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_281: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_38: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_281);  convolution_43 = unsqueeze_281 = None
        add_84: "f32[64]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_35: "f32[64]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_35: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_144: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
        unsqueeze_283: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_145: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_283);  sub_38 = unsqueeze_283 = None
        unsqueeze_284: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_285: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_146: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_285);  mul_145 = unsqueeze_285 = None
        unsqueeze_286: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_287: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_85: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_287);  mul_146 = unsqueeze_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_36: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_85)
        mul_147: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_85, sigmoid_36);  add_85 = sigmoid_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:85 in forward, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        mean_6: "f32[8, 64]" = torch.ops.aten.mean.dim(mul_147, [2, 3])
        view_86: "f32[8, 1, 64]" = torch.ops.aten.reshape.default(mean_6, [8, 1, -1]);  mean_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:86 in forward, code: y = self.conv(y)
        convolution_44: "f32[8, 1, 64]" = torch.ops.aten.convolution.default(view_86, arg26_1, None, [1], [1], [1], False, [0], 1);  view_86 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_37: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_44);  convolution_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:90 in forward, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_87: "f32[8, 64, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_37, [8, -1, 1, 1]);  sigmoid_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:91 in forward, code: return x * y.expand_as(x)
        expand_23: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_87, [8, 64, 64, 64]);  view_87 = None
        mul_148: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_147, expand_23);  mul_147 = expand_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_45: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_148, arg27_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_148 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_288: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_289: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_39: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_289);  convolution_45 = unsqueeze_289 = None
        add_86: "f32[256]" = torch.ops.aten.add.Tensor(arg29_1, 1e-05);  arg29_1 = None
        sqrt_36: "f32[256]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_36: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_149: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_149, -1);  mul_149 = None
        unsqueeze_291: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_150: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_291);  sub_39 = unsqueeze_291 = None
        unsqueeze_292: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_293: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_151: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_293);  mul_150 = unsqueeze_293 = None
        unsqueeze_294: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_295: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_87: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_151, unsqueeze_295);  mul_151 = unsqueeze_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_46: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem_8, arg32_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_8 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_296: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_297: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_40: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_297);  convolution_46 = unsqueeze_297 = None
        add_88: "f32[256]" = torch.ops.aten.add.Tensor(arg34_1, 1e-05);  arg34_1 = None
        sqrt_37: "f32[256]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
        reciprocal_37: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_152: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_152, -1);  mul_152 = None
        unsqueeze_299: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_153: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_299);  sub_40 = unsqueeze_299 = None
        unsqueeze_300: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_301: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_154: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_301);  mul_153 = unsqueeze_301 = None
        unsqueeze_302: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_303: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_89: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_303);  mul_154 = unsqueeze_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_90: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_87, add_89);  add_87 = add_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_38: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_90)
        mul_155: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_90, sigmoid_38);  add_90 = sigmoid_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_47: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_155, arg37_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_304: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_305: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_41: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_305);  convolution_47 = unsqueeze_305 = None
        add_91: "f32[64]" = torch.ops.aten.add.Tensor(arg39_1, 1e-05);  arg39_1 = None
        sqrt_38: "f32[64]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
        reciprocal_38: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_156: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
        unsqueeze_307: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_157: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_307);  sub_41 = unsqueeze_307 = None
        unsqueeze_308: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_309: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_158: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_309);  mul_157 = unsqueeze_309 = None
        unsqueeze_310: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_311: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_92: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_311);  mul_158 = unsqueeze_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_39: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_92)
        mul_159: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_92, sigmoid_39);  add_92 = sigmoid_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_48: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_159, arg42_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4);  mul_159 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_312: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_313: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_42: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_313);  convolution_48 = unsqueeze_313 = None
        add_93: "f32[64]" = torch.ops.aten.add.Tensor(arg44_1, 1e-05);  arg44_1 = None
        sqrt_39: "f32[64]" = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_39: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_160: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
        unsqueeze_315: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_161: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_315);  sub_42 = unsqueeze_315 = None
        unsqueeze_316: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_317: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_162: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_317);  mul_161 = unsqueeze_317 = None
        unsqueeze_318: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_319: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_94: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_319);  mul_162 = unsqueeze_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_40: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_94)
        mul_163: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_94, sigmoid_40);  add_94 = sigmoid_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:85 in forward, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        mean_7: "f32[8, 64]" = torch.ops.aten.mean.dim(mul_163, [2, 3])
        view_88: "f32[8, 1, 64]" = torch.ops.aten.reshape.default(mean_7, [8, 1, -1]);  mean_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:86 in forward, code: y = self.conv(y)
        convolution_49: "f32[8, 1, 64]" = torch.ops.aten.convolution.default(view_88, arg47_1, None, [1], [1], [1], False, [0], 1);  view_88 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_41: "f32[8, 1, 64]" = torch.ops.aten.sigmoid.default(convolution_49);  convolution_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:90 in forward, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_89: "f32[8, 64, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_41, [8, -1, 1, 1]);  sigmoid_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:91 in forward, code: return x * y.expand_as(x)
        expand_24: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(view_89, [8, 64, 64, 64]);  view_89 = None
        mul_164: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_163, expand_24);  mul_163 = expand_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_50: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_164, arg48_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_164 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_43: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_321);  convolution_50 = unsqueeze_321 = None
        add_95: "f32[256]" = torch.ops.aten.add.Tensor(arg50_1, 1e-05);  arg50_1 = None
        sqrt_40: "f32[256]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_40: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_165: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
        unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_166: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_323);  sub_43 = unsqueeze_323 = None
        unsqueeze_324: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_325: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_167: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_325);  mul_166 = unsqueeze_325 = None
        unsqueeze_326: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_327: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_96: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_327);  mul_167 = unsqueeze_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_97: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_96, mul_155);  add_96 = mul_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_42: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_97)
        mul_168: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_97, sigmoid_42);  add_97 = sigmoid_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_51: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_168, arg53_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_328: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_329: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_44: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_329);  convolution_51 = unsqueeze_329 = None
        add_98: "f32[128]" = torch.ops.aten.add.Tensor(arg55_1, 1e-05);  arg55_1 = None
        sqrt_41: "f32[128]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_41: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_169: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_169, -1);  mul_169 = None
        unsqueeze_331: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_170: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_331);  sub_44 = unsqueeze_331 = None
        unsqueeze_332: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_333: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_171: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_333);  mul_170 = unsqueeze_333 = None
        unsqueeze_334: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_335: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_99: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_171, unsqueeze_335);  mul_171 = unsqueeze_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_43: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_99)
        mul_172: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_99, sigmoid_43);  add_99 = sigmoid_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_52: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_172, arg58_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  mul_172 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_336: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_337: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_45: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_337);  convolution_52 = unsqueeze_337 = None
        add_100: "f32[128]" = torch.ops.aten.add.Tensor(arg60_1, 1e-05);  arg60_1 = None
        sqrt_42: "f32[128]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        reciprocal_42: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_173: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_173, -1);  mul_173 = None
        unsqueeze_339: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_174: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_339);  sub_45 = unsqueeze_339 = None
        unsqueeze_340: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_341: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_175: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_341);  mul_174 = unsqueeze_341 = None
        unsqueeze_342: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_343: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_101: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_343);  mul_175 = unsqueeze_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_44: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_101)
        mul_176: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_101, sigmoid_44);  add_101 = sigmoid_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:85 in forward, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        mean_8: "f32[8, 128]" = torch.ops.aten.mean.dim(mul_176, [2, 3])
        view_90: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(mean_8, [8, 1, -1]);  mean_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:86 in forward, code: y = self.conv(y)
        convolution_53: "f32[8, 1, 128]" = torch.ops.aten.convolution.default(view_90, arg63_1, None, [1], [2], [1], False, [0], 1);  view_90 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_45: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:90 in forward, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_91: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_45, [8, -1, 1, 1]);  sigmoid_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:91 in forward, code: return x * y.expand_as(x)
        expand_25: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_91, [8, 128, 32, 32]);  view_91 = None
        mul_177: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_176, expand_25);  mul_176 = expand_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_54: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_177, arg64_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_177 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_344: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_345: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_46: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_345);  convolution_54 = unsqueeze_345 = None
        add_102: "f32[512]" = torch.ops.aten.add.Tensor(arg66_1, 1e-05);  arg66_1 = None
        sqrt_43: "f32[512]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_43: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_178: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_178, -1);  mul_178 = None
        unsqueeze_347: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_179: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_347);  sub_46 = unsqueeze_347 = None
        unsqueeze_348: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_349: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_180: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_349);  mul_179 = unsqueeze_349 = None
        unsqueeze_350: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_351: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_103: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_180, unsqueeze_351);  mul_180 = unsqueeze_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_55: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_168, arg69_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_168 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_352: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_353: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_47: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_353);  convolution_55 = unsqueeze_353 = None
        add_104: "f32[512]" = torch.ops.aten.add.Tensor(arg71_1, 1e-05);  arg71_1 = None
        sqrt_44: "f32[512]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_44: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_181: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_181, -1);  mul_181 = None
        unsqueeze_355: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_182: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_355);  sub_47 = unsqueeze_355 = None
        unsqueeze_356: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_357: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_183: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_357);  mul_182 = unsqueeze_357 = None
        unsqueeze_358: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_359: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_105: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_183, unsqueeze_359);  mul_183 = unsqueeze_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_106: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_103, add_105);  add_103 = add_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_46: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_106)
        mul_184: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_106, sigmoid_46);  add_106 = sigmoid_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_56: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_184, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_360: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_361: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_48: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_361);  convolution_56 = unsqueeze_361 = None
        add_107: "f32[128]" = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_45: "f32[128]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
        reciprocal_45: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_185: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_185, -1);  mul_185 = None
        unsqueeze_363: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_186: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_363);  sub_48 = unsqueeze_363 = None
        unsqueeze_364: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_365: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_187: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_186, unsqueeze_365);  mul_186 = unsqueeze_365 = None
        unsqueeze_366: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_367: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_108: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_187, unsqueeze_367);  mul_187 = unsqueeze_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_47: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_108)
        mul_188: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_108, sigmoid_47);  add_108 = sigmoid_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_57: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_188, arg79_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_188 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_368: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_369: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        sub_49: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_369);  convolution_57 = unsqueeze_369 = None
        add_109: "f32[128]" = torch.ops.aten.add.Tensor(arg81_1, 1e-05);  arg81_1 = None
        sqrt_46: "f32[128]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_46: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_189: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_370: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_371: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        mul_190: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_371);  sub_49 = unsqueeze_371 = None
        unsqueeze_372: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_373: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_191: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_373);  mul_190 = unsqueeze_373 = None
        unsqueeze_374: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_375: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_110: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_375);  mul_191 = unsqueeze_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_48: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_110)
        mul_192: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_110, sigmoid_48);  add_110 = sigmoid_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:85 in forward, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        mean_9: "f32[8, 128]" = torch.ops.aten.mean.dim(mul_192, [2, 3])
        view_92: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(mean_9, [8, 1, -1]);  mean_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:86 in forward, code: y = self.conv(y)
        convolution_58: "f32[8, 1, 128]" = torch.ops.aten.convolution.default(view_92, arg84_1, None, [1], [2], [1], False, [0], 1);  view_92 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_49: "f32[8, 1, 128]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:90 in forward, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_93: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_49, [8, -1, 1, 1]);  sigmoid_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:91 in forward, code: return x * y.expand_as(x)
        expand_26: "f32[8, 128, 32, 32]" = torch.ops.aten.expand.default(view_93, [8, 128, 32, 32]);  view_93 = None
        mul_193: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_192, expand_26);  mul_192 = expand_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_59: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_193, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_193 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_376: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_377: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        sub_50: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_377);  convolution_59 = unsqueeze_377 = None
        add_111: "f32[512]" = torch.ops.aten.add.Tensor(arg87_1, 1e-05);  arg87_1 = None
        sqrt_47: "f32[512]" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_47: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_194: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_378: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_194, -1);  mul_194 = None
        unsqueeze_379: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        mul_195: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_379);  sub_50 = unsqueeze_379 = None
        unsqueeze_380: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_381: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_196: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_381);  mul_195 = unsqueeze_381 = None
        unsqueeze_382: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_383: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_112: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_196, unsqueeze_383);  mul_196 = unsqueeze_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_113: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_112, mul_184);  add_112 = mul_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_50: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_113)
        mul_197: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_113, sigmoid_50);  add_113 = sigmoid_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_60: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_197, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_384: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_385: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        sub_51: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_385);  convolution_60 = unsqueeze_385 = None
        add_114: "f32[256]" = torch.ops.aten.add.Tensor(arg92_1, 1e-05);  arg92_1 = None
        sqrt_48: "f32[256]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_48: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_386: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_387: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        mul_199: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_387);  sub_51 = unsqueeze_387 = None
        unsqueeze_388: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_389: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_200: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_389);  mul_199 = unsqueeze_389 = None
        unsqueeze_390: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_391: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_115: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_391);  mul_200 = unsqueeze_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_51: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_115)
        mul_201: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_115, sigmoid_51);  add_115 = sigmoid_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_61: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_201, arg95_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  mul_201 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_392: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_393: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        sub_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_393);  convolution_61 = unsqueeze_393 = None
        add_116: "f32[256]" = torch.ops.aten.add.Tensor(arg97_1, 1e-05);  arg97_1 = None
        sqrt_49: "f32[256]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_49: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_202: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_394: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_202, -1);  mul_202 = None
        unsqueeze_395: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        mul_203: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_395);  sub_52 = unsqueeze_395 = None
        unsqueeze_396: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_397: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_204: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_397);  mul_203 = unsqueeze_397 = None
        unsqueeze_398: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_399: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_117: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_399);  mul_204 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_117)
        mul_205: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_117, sigmoid_52);  add_117 = sigmoid_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:85 in forward, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        mean_10: "f32[8, 256]" = torch.ops.aten.mean.dim(mul_205, [2, 3])
        view_94: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(mean_10, [8, 1, -1]);  mean_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:86 in forward, code: y = self.conv(y)
        convolution_62: "f32[8, 1, 256]" = torch.ops.aten.convolution.default(view_94, arg100_1, None, [1], [2], [1], False, [0], 1);  view_94 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_53: "f32[8, 1, 256]" = torch.ops.aten.sigmoid.default(convolution_62);  convolution_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:90 in forward, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
        view_95: "f32[8, 256, 1, 1]" = torch.ops.aten.reshape.default(sigmoid_53, [8, -1, 1, 1]);  sigmoid_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/eca.py:91 in forward, code: return x * y.expand_as(x)
        expand_27: "f32[8, 256, 16, 16]" = torch.ops.aten.expand.default(view_95, [8, 256, 16, 16]);  view_95 = None
        mul_206: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_205, expand_27);  mul_205 = expand_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_63: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_206, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_206 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_400: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_401: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        sub_53: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_401);  convolution_63 = unsqueeze_401 = None
        add_118: "f32[1024]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_50: "f32[1024]" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_50: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_207: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_402: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_403: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        mul_208: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_403);  sub_53 = unsqueeze_403 = None
        unsqueeze_404: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_405: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_209: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_405);  mul_208 = unsqueeze_405 = None
        unsqueeze_406: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_407: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_119: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_407);  mul_209 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_64: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_197, arg106_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_197 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_408: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_409: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        sub_54: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_409);  convolution_64 = unsqueeze_409 = None
        add_120: "f32[1024]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_51: "f32[1024]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_51: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_210: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_410: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_411: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        mul_211: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_411);  sub_54 = unsqueeze_411 = None
        unsqueeze_412: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_413: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_212: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_413);  mul_211 = unsqueeze_413 = None
        unsqueeze_414: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_415: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_121: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_415);  mul_212 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_122: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_119, add_121);  add_119 = add_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        sigmoid_54: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_122)
        mul_213: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_122, sigmoid_54);  add_122 = sigmoid_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_65: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_213, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_416: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_417: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        sub_55: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_417);  convolution_65 = unsqueeze_417 = None
        add_123: "f32[256]" = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_52: "f32[256]" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_52: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_214: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_418: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_214, -1);  mul_214 = None
        unsqueeze_419: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        mul_215: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_419);  sub_55 = unsqueeze_419 = None
        unsqueeze_420: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_421: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_216: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_215, unsqueeze_421);  mul_215 = unsqueeze_421 = None
        unsqueeze_422: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_423: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_124: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_423);  mul_216 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_55: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_124)
        mul_217: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_124, sigmoid_55);  add_124 = sigmoid_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:186 in forward, code: kv = self.kv(x)
        convolution_67: "f32[8, 384, 16, 16]" = torch.ops.aten.convolution.default(mul_217, arg117_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg117_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_15: "f32[8, 384, 20, 20]" = torch.ops.aten.constant_pad_nd.default(convolution_67, [2, 2, 2, 2], 0.0);  convolution_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:191 in forward, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
        unfold_6: "f32[8, 384, 2, 20, 12]" = torch.ops.aten.unfold.default(constant_pad_nd_15, 2, 12, 8);  constant_pad_nd_15 = None
        unfold_7: "f32[8, 384, 2, 2, 12, 12]" = torch.ops.aten.unfold.default(unfold_6, 3, 12, 8);  unfold_6 = None
        clone_26: "f32[8, 384, 2, 2, 12, 12]" = torch.ops.aten.clone.default(unfold_7, memory_format = torch.contiguous_format);  unfold_7 = None
        view_98: "f32[64, 48, 4, 144]" = torch.ops.aten.reshape.default(clone_26, [64, 48, 4, 144]);  clone_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:192 in forward, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        permute_36: "f32[64, 4, 144, 48]" = torch.ops.aten.permute.default(view_98, [0, 2, 3, 1]);  view_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:193 in forward, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(permute_36, [16, 32], -1);  permute_36 = None
        getitem_10: "f32[64, 4, 144, 16]" = split_with_sizes_3[0]
        getitem_11: "f32[64, 4, 144, 32]" = split_with_sizes_3[1];  split_with_sizes_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:177 in forward, code: q = self.q(x)
        convolution_66: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(mul_217, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_217 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:179 in forward, code: q = q.reshape(
        view_96: "f32[64, 16, 2, 8, 2, 8]" = torch.ops.aten.reshape.default(convolution_66, [-1, 16, 2, 8, 2, 8]);  convolution_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:181 in forward, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        permute_34: "f32[64, 16, 8, 8, 2, 2]" = torch.ops.aten.permute.default(view_96, [0, 1, 3, 5, 2, 4]);  view_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:183 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
        clone_25: "f32[64, 16, 8, 8, 2, 2]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_97: "f32[64, 16, 64, 4]" = torch.ops.aten.reshape.default(clone_25, [64, 16, 64, 4]);  clone_25 = None
        permute_35: "f32[64, 4, 64, 16]" = torch.ops.aten.permute.default(view_97, [0, 3, 2, 1]);  view_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:199 in forward, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
        expand_28: "f32[64, 4, 64, 16]" = torch.ops.aten.expand.default(permute_35, [64, 4, 64, 16])
        clone_27: "f32[64, 4, 64, 16]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_99: "f32[256, 64, 16]" = torch.ops.aten.reshape.default(clone_27, [256, 64, 16]);  clone_27 = None
        permute_37: "f32[64, 4, 16, 144]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
        expand_29: "f32[64, 4, 16, 144]" = torch.ops.aten.expand.default(permute_37, [64, 4, 16, 144]);  permute_37 = None
        clone_28: "f32[64, 4, 16, 144]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_100: "f32[256, 16, 144]" = torch.ops.aten.reshape.default(clone_28, [256, 16, 144]);  clone_28 = None
        bmm_6: "f32[256, 64, 144]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
        view_101: "f32[64, 4, 64, 144]" = torch.ops.aten.reshape.default(bmm_6, [64, 4, 64, 144]);  bmm_6 = None
        mul_218: "f32[64, 4, 64, 144]" = torch.ops.aten.mul.Tensor(view_101, 0.25);  view_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:85 in forward, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        clone_29: "f32[64, 4, 64, 16]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_102: "f32[256, 8, 8, 16]" = torch.ops.aten.reshape.default(clone_29, [256, 8, 8, 16]);  clone_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:89 in forward, code: q = q.transpose(1, 2)
        permute_40: "f32[256, 8, 8, 16]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:45 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_30: "f32[256, 8, 8, 16]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_109: "f32[16384, 16]" = torch.ops.aten.reshape.default(clone_30, [16384, 16]);  clone_30 = None
        permute_41: "f32[16, 23]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_7: "f32[16384, 23]" = torch.ops.aten.mm.default(view_109, permute_41);  view_109 = permute_41 = None
        view_110: "f32[256, 8, 8, 23]" = torch.ops.aten.reshape.default(mm_7, [256, 8, 8, 23]);  mm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:46 in rel_logits_1d, code: x = x.reshape(-1, W, rel_size)
        view_111: "f32[2048, 8, 23]" = torch.ops.aten.reshape.default(view_110, [-1, 8, 23]);  view_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_18: "f32[2048, 8, 24]" = torch.ops.aten.constant_pad_nd.default(view_111, [0, 1], 0.0);  view_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:49 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_112: "f32[2048, 192]" = torch.ops.aten.reshape.default(constant_pad_nd_18, [2048, 192]);  constant_pad_nd_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_19: "f32[2048, 207]" = torch.ops.aten.constant_pad_nd.default(view_112, [0, 15], 0.0);  view_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:53 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
        view_113: "f32[2048, 9, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_19, [-1, 9, 23]);  constant_pad_nd_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:54 in rel_logits_1d, code: x = x_pad[:, :W, win_size - 1:]
        slice_23: "f32[2048, 8, 23]" = torch.ops.aten.slice.Tensor(view_113, 1, 0, 8);  view_113 = None
        slice_24: "f32[2048, 8, 12]" = torch.ops.aten.slice.Tensor(slice_23, 2, 11, 9223372036854775807);  slice_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:57 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
        view_114: "f32[256, 8, 1, 8, 12]" = torch.ops.aten.reshape.default(slice_24, [256, 8, 1, 8, 12]);  slice_24 = None
        expand_31: "f32[256, 8, 12, 8, 12]" = torch.ops.aten.expand.default(view_114, [-1, -1, 12, -1, -1]);  view_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:58 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_42: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.permute.default(expand_31, [0, 3, 1, 4, 2]);  expand_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:45 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        view_103: "f32[16384, 16]" = torch.ops.aten.reshape.default(view_102, [16384, 16]);  view_102 = None
        permute_38: "f32[16, 23]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_6: "f32[16384, 23]" = torch.ops.aten.mm.default(view_103, permute_38);  view_103 = permute_38 = None
        view_104: "f32[256, 8, 8, 23]" = torch.ops.aten.reshape.default(mm_6, [256, 8, 8, 23]);  mm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:46 in rel_logits_1d, code: x = x.reshape(-1, W, rel_size)
        view_105: "f32[2048, 8, 23]" = torch.ops.aten.reshape.default(view_104, [-1, 8, 23]);  view_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_16: "f32[2048, 8, 24]" = torch.ops.aten.constant_pad_nd.default(view_105, [0, 1], 0.0);  view_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:49 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_106: "f32[2048, 192]" = torch.ops.aten.reshape.default(constant_pad_nd_16, [2048, 192]);  constant_pad_nd_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_17: "f32[2048, 207]" = torch.ops.aten.constant_pad_nd.default(view_106, [0, 15], 0.0);  view_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:53 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
        view_107: "f32[2048, 9, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_17, [-1, 9, 23]);  constant_pad_nd_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:54 in rel_logits_1d, code: x = x_pad[:, :W, win_size - 1:]
        slice_20: "f32[2048, 8, 23]" = torch.ops.aten.slice.Tensor(view_107, 1, 0, 8);  view_107 = None
        slice_21: "f32[2048, 8, 12]" = torch.ops.aten.slice.Tensor(slice_20, 2, 11, 9223372036854775807);  slice_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:57 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
        view_108: "f32[256, 8, 1, 8, 12]" = torch.ops.aten.reshape.default(slice_21, [256, 8, 1, 8, 12]);  slice_21 = None
        expand_30: "f32[256, 8, 12, 8, 12]" = torch.ops.aten.expand.default(view_108, [-1, -1, 12, -1, -1]);  view_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:58 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_39: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.permute.default(expand_30, [0, 1, 3, 2, 4]);  expand_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:92 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_125: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.add.Tensor(permute_42, permute_39);  permute_42 = permute_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:93 in forward, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
        clone_31: "f32[256, 8, 8, 12, 12]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format);  add_125 = None
        view_115: "f32[64, 4, 64, 144]" = torch.ops.aten.reshape.default(clone_31, [64, 4, 64, 144]);  clone_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:199 in forward, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
        add_126: "f32[64, 4, 64, 144]" = torch.ops.aten.add.Tensor(mul_218, view_115);  mul_218 = view_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:201 in forward, code: attn = attn.softmax(dim=-1)
        amax_3: "f32[64, 4, 64, 1]" = torch.ops.aten.amax.default(add_126, [-1], True)
        sub_56: "f32[64, 4, 64, 144]" = torch.ops.aten.sub.Tensor(add_126, amax_3);  add_126 = amax_3 = None
        exp_3: "f32[64, 4, 64, 144]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
        sum_4: "f32[64, 4, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[64, 4, 64, 144]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:203 in forward, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        expand_32: "f32[64, 4, 64, 144]" = torch.ops.aten.expand.default(div_3, [64, 4, 64, 144]);  div_3 = None
        view_116: "f32[256, 64, 144]" = torch.ops.aten.reshape.default(expand_32, [256, 64, 144]);  expand_32 = None
        expand_33: "f32[64, 4, 144, 32]" = torch.ops.aten.expand.default(getitem_11, [64, 4, 144, 32]);  getitem_11 = None
        clone_32: "f32[64, 4, 144, 32]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_117: "f32[256, 144, 32]" = torch.ops.aten.reshape.default(clone_32, [256, 144, 32]);  clone_32 = None
        bmm_7: "f32[256, 64, 32]" = torch.ops.aten.bmm.default(view_116, view_117);  view_116 = view_117 = None
        view_118: "f32[64, 4, 64, 32]" = torch.ops.aten.reshape.default(bmm_7, [64, 4, 64, 32]);  bmm_7 = None
        permute_43: "f32[64, 32, 64, 4]" = torch.ops.aten.permute.default(view_118, [0, 3, 2, 1]);  view_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:205 in forward, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        clone_33: "f32[64, 32, 64, 4]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_119: "f32[2048, 8, 8, 2, 2]" = torch.ops.aten.reshape.default(clone_33, [2048, 8, 8, 2, 2]);  clone_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:206 in forward, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
        permute_44: "f32[2048, 2, 8, 2, 8]" = torch.ops.aten.permute.default(view_119, [0, 3, 1, 4, 2]);  view_119 = None
        clone_34: "f32[2048, 2, 8, 2, 8]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_120: "f32[8, 256, 16, 16]" = torch.ops.aten.reshape.default(clone_34, [8, 256, 16, 16]);  clone_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_424: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_425: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        sub_57: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_425);  view_120 = unsqueeze_425 = None
        add_127: "f32[256]" = torch.ops.aten.add.Tensor(arg121_1, 1e-05);  arg121_1 = None
        sqrt_53: "f32[256]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_53: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_426: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_427: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        mul_220: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_427);  sub_57 = unsqueeze_427 = None
        unsqueeze_428: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_429: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_221: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_429);  mul_220 = unsqueeze_429 = None
        unsqueeze_430: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_431: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_128: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_431);  mul_221 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_56: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_128)
        mul_222: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_128, sigmoid_56);  add_128 = sigmoid_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_68: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_222, arg124_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_222 = arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_432: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_433: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        sub_58: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_433);  convolution_68 = unsqueeze_433 = None
        add_129: "f32[1024]" = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
        sqrt_54: "f32[1024]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_54: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_223: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_434: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_223, -1);  mul_223 = None
        unsqueeze_435: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        mul_224: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_435);  sub_58 = unsqueeze_435 = None
        unsqueeze_436: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_437: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_225: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_437);  mul_224 = unsqueeze_437 = None
        unsqueeze_438: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_439: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_130: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_225, unsqueeze_439);  mul_225 = unsqueeze_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_131: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_130, mul_213);  add_130 = mul_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        sigmoid_57: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_131)
        mul_226: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_131, sigmoid_57);  add_131 = sigmoid_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_69: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_226, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_440: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_441: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        sub_59: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_441);  convolution_69 = unsqueeze_441 = None
        add_132: "f32[512]" = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_55: "f32[512]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_55: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_227: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_442: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_227, -1);  mul_227 = None
        unsqueeze_443: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        mul_228: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_443);  sub_59 = unsqueeze_443 = None
        unsqueeze_444: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_445: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_229: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_228, unsqueeze_445);  mul_228 = unsqueeze_445 = None
        unsqueeze_446: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_447: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_133: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_229, unsqueeze_447);  mul_229 = unsqueeze_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_58: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_133)
        mul_230: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_133, sigmoid_58);  add_133 = sigmoid_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:186 in forward, code: kv = self.kv(x)
        convolution_71: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(mul_230, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_20: "f32[8, 640, 20, 20]" = torch.ops.aten.constant_pad_nd.default(convolution_71, [2, 2, 2, 2], 0.0);  convolution_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:191 in forward, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
        unfold_8: "f32[8, 640, 2, 20, 12]" = torch.ops.aten.unfold.default(constant_pad_nd_20, 2, 12, 8);  constant_pad_nd_20 = None
        unfold_9: "f32[8, 640, 2, 2, 12, 12]" = torch.ops.aten.unfold.default(unfold_8, 3, 12, 8);  unfold_8 = None
        clone_36: "f32[8, 640, 2, 2, 12, 12]" = torch.ops.aten.clone.default(unfold_9, memory_format = torch.contiguous_format);  unfold_9 = None
        view_123: "f32[64, 80, 4, 144]" = torch.ops.aten.reshape.default(clone_36, [64, 80, 4, 144]);  clone_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:192 in forward, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        permute_47: "f32[64, 4, 144, 80]" = torch.ops.aten.permute.default(view_123, [0, 2, 3, 1]);  view_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:193 in forward, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(permute_47, [16, 64], -1);  permute_47 = None
        getitem_12: "f32[64, 4, 144, 16]" = split_with_sizes_4[0]
        getitem_13: "f32[64, 4, 144, 64]" = split_with_sizes_4[1];  split_with_sizes_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:177 in forward, code: q = self.q(x)
        convolution_70: "f32[8, 128, 8, 8]" = torch.ops.aten.convolution.default(mul_230, arg134_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_230 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:179 in forward, code: q = q.reshape(
        view_121: "f32[64, 16, 2, 4, 2, 4]" = torch.ops.aten.reshape.default(convolution_70, [-1, 16, 2, 4, 2, 4]);  convolution_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:181 in forward, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        permute_45: "f32[64, 16, 4, 4, 2, 2]" = torch.ops.aten.permute.default(view_121, [0, 1, 3, 5, 2, 4]);  view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:183 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
        clone_35: "f32[64, 16, 4, 4, 2, 2]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_122: "f32[64, 16, 16, 4]" = torch.ops.aten.reshape.default(clone_35, [64, 16, 16, 4]);  clone_35 = None
        permute_46: "f32[64, 4, 16, 16]" = torch.ops.aten.permute.default(view_122, [0, 3, 2, 1]);  view_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:199 in forward, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
        expand_34: "f32[64, 4, 16, 16]" = torch.ops.aten.expand.default(permute_46, [64, 4, 16, 16])
        clone_37: "f32[64, 4, 16, 16]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_124: "f32[256, 16, 16]" = torch.ops.aten.reshape.default(clone_37, [256, 16, 16]);  clone_37 = None
        permute_48: "f32[64, 4, 16, 144]" = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_35: "f32[64, 4, 16, 144]" = torch.ops.aten.expand.default(permute_48, [64, 4, 16, 144]);  permute_48 = None
        clone_38: "f32[64, 4, 16, 144]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_125: "f32[256, 16, 144]" = torch.ops.aten.reshape.default(clone_38, [256, 16, 144]);  clone_38 = None
        bmm_8: "f32[256, 16, 144]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
        view_126: "f32[64, 4, 16, 144]" = torch.ops.aten.reshape.default(bmm_8, [64, 4, 16, 144]);  bmm_8 = None
        mul_231: "f32[64, 4, 16, 144]" = torch.ops.aten.mul.Tensor(view_126, 0.25);  view_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:85 in forward, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        clone_39: "f32[64, 4, 16, 16]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_127: "f32[256, 4, 4, 16]" = torch.ops.aten.reshape.default(clone_39, [256, 4, 4, 16]);  clone_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:89 in forward, code: q = q.transpose(1, 2)
        permute_51: "f32[256, 4, 4, 16]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:45 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_40: "f32[256, 4, 4, 16]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_134: "f32[4096, 16]" = torch.ops.aten.reshape.default(clone_40, [4096, 16]);  clone_40 = None
        permute_52: "f32[16, 23]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_9: "f32[4096, 23]" = torch.ops.aten.mm.default(view_134, permute_52);  view_134 = permute_52 = None
        view_135: "f32[256, 4, 4, 23]" = torch.ops.aten.reshape.default(mm_9, [256, 4, 4, 23]);  mm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:46 in rel_logits_1d, code: x = x.reshape(-1, W, rel_size)
        view_136: "f32[1024, 4, 23]" = torch.ops.aten.reshape.default(view_135, [-1, 4, 23]);  view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_23: "f32[1024, 4, 24]" = torch.ops.aten.constant_pad_nd.default(view_136, [0, 1], 0.0);  view_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:49 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_137: "f32[1024, 96]" = torch.ops.aten.reshape.default(constant_pad_nd_23, [1024, 96]);  constant_pad_nd_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_24: "f32[1024, 115]" = torch.ops.aten.constant_pad_nd.default(view_137, [0, 19], 0.0);  view_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:53 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
        view_138: "f32[1024, 5, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_24, [-1, 5, 23]);  constant_pad_nd_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:54 in rel_logits_1d, code: x = x_pad[:, :W, win_size - 1:]
        slice_29: "f32[1024, 4, 23]" = torch.ops.aten.slice.Tensor(view_138, 1, 0, 4);  view_138 = None
        slice_30: "f32[1024, 4, 12]" = torch.ops.aten.slice.Tensor(slice_29, 2, 11, 9223372036854775807);  slice_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:57 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
        view_139: "f32[256, 4, 1, 4, 12]" = torch.ops.aten.reshape.default(slice_30, [256, 4, 1, 4, 12]);  slice_30 = None
        expand_37: "f32[256, 4, 12, 4, 12]" = torch.ops.aten.expand.default(view_139, [-1, -1, 12, -1, -1]);  view_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:58 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_53: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.permute.default(expand_37, [0, 3, 1, 4, 2]);  expand_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:45 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        view_128: "f32[4096, 16]" = torch.ops.aten.reshape.default(view_127, [4096, 16]);  view_127 = None
        permute_49: "f32[16, 23]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_8: "f32[4096, 23]" = torch.ops.aten.mm.default(view_128, permute_49);  view_128 = permute_49 = None
        view_129: "f32[256, 4, 4, 23]" = torch.ops.aten.reshape.default(mm_8, [256, 4, 4, 23]);  mm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:46 in rel_logits_1d, code: x = x.reshape(-1, W, rel_size)
        view_130: "f32[1024, 4, 23]" = torch.ops.aten.reshape.default(view_129, [-1, 4, 23]);  view_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_21: "f32[1024, 4, 24]" = torch.ops.aten.constant_pad_nd.default(view_130, [0, 1], 0.0);  view_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:49 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_131: "f32[1024, 96]" = torch.ops.aten.reshape.default(constant_pad_nd_21, [1024, 96]);  constant_pad_nd_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_22: "f32[1024, 115]" = torch.ops.aten.constant_pad_nd.default(view_131, [0, 19], 0.0);  view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:53 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
        view_132: "f32[1024, 5, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_22, [-1, 5, 23]);  constant_pad_nd_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:54 in rel_logits_1d, code: x = x_pad[:, :W, win_size - 1:]
        slice_26: "f32[1024, 4, 23]" = torch.ops.aten.slice.Tensor(view_132, 1, 0, 4);  view_132 = None
        slice_27: "f32[1024, 4, 12]" = torch.ops.aten.slice.Tensor(slice_26, 2, 11, 9223372036854775807);  slice_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:57 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
        view_133: "f32[256, 4, 1, 4, 12]" = torch.ops.aten.reshape.default(slice_27, [256, 4, 1, 4, 12]);  slice_27 = None
        expand_36: "f32[256, 4, 12, 4, 12]" = torch.ops.aten.expand.default(view_133, [-1, -1, 12, -1, -1]);  view_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:58 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_50: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.permute.default(expand_36, [0, 1, 3, 2, 4]);  expand_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:92 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_134: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.add.Tensor(permute_53, permute_50);  permute_53 = permute_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:93 in forward, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
        clone_41: "f32[256, 4, 4, 12, 12]" = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format);  add_134 = None
        view_140: "f32[64, 4, 16, 144]" = torch.ops.aten.reshape.default(clone_41, [64, 4, 16, 144]);  clone_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:199 in forward, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
        add_135: "f32[64, 4, 16, 144]" = torch.ops.aten.add.Tensor(mul_231, view_140);  mul_231 = view_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:201 in forward, code: attn = attn.softmax(dim=-1)
        amax_4: "f32[64, 4, 16, 1]" = torch.ops.aten.amax.default(add_135, [-1], True)
        sub_60: "f32[64, 4, 16, 144]" = torch.ops.aten.sub.Tensor(add_135, amax_4);  add_135 = amax_4 = None
        exp_4: "f32[64, 4, 16, 144]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_5: "f32[64, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[64, 4, 16, 144]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:203 in forward, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        expand_38: "f32[64, 4, 16, 144]" = torch.ops.aten.expand.default(div_4, [64, 4, 16, 144]);  div_4 = None
        view_141: "f32[256, 16, 144]" = torch.ops.aten.reshape.default(expand_38, [256, 16, 144]);  expand_38 = None
        expand_39: "f32[64, 4, 144, 64]" = torch.ops.aten.expand.default(getitem_13, [64, 4, 144, 64]);  getitem_13 = None
        clone_42: "f32[64, 4, 144, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_142: "f32[256, 144, 64]" = torch.ops.aten.reshape.default(clone_42, [256, 144, 64]);  clone_42 = None
        bmm_9: "f32[256, 16, 64]" = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
        view_143: "f32[64, 4, 16, 64]" = torch.ops.aten.reshape.default(bmm_9, [64, 4, 16, 64]);  bmm_9 = None
        permute_54: "f32[64, 64, 16, 4]" = torch.ops.aten.permute.default(view_143, [0, 3, 2, 1]);  view_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:205 in forward, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        clone_43: "f32[64, 64, 16, 4]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_144: "f32[4096, 4, 4, 2, 2]" = torch.ops.aten.reshape.default(clone_43, [4096, 4, 4, 2, 2]);  clone_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:206 in forward, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
        permute_55: "f32[4096, 2, 4, 2, 4]" = torch.ops.aten.permute.default(view_144, [0, 3, 1, 4, 2]);  view_144 = None
        clone_44: "f32[4096, 2, 4, 2, 4]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_145: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(clone_44, [8, 512, 8, 8]);  clone_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_448: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_449: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        sub_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_145, unsqueeze_449);  view_145 = unsqueeze_449 = None
        add_136: "f32[512]" = torch.ops.aten.add.Tensor(arg139_1, 1e-05);  arg139_1 = None
        sqrt_56: "f32[512]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_56: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_450: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_232, -1);  mul_232 = None
        unsqueeze_451: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        mul_233: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_451);  sub_61 = unsqueeze_451 = None
        unsqueeze_452: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_453: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_234: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_453);  mul_233 = unsqueeze_453 = None
        unsqueeze_454: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_455: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_137: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_234, unsqueeze_455);  mul_234 = unsqueeze_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_59: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_137)
        mul_235: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_59);  add_137 = sigmoid_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_72: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_235, arg142_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_235 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_456: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_457: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        sub_62: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_457);  convolution_72 = unsqueeze_457 = None
        add_138: "f32[2048]" = torch.ops.aten.add.Tensor(arg144_1, 1e-05);  arg144_1 = None
        sqrt_57: "f32[2048]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_57: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_236: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_458: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_236, -1);  mul_236 = None
        unsqueeze_459: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        mul_237: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_459);  sub_62 = unsqueeze_459 = None
        unsqueeze_460: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_461: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_238: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_461);  mul_237 = unsqueeze_461 = None
        unsqueeze_462: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_463: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_139: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_463);  mul_238 = unsqueeze_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_73: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_226, arg147_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_226 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_464: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_465: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        sub_63: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_465);  convolution_73 = unsqueeze_465 = None
        add_140: "f32[2048]" = torch.ops.aten.add.Tensor(arg149_1, 1e-05);  arg149_1 = None
        sqrt_58: "f32[2048]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_58: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_239: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_466: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_239, -1);  mul_239 = None
        unsqueeze_467: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        mul_240: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_467);  sub_63 = unsqueeze_467 = None
        unsqueeze_468: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_469: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_241: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_469);  mul_240 = unsqueeze_469 = None
        unsqueeze_470: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_471: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_141: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_241, unsqueeze_471);  mul_241 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_142: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_139, add_141);  add_139 = add_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        sigmoid_60: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_142)
        mul_242: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_142, sigmoid_60);  add_142 = sigmoid_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_74: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_242, arg152_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_472: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_473: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        sub_64: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_473);  convolution_74 = unsqueeze_473 = None
        add_143: "f32[512]" = torch.ops.aten.add.Tensor(arg154_1, 1e-05);  arg154_1 = None
        sqrt_59: "f32[512]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_59: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_243: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_474: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_475: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        mul_244: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_475);  sub_64 = unsqueeze_475 = None
        unsqueeze_476: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_477: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_245: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_477);  mul_244 = unsqueeze_477 = None
        unsqueeze_478: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_479: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_144: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_479);  mul_245 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_144)
        mul_246: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_144, sigmoid_61);  add_144 = sigmoid_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:186 in forward, code: kv = self.kv(x)
        convolution_76: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(mul_246, arg158_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_25: "f32[8, 640, 12, 12]" = torch.ops.aten.constant_pad_nd.default(convolution_76, [2, 2, 2, 2], 0.0);  convolution_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:191 in forward, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
        unfold_10: "f32[8, 640, 1, 12, 12]" = torch.ops.aten.unfold.default(constant_pad_nd_25, 2, 12, 8);  constant_pad_nd_25 = None
        unfold_11: "f32[8, 640, 1, 1, 12, 12]" = torch.ops.aten.unfold.default(unfold_10, 3, 12, 8);  unfold_10 = None
        view_148: "f32[64, 80, 1, 144]" = torch.ops.aten.reshape.default(unfold_11, [64, 80, 1, -1]);  unfold_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:192 in forward, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        permute_58: "f32[64, 1, 144, 80]" = torch.ops.aten.permute.default(view_148, [0, 2, 3, 1]);  view_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:193 in forward, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(permute_58, [16, 64], -1);  permute_58 = None
        getitem_14: "f32[64, 1, 144, 16]" = split_with_sizes_5[0]
        getitem_15: "f32[64, 1, 144, 64]" = split_with_sizes_5[1];  split_with_sizes_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:177 in forward, code: q = self.q(x)
        convolution_75: "f32[8, 128, 8, 8]" = torch.ops.aten.convolution.default(mul_246, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_246 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:179 in forward, code: q = q.reshape(
        view_146: "f32[64, 16, 1, 8, 1, 8]" = torch.ops.aten.reshape.default(convolution_75, [-1, 16, 1, 8, 1, 8]);  convolution_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:181 in forward, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        permute_56: "f32[64, 16, 8, 8, 1, 1]" = torch.ops.aten.permute.default(view_146, [0, 1, 3, 5, 2, 4]);  view_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:183 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
        view_147: "f32[64, 16, 64, 1]" = torch.ops.aten.reshape.default(permute_56, [64, 16, -1, 1]);  permute_56 = None
        permute_57: "f32[64, 1, 64, 16]" = torch.ops.aten.permute.default(view_147, [0, 3, 2, 1]);  view_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:199 in forward, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
        expand_40: "f32[64, 1, 64, 16]" = torch.ops.aten.expand.default(permute_57, [64, 1, 64, 16])
        view_149: "f32[64, 64, 16]" = torch.ops.aten.reshape.default(expand_40, [64, 64, 16]);  expand_40 = None
        permute_59: "f32[64, 1, 16, 144]" = torch.ops.aten.permute.default(getitem_14, [0, 1, 3, 2]);  getitem_14 = None
        expand_41: "f32[64, 1, 16, 144]" = torch.ops.aten.expand.default(permute_59, [64, 1, 16, 144]);  permute_59 = None
        view_150: "f32[64, 16, 144]" = torch.ops.aten.reshape.default(expand_41, [64, 16, 144]);  expand_41 = None
        bmm_10: "f32[64, 64, 144]" = torch.ops.aten.bmm.default(view_149, view_150);  view_149 = view_150 = None
        view_151: "f32[64, 1, 64, 144]" = torch.ops.aten.reshape.default(bmm_10, [64, 1, 64, 144]);  bmm_10 = None
        mul_247: "f32[64, 1, 64, 144]" = torch.ops.aten.mul.Tensor(view_151, 0.25);  view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:85 in forward, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        view_152: "f32[64, 8, 8, 16]" = torch.ops.aten.reshape.default(permute_57, [64, 8, 8, 16]);  permute_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:89 in forward, code: q = q.transpose(1, 2)
        permute_62: "f32[64, 8, 8, 16]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:45 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_46: "f32[64, 8, 8, 16]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_159: "f32[4096, 16]" = torch.ops.aten.reshape.default(clone_46, [4096, 16]);  clone_46 = None
        permute_63: "f32[16, 23]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_11: "f32[4096, 23]" = torch.ops.aten.mm.default(view_159, permute_63);  view_159 = permute_63 = None
        view_160: "f32[64, 8, 8, 23]" = torch.ops.aten.reshape.default(mm_11, [64, 8, 8, 23]);  mm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:46 in rel_logits_1d, code: x = x.reshape(-1, W, rel_size)
        view_161: "f32[512, 8, 23]" = torch.ops.aten.reshape.default(view_160, [-1, 8, 23]);  view_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_28: "f32[512, 8, 24]" = torch.ops.aten.constant_pad_nd.default(view_161, [0, 1], 0.0);  view_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:49 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_162: "f32[512, 192]" = torch.ops.aten.reshape.default(constant_pad_nd_28, [512, 192]);  constant_pad_nd_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_29: "f32[512, 207]" = torch.ops.aten.constant_pad_nd.default(view_162, [0, 15], 0.0);  view_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:53 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
        view_163: "f32[512, 9, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_29, [-1, 9, 23]);  constant_pad_nd_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:54 in rel_logits_1d, code: x = x_pad[:, :W, win_size - 1:]
        slice_35: "f32[512, 8, 23]" = torch.ops.aten.slice.Tensor(view_163, 1, 0, 8);  view_163 = None
        slice_36: "f32[512, 8, 12]" = torch.ops.aten.slice.Tensor(slice_35, 2, 11, 9223372036854775807);  slice_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:57 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
        view_164: "f32[64, 8, 1, 8, 12]" = torch.ops.aten.reshape.default(slice_36, [64, 8, 1, 8, 12]);  slice_36 = None
        expand_43: "f32[64, 8, 12, 8, 12]" = torch.ops.aten.expand.default(view_164, [-1, -1, 12, -1, -1]);  view_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:58 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_64: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.permute.default(expand_43, [0, 3, 1, 4, 2]);  expand_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:45 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_45: "f32[64, 8, 8, 16]" = torch.ops.aten.clone.default(view_152, memory_format = torch.contiguous_format);  view_152 = None
        view_153: "f32[4096, 16]" = torch.ops.aten.reshape.default(clone_45, [4096, 16]);  clone_45 = None
        permute_60: "f32[16, 23]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_10: "f32[4096, 23]" = torch.ops.aten.mm.default(view_153, permute_60);  view_153 = permute_60 = None
        view_154: "f32[64, 8, 8, 23]" = torch.ops.aten.reshape.default(mm_10, [64, 8, 8, 23]);  mm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:46 in rel_logits_1d, code: x = x.reshape(-1, W, rel_size)
        view_155: "f32[512, 8, 23]" = torch.ops.aten.reshape.default(view_154, [-1, 8, 23]);  view_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_26: "f32[512, 8, 24]" = torch.ops.aten.constant_pad_nd.default(view_155, [0, 1], 0.0);  view_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:49 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_156: "f32[512, 192]" = torch.ops.aten.reshape.default(constant_pad_nd_26, [512, 192]);  constant_pad_nd_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_27: "f32[512, 207]" = torch.ops.aten.constant_pad_nd.default(view_156, [0, 15], 0.0);  view_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:53 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
        view_157: "f32[512, 9, 23]" = torch.ops.aten.reshape.default(constant_pad_nd_27, [-1, 9, 23]);  constant_pad_nd_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:54 in rel_logits_1d, code: x = x_pad[:, :W, win_size - 1:]
        slice_32: "f32[512, 8, 23]" = torch.ops.aten.slice.Tensor(view_157, 1, 0, 8);  view_157 = None
        slice_33: "f32[512, 8, 12]" = torch.ops.aten.slice.Tensor(slice_32, 2, 11, 9223372036854775807);  slice_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:57 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
        view_158: "f32[64, 8, 1, 8, 12]" = torch.ops.aten.reshape.default(slice_33, [64, 8, 1, 8, 12]);  slice_33 = None
        expand_42: "f32[64, 8, 12, 8, 12]" = torch.ops.aten.expand.default(view_158, [-1, -1, 12, -1, -1]);  view_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:58 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_61: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.permute.default(expand_42, [0, 1, 3, 2, 4]);  expand_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:92 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_145: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.add.Tensor(permute_64, permute_61);  permute_64 = permute_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:93 in forward, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
        clone_47: "f32[64, 8, 8, 12, 12]" = torch.ops.aten.clone.default(add_145, memory_format = torch.contiguous_format);  add_145 = None
        view_165: "f32[64, 1, 64, 144]" = torch.ops.aten.reshape.default(clone_47, [64, 1, 64, 144]);  clone_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:199 in forward, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
        add_146: "f32[64, 1, 64, 144]" = torch.ops.aten.add.Tensor(mul_247, view_165);  mul_247 = view_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:201 in forward, code: attn = attn.softmax(dim=-1)
        amax_5: "f32[64, 1, 64, 1]" = torch.ops.aten.amax.default(add_146, [-1], True)
        sub_65: "f32[64, 1, 64, 144]" = torch.ops.aten.sub.Tensor(add_146, amax_5);  add_146 = amax_5 = None
        exp_5: "f32[64, 1, 64, 144]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
        sum_6: "f32[64, 1, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[64, 1, 64, 144]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:203 in forward, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        expand_44: "f32[64, 1, 64, 144]" = torch.ops.aten.expand.default(div_5, [64, 1, 64, 144]);  div_5 = None
        view_166: "f32[64, 64, 144]" = torch.ops.aten.reshape.default(expand_44, [64, 64, 144]);  expand_44 = None
        expand_45: "f32[64, 1, 144, 64]" = torch.ops.aten.expand.default(getitem_15, [64, 1, 144, 64]);  getitem_15 = None
        view_167: "f32[64, 144, 64]" = torch.ops.aten.reshape.default(expand_45, [64, 144, 64]);  expand_45 = None
        bmm_11: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
        view_168: "f32[64, 1, 64, 64]" = torch.ops.aten.reshape.default(bmm_11, [64, 1, 64, 64]);  bmm_11 = None
        permute_65: "f32[64, 64, 64, 1]" = torch.ops.aten.permute.default(view_168, [0, 3, 2, 1]);  view_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:205 in forward, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        clone_48: "f32[64, 64, 64, 1]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_169: "f32[4096, 8, 8, 1, 1]" = torch.ops.aten.reshape.default(clone_48, [4096, 8, 8, 1, 1]);  clone_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/halo_attn.py:206 in forward, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
        permute_66: "f32[4096, 1, 8, 1, 8]" = torch.ops.aten.permute.default(view_169, [0, 3, 1, 4, 2]);  view_169 = None
        view_170: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(permute_66, [8, 512, 8, 8]);  permute_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_480: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_481: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        sub_66: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_170, unsqueeze_481);  view_170 = unsqueeze_481 = None
        add_147: "f32[512]" = torch.ops.aten.add.Tensor(arg162_1, 1e-05);  arg162_1 = None
        sqrt_60: "f32[512]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_60: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_248: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_482: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_248, -1);  mul_248 = None
        unsqueeze_483: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        mul_249: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_483);  sub_66 = unsqueeze_483 = None
        unsqueeze_484: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_485: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_250: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_249, unsqueeze_485);  mul_249 = unsqueeze_485 = None
        unsqueeze_486: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_487: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_148: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_250, unsqueeze_487);  mul_250 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        sigmoid_62: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_148)
        mul_251: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_148, sigmoid_62);  add_148 = sigmoid_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_77: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(mul_251, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_251 = arg165_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_488: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_489: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        sub_67: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_489);  convolution_77 = unsqueeze_489 = None
        add_149: "f32[2048]" = torch.ops.aten.add.Tensor(arg167_1, 1e-05);  arg167_1 = None
        sqrt_61: "f32[2048]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_61: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_252: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_490: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
        unsqueeze_491: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        mul_253: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_491);  sub_67 = unsqueeze_491 = None
        unsqueeze_492: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_493: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_254: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_493);  mul_253 = unsqueeze_493 = None
        unsqueeze_494: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_495: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_150: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_495);  mul_254 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_151: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_150, mul_242);  add_150 = mul_242 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        sigmoid_63: "f32[8, 2048, 8, 8]" = torch.ops.aten.sigmoid.default(add_151)
        mul_255: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(add_151, sigmoid_63);  add_151 = sigmoid_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_11: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(mul_255, [-1, -2], True);  mul_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_171: "f32[8, 2048]" = torch.ops.aten.reshape.default(mean_11, [8, 2048]);  mean_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_67: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg171_1, view_171, permute_67);  arg171_1 = view_171 = permute_67 = None
        return (addmm_1,)
        