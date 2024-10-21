class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[24, 3, 3, 3]", arg1_1: "f32[8, 3, 256, 256]", arg2_1: "f32[24]", arg3_1: "f32[24]", arg4_1: "f32[24]", arg5_1: "f32[24]", arg6_1: "f32[32, 24, 3, 3]", arg7_1: "f32[32]", arg8_1: "f32[32]", arg9_1: "f32[32]", arg10_1: "f32[32]", arg11_1: "f32[64, 32, 3, 3]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[64, 64, 1, 1]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[64, 64, 3, 3]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[256, 64, 1, 1]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[256, 64, 1, 1]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[64, 256, 1, 1]", arg37_1: "f32[64]", arg38_1: "f32[64]", arg39_1: "f32[64]", arg40_1: "f32[64]", arg41_1: "f32[64, 64, 3, 3]", arg42_1: "f32[64]", arg43_1: "f32[64]", arg44_1: "f32[64]", arg45_1: "f32[64]", arg46_1: "f32[256, 64, 1, 1]", arg47_1: "f32[256]", arg48_1: "f32[256]", arg49_1: "f32[256]", arg50_1: "f32[256]", arg51_1: "f32[128, 256, 1, 1]", arg52_1: "f32[128]", arg53_1: "f32[128]", arg54_1: "f32[128]", arg55_1: "f32[128]", arg56_1: "f32[128, 128, 3, 3]", arg57_1: "f32[128]", arg58_1: "f32[128]", arg59_1: "f32[128]", arg60_1: "f32[128]", arg61_1: "f32[512, 128, 1, 1]", arg62_1: "f32[512]", arg63_1: "f32[512]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[512, 256, 1, 1]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[512]", arg71_1: "f32[128, 512, 1, 1]", arg72_1: "f32[128]", arg73_1: "f32[128]", arg74_1: "f32[128]", arg75_1: "f32[128]", arg76_1: "f32[128, 128, 3, 3]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[128]", arg80_1: "f32[128]", arg81_1: "f32[512, 128, 1, 1]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[512]", arg86_1: "f32[256, 512, 1, 1]", arg87_1: "f32[256]", arg88_1: "f32[256]", arg89_1: "f32[256]", arg90_1: "f32[256]", arg91_1: "f32[256, 256, 3, 3]", arg92_1: "f32[256]", arg93_1: "f32[256]", arg94_1: "f32[256]", arg95_1: "f32[256]", arg96_1: "f32[1024, 256, 1, 1]", arg97_1: "f32[1024]", arg98_1: "f32[1024]", arg99_1: "f32[1024]", arg100_1: "f32[1024]", arg101_1: "f32[1024, 512, 1, 1]", arg102_1: "f32[1024]", arg103_1: "f32[1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024]", arg106_1: "f32[256, 1024, 1, 1]", arg107_1: "f32[256]", arg108_1: "f32[256]", arg109_1: "f32[256]", arg110_1: "f32[256]", arg111_1: "f32[768, 256, 1, 1]", arg112_1: "f32[31, 64]", arg113_1: "f32[31, 64]", arg114_1: "f32[256]", arg115_1: "f32[256]", arg116_1: "f32[256]", arg117_1: "f32[256]", arg118_1: "f32[1024, 256, 1, 1]", arg119_1: "f32[1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024]", arg122_1: "f32[1024]", arg123_1: "f32[512, 1024, 1, 1]", arg124_1: "f32[512]", arg125_1: "f32[512]", arg126_1: "f32[512]", arg127_1: "f32[512]", arg128_1: "f32[1536, 512, 1, 1]", arg129_1: "f32[31, 128]", arg130_1: "f32[31, 128]", arg131_1: "f32[512]", arg132_1: "f32[512]", arg133_1: "f32[512]", arg134_1: "f32[512]", arg135_1: "f32[2048, 512, 1, 1]", arg136_1: "f32[2048]", arg137_1: "f32[2048]", arg138_1: "f32[2048]", arg139_1: "f32[2048]", arg140_1: "f32[2048, 1024, 1, 1]", arg141_1: "f32[2048]", arg142_1: "f32[2048]", arg143_1: "f32[2048]", arg144_1: "f32[2048]", arg145_1: "f32[512, 2048, 1, 1]", arg146_1: "f32[512]", arg147_1: "f32[512]", arg148_1: "f32[512]", arg149_1: "f32[512]", arg150_1: "f32[1536, 512, 1, 1]", arg151_1: "f32[15, 128]", arg152_1: "f32[15, 128]", arg153_1: "f32[512]", arg154_1: "f32[512]", arg155_1: "f32[512]", arg156_1: "f32[512]", arg157_1: "f32[2048, 512, 1, 1]", arg158_1: "f32[2048]", arg159_1: "f32[2048]", arg160_1: "f32[2048]", arg161_1: "f32[2048]", arg162_1: "f32[1000, 2048]", arg163_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_31: "f32[8, 24, 128, 128]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_248: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_249: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_34: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  convolution_31 = unsqueeze_249 = None
        add_76: "f32[24]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_31: "f32[24]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
        reciprocal_31: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_96: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
        unsqueeze_251: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_97: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_251);  sub_34 = unsqueeze_251 = None
        unsqueeze_252: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_253: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_98: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_253);  mul_97 = unsqueeze_253 = None
        unsqueeze_254: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_255: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_77: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_255);  mul_98 = unsqueeze_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_27: "f32[8, 24, 128, 128]" = torch.ops.aten.relu.default(add_77);  add_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_32: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(relu_27, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_27 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_256: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_257: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_35: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  convolution_32 = unsqueeze_257 = None
        add_78: "f32[32]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_32: "f32[32]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        reciprocal_32: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_99: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
        unsqueeze_259: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_100: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_259);  sub_35 = unsqueeze_259 = None
        unsqueeze_260: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_261: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_101: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_261);  mul_100 = unsqueeze_261 = None
        unsqueeze_262: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_263: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_79: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_263);  mul_101 = unsqueeze_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_28: "f32[8, 32, 128, 128]" = torch.ops.aten.relu.default(add_79);  add_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_33: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(relu_28, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_28 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_264: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_265: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_36: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  convolution_33 = unsqueeze_265 = None
        add_80: "f32[64]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_33: "f32[64]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_33: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_102: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
        unsqueeze_267: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_103: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_267);  sub_36 = unsqueeze_267 = None
        unsqueeze_268: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_269: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_104: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_269);  mul_103 = unsqueeze_269 = None
        unsqueeze_270: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_271: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_81: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_271);  mul_104 = unsqueeze_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_29: "f32[8, 64, 128, 128]" = torch.ops.aten.relu.default(add_81);  add_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:1421 in forward_features, code: x = self.stem(x)
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_29, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_29 = None
        getitem_11: "f32[8, 64, 64, 64]" = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_34: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_11, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_272: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_273: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_37: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  convolution_34 = unsqueeze_273 = None
        add_82: "f32[64]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_34: "f32[64]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        reciprocal_34: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_105: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
        unsqueeze_275: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_106: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_275);  sub_37 = unsqueeze_275 = None
        unsqueeze_276: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_277: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_107: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_277);  mul_106 = unsqueeze_277 = None
        unsqueeze_278: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_279: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_83: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_279);  mul_107 = unsqueeze_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_30: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_83);  add_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_35: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_30, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_30 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_280: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_281: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_38: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  convolution_35 = unsqueeze_281 = None
        add_84: "f32[64]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_35: "f32[64]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_35: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_108: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
        unsqueeze_283: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_109: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_283);  sub_38 = unsqueeze_283 = None
        unsqueeze_284: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_285: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_110: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_285);  mul_109 = unsqueeze_285 = None
        unsqueeze_286: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_287: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_85: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_287);  mul_110 = unsqueeze_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_31: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_85);  add_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_36: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_31, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_31 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_288: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_289: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_39: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  convolution_36 = unsqueeze_289 = None
        add_86: "f32[256]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_36: "f32[256]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_36: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_111: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
        unsqueeze_291: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_112: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_291);  sub_39 = unsqueeze_291 = None
        unsqueeze_292: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_293: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_113: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_293);  mul_112 = unsqueeze_293 = None
        unsqueeze_294: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_295: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_87: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_295);  mul_113 = unsqueeze_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_37: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem_11, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_11 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_296: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_297: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_40: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  convolution_37 = unsqueeze_297 = None
        add_88: "f32[256]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_37: "f32[256]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
        reciprocal_37: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_114: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
        unsqueeze_299: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_115: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_299);  sub_40 = unsqueeze_299 = None
        unsqueeze_300: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_301: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_116: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_301);  mul_115 = unsqueeze_301 = None
        unsqueeze_302: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_303: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_89: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_303);  mul_116 = unsqueeze_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_90: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_87, add_89);  add_87 = add_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        relu_32: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_90);  add_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_38: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_32, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_304: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_305: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_41: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  convolution_38 = unsqueeze_305 = None
        add_91: "f32[64]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_38: "f32[64]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
        reciprocal_38: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_117: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
        unsqueeze_307: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_118: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_307);  sub_41 = unsqueeze_307 = None
        unsqueeze_308: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_309: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_119: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_309);  mul_118 = unsqueeze_309 = None
        unsqueeze_310: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_311: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_92: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_311);  mul_119 = unsqueeze_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_33: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_92);  add_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_39: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_33, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_33 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_312: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_313: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_42: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  convolution_39 = unsqueeze_313 = None
        add_93: "f32[64]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_39: "f32[64]" = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_39: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_120: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
        unsqueeze_315: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_121: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_315);  sub_42 = unsqueeze_315 = None
        unsqueeze_316: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_317: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_122: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_317);  mul_121 = unsqueeze_317 = None
        unsqueeze_318: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_319: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_94: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_319);  mul_122 = unsqueeze_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_34: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_94);  add_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_40: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_34, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_34 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_43: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  convolution_40 = unsqueeze_321 = None
        add_95: "f32[256]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_40: "f32[256]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_40: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_124: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_323);  sub_43 = unsqueeze_323 = None
        unsqueeze_324: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_325: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_125: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_325);  mul_124 = unsqueeze_325 = None
        unsqueeze_326: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_327: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_96: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_327);  mul_125 = unsqueeze_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_97: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_96, relu_32);  add_96 = relu_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        relu_35: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_97);  add_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_41: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_35, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_328: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_329: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_44: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
        add_98: "f32[128]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_41: "f32[128]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_41: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_126: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
        unsqueeze_331: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_127: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_331);  sub_44 = unsqueeze_331 = None
        unsqueeze_332: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_333: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_128: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_333);  mul_127 = unsqueeze_333 = None
        unsqueeze_334: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_335: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_99: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_335);  mul_128 = unsqueeze_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_36: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_99);  add_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_42: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_36, arg56_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_36 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_336: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_337: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_45: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
        add_100: "f32[128]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_42: "f32[128]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        reciprocal_42: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_129: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
        unsqueeze_339: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_130: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_339);  sub_45 = unsqueeze_339 = None
        unsqueeze_340: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_341: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_131: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_341);  mul_130 = unsqueeze_341 = None
        unsqueeze_342: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_343: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_101: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_343);  mul_131 = unsqueeze_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_37: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_101);  add_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_43: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_37, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_37 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_344: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_345: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_46: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
        add_102: "f32[512]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_43: "f32[512]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_43: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_132: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
        unsqueeze_347: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_133: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_347);  sub_46 = unsqueeze_347 = None
        unsqueeze_348: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_349: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_134: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_349);  mul_133 = unsqueeze_349 = None
        unsqueeze_350: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_351: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_103: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_351);  mul_134 = unsqueeze_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_44: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_35, arg66_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_35 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_352: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_353: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_47: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
        add_104: "f32[512]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_44: "f32[512]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_44: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_135: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_355: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_136: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_355);  sub_47 = unsqueeze_355 = None
        unsqueeze_356: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_357: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_137: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_357);  mul_136 = unsqueeze_357 = None
        unsqueeze_358: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_359: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_105: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_359);  mul_137 = unsqueeze_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_106: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_103, add_105);  add_103 = add_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        relu_38: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_106);  add_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_45: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_38, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_360: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_361: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_48: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
        add_107: "f32[128]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_45: "f32[128]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
        reciprocal_45: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
        unsqueeze_363: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_139: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_363);  sub_48 = unsqueeze_363 = None
        unsqueeze_364: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_365: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_140: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_365);  mul_139 = unsqueeze_365 = None
        unsqueeze_366: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_367: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_108: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_367);  mul_140 = unsqueeze_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_39: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_108);  add_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_46: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_39, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_39 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_368: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_369: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        sub_49: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
        add_109: "f32[128]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_46: "f32[128]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_46: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_141: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_370: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
        unsqueeze_371: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        mul_142: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_371);  sub_49 = unsqueeze_371 = None
        unsqueeze_372: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_373: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_143: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_373);  mul_142 = unsqueeze_373 = None
        unsqueeze_374: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_375: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_110: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_375);  mul_143 = unsqueeze_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_40: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_110);  add_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_47: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_40, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_40 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_376: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_377: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        sub_50: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
        add_111: "f32[512]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_47: "f32[512]" = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_47: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_144: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_378: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
        unsqueeze_379: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        mul_145: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_379);  sub_50 = unsqueeze_379 = None
        unsqueeze_380: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_381: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_146: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_381);  mul_145 = unsqueeze_381 = None
        unsqueeze_382: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_383: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_112: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_383);  mul_146 = unsqueeze_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_113: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_112, relu_38);  add_112 = relu_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        relu_41: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_113);  add_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_48: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_41, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_384: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_385: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        sub_51: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
        add_114: "f32[256]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_48: "f32[256]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_48: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_147: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_386: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_387: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        mul_148: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_387);  sub_51 = unsqueeze_387 = None
        unsqueeze_388: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_389: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_149: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_389);  mul_148 = unsqueeze_389 = None
        unsqueeze_390: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_391: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_115: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_391);  mul_149 = unsqueeze_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_42: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_115);  add_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_49: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_42, arg91_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_42 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_392: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_393: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        sub_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
        add_116: "f32[256]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_49: "f32[256]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_49: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_150: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_394: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
        unsqueeze_395: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        mul_151: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_395);  sub_52 = unsqueeze_395 = None
        unsqueeze_396: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_397: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_152: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_397);  mul_151 = unsqueeze_397 = None
        unsqueeze_398: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_399: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_117: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_399);  mul_152 = unsqueeze_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_43: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_117);  add_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_50: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(relu_43, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_43 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_400: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_401: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        sub_53: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
        add_118: "f32[1024]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_50: "f32[1024]" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_50: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_153: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_402: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
        unsqueeze_403: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        mul_154: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_403);  sub_53 = unsqueeze_403 = None
        unsqueeze_404: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_405: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_155: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_405);  mul_154 = unsqueeze_405 = None
        unsqueeze_406: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_407: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_119: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_407);  mul_155 = unsqueeze_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_51: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(relu_41, arg101_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_41 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_408: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_409: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        sub_54: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
        add_120: "f32[1024]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_51: "f32[1024]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_51: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_156: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_410: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
        unsqueeze_411: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        mul_157: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_411);  sub_54 = unsqueeze_411 = None
        unsqueeze_412: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_413: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_158: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_413);  mul_157 = unsqueeze_413 = None
        unsqueeze_414: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_415: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_121: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_415);  mul_158 = unsqueeze_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:348 in forward, code: x = x + self.shortcut(shortcut)
        add_122: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_119, add_121);  add_119 = add_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:349 in forward, code: return self.act(x)
        relu_44: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_122);  add_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_52: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_44, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_416: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_417: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        sub_55: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
        add_123: "f32[256]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_52: "f32[256]" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_52: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_159: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_418: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_419: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        mul_160: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_419);  sub_55 = unsqueeze_419 = None
        unsqueeze_420: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_421: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_161: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_421);  mul_160 = unsqueeze_421 = None
        unsqueeze_422: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_423: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_124: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_423);  mul_161 = unsqueeze_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_45: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_124);  add_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:140 in forward, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_53: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(relu_45, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_45 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:144 in forward, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(convolution_53, [256, 256, 256], 1);  convolution_53 = None
        getitem_13: "f32[8, 256, 16, 16]" = split_with_sizes_3[0]
        getitem_14: "f32[8, 256, 16, 16]" = split_with_sizes_3[1]
        getitem_15: "f32[8, 256, 16, 16]" = split_with_sizes_3[2];  split_with_sizes_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:145 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        clone_22: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_13, memory_format = torch.contiguous_format);  getitem_13 = None
        view_73: "f32[32, 64, 256]" = torch.ops.aten.reshape.default(clone_22, [32, 64, 256]);  clone_22 = None
        permute_25: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_18: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_25, [32, 256, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:146 in forward, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        clone_23: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_14, memory_format = torch.contiguous_format);  getitem_14 = None
        view_74: "f32[32, 64, 256]" = torch.ops.aten.reshape.default(clone_23, [32, 64, 256]);  clone_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_19: "f32[32, 64, 256]" = torch.ops.aten.expand.default(view_74, [32, 64, 256]);  view_74 = None
        bmm_6: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(expand_18, expand_19);  expand_18 = expand_19 = None
        mul_162: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_6, 0.125);  bmm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:72 in forward, code: q = q.reshape(B, self.height, self.width, -1)
        view_79: "f32[32, 16, 16, 64]" = torch.ops.aten.reshape.default(permute_25, [32, 16, 16, 64]);  permute_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:76 in forward, code: q = q.transpose(1, 2)
        permute_29: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_26: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_86: "f32[8192, 64]" = torch.ops.aten.reshape.default(clone_26, [8192, 64]);  clone_26 = None
        permute_30: "f32[64, 31]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_7: "f32[8192, 31]" = torch.ops.aten.mm.default(view_86, permute_30);  view_86 = permute_30 = None
        view_87: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_7, [32, 16, 16, 31]);  mm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_88: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_87, [-1, 16, 31]);  view_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_14: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_88, [0, 1], 0.0);  view_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_89: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_14, [512, 512]);  constant_pad_nd_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_15: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_89, [0, 15], 0.0);  view_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_90: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_15, [-1, 17, 31]);  constant_pad_nd_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_23: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_90, 1, 0, 16);  view_90 = None
        slice_24: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_23, 2, 15, 9223372036854775807);  slice_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_91: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_24, [32, 16, 1, 16, 16]);  slice_24 = None
        expand_21: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_91, [-1, -1, 16, -1, -1]);  view_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_31: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_21, [0, 3, 1, 4, 2]);  expand_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_25: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(view_79, memory_format = torch.contiguous_format);  view_79 = None
        view_80: "f32[8192, 64]" = torch.ops.aten.reshape.default(clone_25, [8192, 64]);  clone_25 = None
        permute_27: "f32[64, 31]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_6: "f32[8192, 31]" = torch.ops.aten.mm.default(view_80, permute_27);  view_80 = permute_27 = None
        view_81: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_6, [32, 16, 16, 31]);  mm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_82: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_81, [-1, 16, 31]);  view_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_12: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_82, [0, 1], 0.0);  view_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_83: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_12, [512, 512]);  constant_pad_nd_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_13: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_83, [0, 15], 0.0);  view_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_84: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_13, [-1, 17, 31]);  constant_pad_nd_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_20: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_84, 1, 0, 16);  view_84 = None
        slice_21: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_20, 2, 15, 9223372036854775807);  slice_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_85: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_21, [32, 16, 1, 16, 16]);  slice_21 = None
        expand_20: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_85, [-1, -1, 16, -1, -1]);  view_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_28: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_20, [0, 1, 3, 2, 4]);  expand_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:79 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_125: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_31, permute_28);  permute_31 = permute_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:80 in forward, code: rel_logits = rel_logits.reshape(B, HW, HW)
        clone_27: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format);  add_125 = None
        view_92: "f32[32, 256, 256]" = torch.ops.aten.reshape.default(clone_27, [32, 256, 256]);  clone_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_126: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_162, view_92);  mul_162 = view_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:153 in forward, code: attn = attn.softmax(dim=-1)
        amax_3: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_126, [-1], True)
        sub_56: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_126, amax_3);  add_126 = amax_3 = None
        exp_3: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
        sum_4: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_22: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_3, [32, 256, 256]);  div_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:147 in forward, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        clone_24: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format);  getitem_15 = None
        view_75: "f32[32, 64, 256]" = torch.ops.aten.reshape.default(clone_24, [32, 64, 256]);  clone_24 = None
        permute_26: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_23: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_26, [32, 256, 64]);  permute_26 = None
        bmm_7: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(expand_22, expand_23);  expand_22 = expand_23 = None
        permute_32: "f32[32, 64, 256]" = torch.ops.aten.permute.default(bmm_7, [0, 2, 1]);  bmm_7 = None
        clone_28: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_96: "f32[8, 256, 16, 16]" = torch.ops.aten.reshape.default(clone_28, [8, 256, 16, 16]);  clone_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_424: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_425: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        sub_57: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_425);  view_96 = unsqueeze_425 = None
        add_127: "f32[256]" = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_53: "f32[256]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_53: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_163: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_426: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_163, -1);  mul_163 = None
        unsqueeze_427: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        mul_164: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_427);  sub_57 = unsqueeze_427 = None
        unsqueeze_428: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_429: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_165: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_429);  mul_164 = unsqueeze_429 = None
        unsqueeze_430: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_431: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_128: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_165, unsqueeze_431);  mul_165 = unsqueeze_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_46: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_128);  add_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_54: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(relu_46, arg118_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_46 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_432: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_433: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        sub_58: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
        add_129: "f32[1024]" = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_54: "f32[1024]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_54: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_166: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_434: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_166, -1);  mul_166 = None
        unsqueeze_435: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        mul_167: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_435);  sub_58 = unsqueeze_435 = None
        unsqueeze_436: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_437: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_168: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_437);  mul_167 = unsqueeze_437 = None
        unsqueeze_438: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_439: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_130: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_168, unsqueeze_439);  mul_168 = unsqueeze_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_131: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_130, relu_44);  add_130 = relu_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        relu_47: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_131);  add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_55: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_47, arg123_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_440: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_441: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        sub_59: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
        add_132: "f32[512]" = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_55: "f32[512]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_55: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_169: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_442: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_169, -1);  mul_169 = None
        unsqueeze_443: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        mul_170: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_443);  sub_59 = unsqueeze_443 = None
        unsqueeze_444: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_445: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_171: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_445);  mul_170 = unsqueeze_445 = None
        unsqueeze_446: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_447: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_133: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_171, unsqueeze_447);  mul_171 = unsqueeze_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_48: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_133);  add_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:140 in forward, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_56: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(relu_48, arg128_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_48 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:144 in forward, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(convolution_56, [512, 512, 512], 1);  convolution_56 = None
        getitem_16: "f32[8, 512, 16, 16]" = split_with_sizes_4[0]
        getitem_17: "f32[8, 512, 16, 16]" = split_with_sizes_4[1]
        getitem_18: "f32[8, 512, 16, 16]" = split_with_sizes_4[2];  split_with_sizes_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:145 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        clone_29: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_16, memory_format = torch.contiguous_format);  getitem_16 = None
        view_97: "f32[32, 128, 256]" = torch.ops.aten.reshape.default(clone_29, [32, 128, 256]);  clone_29 = None
        permute_33: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_24: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_33, [32, 256, 128])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:146 in forward, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        clone_30: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_17, memory_format = torch.contiguous_format);  getitem_17 = None
        view_98: "f32[32, 128, 256]" = torch.ops.aten.reshape.default(clone_30, [32, 128, 256]);  clone_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_25: "f32[32, 128, 256]" = torch.ops.aten.expand.default(view_98, [32, 128, 256]);  view_98 = None
        bmm_8: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(expand_24, expand_25);  expand_24 = expand_25 = None
        mul_172: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_8, 0.08838834764831845);  bmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:72 in forward, code: q = q.reshape(B, self.height, self.width, -1)
        view_103: "f32[32, 16, 16, 128]" = torch.ops.aten.reshape.default(permute_33, [32, 16, 16, 128]);  permute_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:76 in forward, code: q = q.transpose(1, 2)
        permute_37: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_33: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_110: "f32[8192, 128]" = torch.ops.aten.reshape.default(clone_33, [8192, 128]);  clone_33 = None
        permute_38: "f32[128, 31]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_9: "f32[8192, 31]" = torch.ops.aten.mm.default(view_110, permute_38);  view_110 = permute_38 = None
        view_111: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_9, [32, 16, 16, 31]);  mm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_112: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_111, [-1, 16, 31]);  view_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_18: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_112, [0, 1], 0.0);  view_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_113: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_18, [512, 512]);  constant_pad_nd_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_19: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_113, [0, 15], 0.0);  view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_114: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_19, [-1, 17, 31]);  constant_pad_nd_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_29: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_114, 1, 0, 16);  view_114 = None
        slice_30: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_29, 2, 15, 9223372036854775807);  slice_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_115: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_30, [32, 16, 1, 16, 16]);  slice_30 = None
        expand_27: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_115, [-1, -1, 16, -1, -1]);  view_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_39: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_27, [0, 3, 1, 4, 2]);  expand_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_32: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(view_103, memory_format = torch.contiguous_format);  view_103 = None
        view_104: "f32[8192, 128]" = torch.ops.aten.reshape.default(clone_32, [8192, 128]);  clone_32 = None
        permute_35: "f32[128, 31]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_8: "f32[8192, 31]" = torch.ops.aten.mm.default(view_104, permute_35);  view_104 = permute_35 = None
        view_105: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_8, [32, 16, 16, 31]);  mm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_106: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_105, [-1, 16, 31]);  view_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_16: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_106, [0, 1], 0.0);  view_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_107: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_16, [512, 512]);  constant_pad_nd_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_17: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_107, [0, 15], 0.0);  view_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_108: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_17, [-1, 17, 31]);  constant_pad_nd_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_26: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_108, 1, 0, 16);  view_108 = None
        slice_27: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_26, 2, 15, 9223372036854775807);  slice_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_109: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_27, [32, 16, 1, 16, 16]);  slice_27 = None
        expand_26: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_109, [-1, -1, 16, -1, -1]);  view_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_36: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_26, [0, 1, 3, 2, 4]);  expand_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:79 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_134: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_39, permute_36);  permute_39 = permute_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:80 in forward, code: rel_logits = rel_logits.reshape(B, HW, HW)
        clone_34: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format);  add_134 = None
        view_116: "f32[32, 256, 256]" = torch.ops.aten.reshape.default(clone_34, [32, 256, 256]);  clone_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_135: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_172, view_116);  mul_172 = view_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:153 in forward, code: attn = attn.softmax(dim=-1)
        amax_4: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_135, [-1], True)
        sub_60: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_135, amax_4);  add_135 = amax_4 = None
        exp_4: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_5: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_28: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_4, [32, 256, 256]);  div_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:147 in forward, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        clone_31: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_18, memory_format = torch.contiguous_format);  getitem_18 = None
        view_99: "f32[32, 128, 256]" = torch.ops.aten.reshape.default(clone_31, [32, 128, 256]);  clone_31 = None
        permute_34: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_29: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_34, [32, 256, 128]);  permute_34 = None
        bmm_9: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(expand_28, expand_29);  expand_28 = expand_29 = None
        permute_40: "f32[32, 128, 256]" = torch.ops.aten.permute.default(bmm_9, [0, 2, 1]);  bmm_9 = None
        clone_35: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_120: "f32[8, 512, 16, 16]" = torch.ops.aten.reshape.default(clone_35, [8, 512, 16, 16]);  clone_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:156 in forward, code: out = self.pool(out)
        avg_pool2d_1: "f32[8, 512, 8, 8]" = torch.ops.aten.avg_pool2d.default(view_120, [2, 2], [2, 2]);  view_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_448: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_449: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        sub_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d_1, unsqueeze_449);  avg_pool2d_1 = unsqueeze_449 = None
        add_136: "f32[512]" = torch.ops.aten.add.Tensor(arg132_1, 1e-05);  arg132_1 = None
        sqrt_56: "f32[512]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_56: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_173: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_450: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_173, -1);  mul_173 = None
        unsqueeze_451: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        mul_174: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_451);  sub_61 = unsqueeze_451 = None
        unsqueeze_452: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_453: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_175: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_453);  mul_174 = unsqueeze_453 = None
        unsqueeze_454: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_455: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_137: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_455);  mul_175 = unsqueeze_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_49: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_137);  add_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_57: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(relu_49, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_49 = arg135_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_456: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_457: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        sub_62: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_457);  convolution_57 = unsqueeze_457 = None
        add_138: "f32[2048]" = torch.ops.aten.add.Tensor(arg137_1, 1e-05);  arg137_1 = None
        sqrt_57: "f32[2048]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_57: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_176: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_458: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_176, -1);  mul_176 = None
        unsqueeze_459: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        mul_177: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_459);  sub_62 = unsqueeze_459 = None
        unsqueeze_460: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_461: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_178: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_177, unsqueeze_461);  mul_177 = unsqueeze_461 = None
        unsqueeze_462: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_463: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_139: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_178, unsqueeze_463);  mul_178 = unsqueeze_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_58: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(relu_47, arg140_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_47 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_464: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_465: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        sub_63: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_465);  convolution_58 = unsqueeze_465 = None
        add_140: "f32[2048]" = torch.ops.aten.add.Tensor(arg142_1, 1e-05);  arg142_1 = None
        sqrt_58: "f32[2048]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_58: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_179: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_466: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
        unsqueeze_467: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        mul_180: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_467);  sub_63 = unsqueeze_467 = None
        unsqueeze_468: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_469: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_181: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_469);  mul_180 = unsqueeze_469 = None
        unsqueeze_470: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_471: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_141: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_471);  mul_181 = unsqueeze_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_142: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_139, add_141);  add_139 = add_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        relu_50: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_142);  add_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_59: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(relu_50, arg145_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_472: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_473: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        sub_64: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_473);  convolution_59 = unsqueeze_473 = None
        add_143: "f32[512]" = torch.ops.aten.add.Tensor(arg147_1, 1e-05);  arg147_1 = None
        sqrt_59: "f32[512]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_59: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_182: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_474: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
        unsqueeze_475: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        mul_183: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_475);  sub_64 = unsqueeze_475 = None
        unsqueeze_476: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_477: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_184: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_477);  mul_183 = unsqueeze_477 = None
        unsqueeze_478: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_479: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_144: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_479);  mul_184 = unsqueeze_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_51: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_144);  add_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:140 in forward, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
        convolution_60: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(relu_51, arg150_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_51 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:144 in forward, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(convolution_60, [512, 512, 512], 1);  convolution_60 = None
        getitem_19: "f32[8, 512, 8, 8]" = split_with_sizes_5[0]
        getitem_20: "f32[8, 512, 8, 8]" = split_with_sizes_5[1]
        getitem_21: "f32[8, 512, 8, 8]" = split_with_sizes_5[2];  split_with_sizes_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:145 in forward, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        clone_36: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_19, memory_format = torch.contiguous_format);  getitem_19 = None
        view_121: "f32[32, 128, 64]" = torch.ops.aten.reshape.default(clone_36, [32, 128, 64]);  clone_36 = None
        permute_41: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_30: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_41, [32, 64, 128])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:146 in forward, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        clone_37: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_20, memory_format = torch.contiguous_format);  getitem_20 = None
        view_122: "f32[32, 128, 64]" = torch.ops.aten.reshape.default(clone_37, [32, 128, 64]);  clone_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        expand_31: "f32[32, 128, 64]" = torch.ops.aten.expand.default(view_122, [32, 128, 64]);  view_122 = None
        bmm_10: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(expand_30, expand_31);  expand_30 = expand_31 = None
        mul_185: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(bmm_10, 0.08838834764831845);  bmm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:72 in forward, code: q = q.reshape(B, self.height, self.width, -1)
        view_127: "f32[32, 8, 8, 128]" = torch.ops.aten.reshape.default(permute_41, [32, 8, 8, 128]);  permute_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:76 in forward, code: q = q.transpose(1, 2)
        permute_45: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3])
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_40: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_134: "f32[2048, 128]" = torch.ops.aten.reshape.default(clone_40, [2048, 128]);  clone_40 = None
        permute_46: "f32[128, 15]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        mm_11: "f32[2048, 15]" = torch.ops.aten.mm.default(view_134, permute_46);  view_134 = permute_46 = None
        view_135: "f32[32, 8, 8, 15]" = torch.ops.aten.reshape.default(mm_11, [32, 8, 8, 15]);  mm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_136: "f32[256, 8, 15]" = torch.ops.aten.reshape.default(view_135, [-1, 8, 15]);  view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_22: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_136, [0, 1], 0.0);  view_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_137: "f32[256, 128]" = torch.ops.aten.reshape.default(constant_pad_nd_22, [256, 128]);  constant_pad_nd_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_23: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_137, [0, 7], 0.0);  view_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_138: "f32[256, 9, 15]" = torch.ops.aten.reshape.default(constant_pad_nd_23, [-1, 9, 15]);  constant_pad_nd_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_35: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(view_138, 1, 0, 8);  view_138 = None
        slice_36: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_35, 2, 7, 9223372036854775807);  slice_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_139: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.reshape.default(slice_36, [32, 8, 1, 8, 8]);  slice_36 = None
        expand_33: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_139, [-1, -1, 8, -1, -1]);  view_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_47: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_33, [0, 3, 1, 4, 2]);  expand_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:40 in rel_logits_1d, code: x = (q @ rel_k.transpose(-1, -2))
        clone_39: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(view_127, memory_format = torch.contiguous_format);  view_127 = None
        view_128: "f32[2048, 128]" = torch.ops.aten.reshape.default(clone_39, [2048, 128]);  clone_39 = None
        permute_43: "f32[128, 15]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_10: "f32[2048, 15]" = torch.ops.aten.mm.default(view_128, permute_43);  view_128 = permute_43 = None
        view_129: "f32[32, 8, 8, 15]" = torch.ops.aten.reshape.default(mm_10, [32, 8, 8, 15]);  mm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:41 in rel_logits_1d, code: x = x.reshape(-1, W, 2 * W -1)
        view_130: "f32[256, 8, 15]" = torch.ops.aten.reshape.default(view_129, [-1, 8, 15]);  view_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_20: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_130, [0, 1], 0.0);  view_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:44 in rel_logits_1d, code: x_pad = F.pad(x, [0, 1]).flatten(1)
        view_131: "f32[256, 128]" = torch.ops.aten.reshape.default(constant_pad_nd_20, [256, 128]);  constant_pad_nd_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/torch/nn/functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_21: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_131, [0, 7], 0.0);  view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:48 in rel_logits_1d, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
        view_132: "f32[256, 9, 15]" = torch.ops.aten.reshape.default(constant_pad_nd_21, [-1, 9, 15]);  constant_pad_nd_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:49 in rel_logits_1d, code: x = x_pad[:, :W, W - 1:]
        slice_32: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(view_132, 1, 0, 8);  view_132 = None
        slice_33: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_32, 2, 7, 9223372036854775807);  slice_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:52 in rel_logits_1d, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
        view_133: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.reshape.default(slice_33, [32, 8, 1, 8, 8]);  slice_33 = None
        expand_32: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_133, [-1, -1, 8, -1, -1]);  view_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:53 in rel_logits_1d, code: return x.permute(permute_mask)
        permute_44: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_32, [0, 1, 3, 2, 4]);  expand_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:79 in forward, code: rel_logits = rel_logits_h + rel_logits_w
        add_145: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.add.Tensor(permute_47, permute_44);  permute_47 = permute_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:80 in forward, code: rel_logits = rel_logits.reshape(B, HW, HW)
        clone_41: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.clone.default(add_145, memory_format = torch.contiguous_format);  add_145 = None
        view_140: "f32[32, 64, 64]" = torch.ops.aten.reshape.default(clone_41, [32, 64, 64]);  clone_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:152 in forward, code: attn = (q @ k) * self.scale + self.pos_embed(q)
        add_146: "f32[32, 64, 64]" = torch.ops.aten.add.Tensor(mul_185, view_140);  mul_185 = view_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:153 in forward, code: attn = attn.softmax(dim=-1)
        amax_5: "f32[32, 64, 1]" = torch.ops.aten.amax.default(add_146, [-1], True)
        sub_65: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(add_146, amax_5);  add_146 = amax_5 = None
        exp_5: "f32[32, 64, 64]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
        sum_6: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 64, 64]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_34: "f32[32, 64, 64]" = torch.ops.aten.expand.default(div_5, [32, 64, 64]);  div_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:147 in forward, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        clone_38: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_21, memory_format = torch.contiguous_format);  getitem_21 = None
        view_123: "f32[32, 128, 64]" = torch.ops.aten.reshape.default(clone_38, [32, 128, 64]);  clone_38 = None
        permute_42: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/bottleneck_attn.py:155 in forward, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        expand_35: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_42, [32, 64, 128]);  permute_42 = None
        bmm_11: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(expand_34, expand_35);  expand_34 = expand_35 = None
        permute_48: "f32[32, 128, 64]" = torch.ops.aten.permute.default(bmm_11, [0, 2, 1]);  bmm_11 = None
        clone_42: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_144: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(clone_42, [8, 512, 8, 8]);  clone_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_480: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_481: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        sub_66: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_481);  view_144 = unsqueeze_481 = None
        add_147: "f32[512]" = torch.ops.aten.add.Tensor(arg154_1, 1e-05);  arg154_1 = None
        sqrt_60: "f32[512]" = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_60: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_186: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_482: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_483: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        mul_187: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_483);  sub_66 = unsqueeze_483 = None
        unsqueeze_484: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_485: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_188: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_485);  mul_187 = unsqueeze_485 = None
        unsqueeze_486: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_487: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_148: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_487);  mul_188 = unsqueeze_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        relu_52: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_148);  add_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_61: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(relu_52, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_52 = arg157_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        unsqueeze_488: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_489: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        sub_67: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_489);  convolution_61 = unsqueeze_489 = None
        add_149: "f32[2048]" = torch.ops.aten.add.Tensor(arg159_1, 1e-05);  arg159_1 = None
        sqrt_61: "f32[2048]" = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_61: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_189: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_490: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_491: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        mul_190: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_491);  sub_67 = unsqueeze_491 = None
        unsqueeze_492: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_493: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_191: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_493);  mul_190 = unsqueeze_493 = None
        unsqueeze_494: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_495: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_150: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_495);  mul_191 = unsqueeze_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:899 in forward, code: x = x + self.shortcut(shortcut)
        add_151: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_150, relu_50);  add_150 = relu_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/byobnet.py:900 in forward, code: return self.act(x)
        relu_53: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_151);  add_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_53, [-1, -2], True);  relu_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_145: "f32[8, 2048]" = torch.ops.aten.reshape.default(mean_1, [8, 2048]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_49: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg163_1, view_145, permute_49);  arg163_1 = view_145 = permute_49 = None
        return (addmm_1,)
        