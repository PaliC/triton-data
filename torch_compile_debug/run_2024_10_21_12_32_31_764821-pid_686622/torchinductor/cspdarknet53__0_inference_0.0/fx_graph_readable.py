class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 3, 3]", arg1_1: "f32[8, 3, 256, 256]", arg2_1: "f32[32]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "f32[32]", arg6_1: "f32[64, 32, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[128, 64, 1, 1]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[32, 64, 1, 1]", arg17_1: "f32[32]", arg18_1: "f32[32]", arg19_1: "f32[32]", arg20_1: "f32[32]", arg21_1: "f32[64, 32, 3, 3]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[64, 64, 1, 1]", arg27_1: "f32[64]", arg28_1: "f32[64]", arg29_1: "f32[64]", arg30_1: "f32[64]", arg31_1: "f32[64, 128, 1, 1]", arg32_1: "f32[64]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[128, 64, 3, 3]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[128, 128, 1, 1]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[128]", arg46_1: "f32[64, 64, 1, 1]", arg47_1: "f32[64]", arg48_1: "f32[64]", arg49_1: "f32[64]", arg50_1: "f32[64]", arg51_1: "f32[64, 64, 3, 3]", arg52_1: "f32[64]", arg53_1: "f32[64]", arg54_1: "f32[64]", arg55_1: "f32[64]", arg56_1: "f32[64, 64, 1, 1]", arg57_1: "f32[64]", arg58_1: "f32[64]", arg59_1: "f32[64]", arg60_1: "f32[64]", arg61_1: "f32[64, 64, 3, 3]", arg62_1: "f32[64]", arg63_1: "f32[64]", arg64_1: "f32[64]", arg65_1: "f32[64]", arg66_1: "f32[64, 64, 1, 1]", arg67_1: "f32[64]", arg68_1: "f32[64]", arg69_1: "f32[64]", arg70_1: "f32[64]", arg71_1: "f32[128, 128, 1, 1]", arg72_1: "f32[128]", arg73_1: "f32[128]", arg74_1: "f32[128]", arg75_1: "f32[128]", arg76_1: "f32[256, 128, 3, 3]", arg77_1: "f32[256]", arg78_1: "f32[256]", arg79_1: "f32[256]", arg80_1: "f32[256]", arg81_1: "f32[256, 256, 1, 1]", arg82_1: "f32[256]", arg83_1: "f32[256]", arg84_1: "f32[256]", arg85_1: "f32[256]", arg86_1: "f32[128, 128, 1, 1]", arg87_1: "f32[128]", arg88_1: "f32[128]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[128, 128, 3, 3]", arg92_1: "f32[128]", arg93_1: "f32[128]", arg94_1: "f32[128]", arg95_1: "f32[128]", arg96_1: "f32[128, 128, 1, 1]", arg97_1: "f32[128]", arg98_1: "f32[128]", arg99_1: "f32[128]", arg100_1: "f32[128]", arg101_1: "f32[128, 128, 3, 3]", arg102_1: "f32[128]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128]", arg106_1: "f32[128, 128, 1, 1]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "f32[128]", arg110_1: "f32[128]", arg111_1: "f32[128, 128, 3, 3]", arg112_1: "f32[128]", arg113_1: "f32[128]", arg114_1: "f32[128]", arg115_1: "f32[128]", arg116_1: "f32[128, 128, 1, 1]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[128]", arg120_1: "f32[128]", arg121_1: "f32[128, 128, 3, 3]", arg122_1: "f32[128]", arg123_1: "f32[128]", arg124_1: "f32[128]", arg125_1: "f32[128]", arg126_1: "f32[128, 128, 1, 1]", arg127_1: "f32[128]", arg128_1: "f32[128]", arg129_1: "f32[128]", arg130_1: "f32[128]", arg131_1: "f32[128, 128, 3, 3]", arg132_1: "f32[128]", arg133_1: "f32[128]", arg134_1: "f32[128]", arg135_1: "f32[128]", arg136_1: "f32[128, 128, 1, 1]", arg137_1: "f32[128]", arg138_1: "f32[128]", arg139_1: "f32[128]", arg140_1: "f32[128]", arg141_1: "f32[128, 128, 3, 3]", arg142_1: "f32[128]", arg143_1: "f32[128]", arg144_1: "f32[128]", arg145_1: "f32[128]", arg146_1: "f32[128, 128, 1, 1]", arg147_1: "f32[128]", arg148_1: "f32[128]", arg149_1: "f32[128]", arg150_1: "f32[128]", arg151_1: "f32[128, 128, 3, 3]", arg152_1: "f32[128]", arg153_1: "f32[128]", arg154_1: "f32[128]", arg155_1: "f32[128]", arg156_1: "f32[128, 128, 1, 1]", arg157_1: "f32[128]", arg158_1: "f32[128]", arg159_1: "f32[128]", arg160_1: "f32[128]", arg161_1: "f32[128, 128, 3, 3]", arg162_1: "f32[128]", arg163_1: "f32[128]", arg164_1: "f32[128]", arg165_1: "f32[128]", arg166_1: "f32[128, 128, 1, 1]", arg167_1: "f32[128]", arg168_1: "f32[128]", arg169_1: "f32[128]", arg170_1: "f32[128]", arg171_1: "f32[256, 256, 1, 1]", arg172_1: "f32[256]", arg173_1: "f32[256]", arg174_1: "f32[256]", arg175_1: "f32[256]", arg176_1: "f32[512, 256, 3, 3]", arg177_1: "f32[512]", arg178_1: "f32[512]", arg179_1: "f32[512]", arg180_1: "f32[512]", arg181_1: "f32[512, 512, 1, 1]", arg182_1: "f32[512]", arg183_1: "f32[512]", arg184_1: "f32[512]", arg185_1: "f32[512]", arg186_1: "f32[256, 256, 1, 1]", arg187_1: "f32[256]", arg188_1: "f32[256]", arg189_1: "f32[256]", arg190_1: "f32[256]", arg191_1: "f32[256, 256, 3, 3]", arg192_1: "f32[256]", arg193_1: "f32[256]", arg194_1: "f32[256]", arg195_1: "f32[256]", arg196_1: "f32[256, 256, 1, 1]", arg197_1: "f32[256]", arg198_1: "f32[256]", arg199_1: "f32[256]", arg200_1: "f32[256]", arg201_1: "f32[256, 256, 3, 3]", arg202_1: "f32[256]", arg203_1: "f32[256]", arg204_1: "f32[256]", arg205_1: "f32[256]", arg206_1: "f32[256, 256, 1, 1]", arg207_1: "f32[256]", arg208_1: "f32[256]", arg209_1: "f32[256]", arg210_1: "f32[256]", arg211_1: "f32[256, 256, 3, 3]", arg212_1: "f32[256]", arg213_1: "f32[256]", arg214_1: "f32[256]", arg215_1: "f32[256]", arg216_1: "f32[256, 256, 1, 1]", arg217_1: "f32[256]", arg218_1: "f32[256]", arg219_1: "f32[256]", arg220_1: "f32[256]", arg221_1: "f32[256, 256, 3, 3]", arg222_1: "f32[256]", arg223_1: "f32[256]", arg224_1: "f32[256]", arg225_1: "f32[256]", arg226_1: "f32[256, 256, 1, 1]", arg227_1: "f32[256]", arg228_1: "f32[256]", arg229_1: "f32[256]", arg230_1: "f32[256]", arg231_1: "f32[256, 256, 3, 3]", arg232_1: "f32[256]", arg233_1: "f32[256]", arg234_1: "f32[256]", arg235_1: "f32[256]", arg236_1: "f32[256, 256, 1, 1]", arg237_1: "f32[256]", arg238_1: "f32[256]", arg239_1: "f32[256]", arg240_1: "f32[256]", arg241_1: "f32[256, 256, 3, 3]", arg242_1: "f32[256]", arg243_1: "f32[256]", arg244_1: "f32[256]", arg245_1: "f32[256]", arg246_1: "f32[256, 256, 1, 1]", arg247_1: "f32[256]", arg248_1: "f32[256]", arg249_1: "f32[256]", arg250_1: "f32[256]", arg251_1: "f32[256, 256, 3, 3]", arg252_1: "f32[256]", arg253_1: "f32[256]", arg254_1: "f32[256]", arg255_1: "f32[256]", arg256_1: "f32[256, 256, 1, 1]", arg257_1: "f32[256]", arg258_1: "f32[256]", arg259_1: "f32[256]", arg260_1: "f32[256]", arg261_1: "f32[256, 256, 3, 3]", arg262_1: "f32[256]", arg263_1: "f32[256]", arg264_1: "f32[256]", arg265_1: "f32[256]", arg266_1: "f32[256, 256, 1, 1]", arg267_1: "f32[256]", arg268_1: "f32[256]", arg269_1: "f32[256]", arg270_1: "f32[256]", arg271_1: "f32[512, 512, 1, 1]", arg272_1: "f32[512]", arg273_1: "f32[512]", arg274_1: "f32[512]", arg275_1: "f32[512]", arg276_1: "f32[1024, 512, 3, 3]", arg277_1: "f32[1024]", arg278_1: "f32[1024]", arg279_1: "f32[1024]", arg280_1: "f32[1024]", arg281_1: "f32[1024, 1024, 1, 1]", arg282_1: "f32[1024]", arg283_1: "f32[1024]", arg284_1: "f32[1024]", arg285_1: "f32[1024]", arg286_1: "f32[512, 512, 1, 1]", arg287_1: "f32[512]", arg288_1: "f32[512]", arg289_1: "f32[512]", arg290_1: "f32[512]", arg291_1: "f32[512, 512, 3, 3]", arg292_1: "f32[512]", arg293_1: "f32[512]", arg294_1: "f32[512]", arg295_1: "f32[512]", arg296_1: "f32[512, 512, 1, 1]", arg297_1: "f32[512]", arg298_1: "f32[512]", arg299_1: "f32[512]", arg300_1: "f32[512]", arg301_1: "f32[512, 512, 3, 3]", arg302_1: "f32[512]", arg303_1: "f32[512]", arg304_1: "f32[512]", arg305_1: "f32[512]", arg306_1: "f32[512, 512, 1, 1]", arg307_1: "f32[512]", arg308_1: "f32[512]", arg309_1: "f32[512]", arg310_1: "f32[512]", arg311_1: "f32[512, 512, 3, 3]", arg312_1: "f32[512]", arg313_1: "f32[512]", arg314_1: "f32[512]", arg315_1: "f32[512]", arg316_1: "f32[512, 512, 1, 1]", arg317_1: "f32[512]", arg318_1: "f32[512]", arg319_1: "f32[512]", arg320_1: "f32[512]", arg321_1: "f32[512, 512, 3, 3]", arg322_1: "f32[512]", arg323_1: "f32[512]", arg324_1: "f32[512]", arg325_1: "f32[512]", arg326_1: "f32[512, 512, 1, 1]", arg327_1: "f32[512]", arg328_1: "f32[512]", arg329_1: "f32[512]", arg330_1: "f32[512]", arg331_1: "f32[1024, 1024, 1, 1]", arg332_1: "f32[1024]", arg333_1: "f32[1024]", arg334_1: "f32[1024]", arg335_1: "f32[1024]", arg336_1: "f32[1000, 1024]", arg337_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_67: "f32[8, 32, 256, 256]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_157: "f32[32]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_67: "f32[32]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_67: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_268: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_537: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_268, -1);  mul_268 = None
        unsqueeze_539: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_537);  convolution_67 = unsqueeze_537 = None
        mul_269: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_541: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_270: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_541);  mul_269 = unsqueeze_541 = None
        unsqueeze_542: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_543: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_158: "f32[8, 32, 256, 256]" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_543);  mul_270 = unsqueeze_543 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_67: "b8[8, 32, 256, 256]" = torch.ops.aten.gt.Scalar(add_158, 0)
        mul_271: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(add_158, 0.01)
        where_67: "f32[8, 32, 256, 256]" = torch.ops.aten.where.self(gt_67, add_158, mul_271);  gt_67 = add_158 = mul_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_68: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(where_67, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_67 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_159: "f32[64]" = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_68: "f32[64]" = torch.ops.aten.sqrt.default(add_159);  add_159 = None
        reciprocal_68: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_272: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_545: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_272, -1);  mul_272 = None
        unsqueeze_547: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_545);  convolution_68 = unsqueeze_545 = None
        mul_273: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_549: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_274: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_549);  mul_273 = unsqueeze_549 = None
        unsqueeze_550: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_551: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_160: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_274, unsqueeze_551);  mul_274 = unsqueeze_551 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_68: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_160, 0)
        mul_275: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_160, 0.01)
        where_68: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_68, add_160, mul_275);  gt_68 = add_160 = mul_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_69: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(where_68, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_68 = arg11_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_161: "f32[128]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_69: "f32[128]" = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_69: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_276: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_553: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
        unsqueeze_555: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_553);  convolution_69 = unsqueeze_553 = None
        mul_277: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_557: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_278: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_557);  mul_277 = unsqueeze_557 = None
        unsqueeze_558: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_559: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_162: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_559);  mul_278 = unsqueeze_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_69: "b8[8, 128, 128, 128]" = torch.ops.aten.gt.Scalar(add_162, 0)
        mul_279: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_162, 0.01)
        where_69: "f32[8, 128, 128, 128]" = torch.ops.aten.where.self(gt_69, add_162, mul_279);  gt_69 = add_162 = mul_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        split_16 = torch.ops.aten.split.Tensor(where_69, 64, 1)
        getitem_33: "f32[8, 64, 128, 128]" = split_16[1];  split_16 = None
        convolution_70: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(getitem_33, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_163: "f32[32]" = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_70: "f32[32]" = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_70: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_280: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_561: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_280, -1);  mul_280 = None
        unsqueeze_563: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_561);  convolution_70 = unsqueeze_561 = None
        mul_281: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_565: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_282: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_281, unsqueeze_565);  mul_281 = unsqueeze_565 = None
        unsqueeze_566: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_567: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_164: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_282, unsqueeze_567);  mul_282 = unsqueeze_567 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_70: "b8[8, 32, 128, 128]" = torch.ops.aten.gt.Scalar(add_164, 0)
        mul_283: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_164, 0.01)
        where_70: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(gt_70, add_164, mul_283);  gt_70 = add_164 = mul_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_71: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(where_70, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_70 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_165: "f32[64]" = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_71: "f32[64]" = torch.ops.aten.sqrt.default(add_165);  add_165 = None
        reciprocal_71: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_284: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_569: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_284, -1);  mul_284 = None
        unsqueeze_571: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_569);  convolution_71 = unsqueeze_569 = None
        mul_285: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_573: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_286: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_285, unsqueeze_573);  mul_285 = unsqueeze_573 = None
        unsqueeze_574: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_575: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_166: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_575);  mul_286 = unsqueeze_575 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_71: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_166, 0)
        mul_287: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_166, 0.01)
        where_71: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_71, add_166, mul_287);  gt_71 = add_166 = mul_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_167: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(where_71, getitem_33);  where_71 = getitem_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_72: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(add_167, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_167 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_168: "f32[64]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_72: "f32[64]" = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_72: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_288: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_577: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_579: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_577);  convolution_72 = unsqueeze_577 = None
        mul_289: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_581: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_290: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_581);  mul_289 = unsqueeze_581 = None
        unsqueeze_582: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_583: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_169: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_583);  mul_290 = unsqueeze_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_72: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_169, 0)
        mul_291: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_169, 0.01)
        where_72: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_72, add_169, mul_291);  gt_72 = add_169 = mul_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:339 in forward, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
        split_17 = torch.ops.aten.split.Tensor(where_69, 64, 1);  where_69 = None
        getitem_34: "f32[8, 64, 128, 128]" = split_17[0];  split_17 = None
        cat_5: "f32[8, 128, 128, 128]" = torch.ops.aten.cat.default([getitem_34, where_72], 1);  getitem_34 = where_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_73: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(cat_5, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_170: "f32[64]" = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_73: "f32[64]" = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_73: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_292: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_585: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_587: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_585);  convolution_73 = unsqueeze_585 = None
        mul_293: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_589: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_294: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_589);  mul_293 = unsqueeze_589 = None
        unsqueeze_590: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_591: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_171: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_591);  mul_294 = unsqueeze_591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_73: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_171, 0)
        mul_295: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_171, 0.01)
        where_73: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_73, add_171, mul_295);  gt_73 = add_171 = mul_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_74: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(where_73, arg36_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_73 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_172: "f32[128]" = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_74: "f32[128]" = torch.ops.aten.sqrt.default(add_172);  add_172 = None
        reciprocal_74: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_296: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_593: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_296, -1);  mul_296 = None
        unsqueeze_595: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_593);  convolution_74 = unsqueeze_593 = None
        mul_297: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_597: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_298: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_297, unsqueeze_597);  mul_297 = unsqueeze_597 = None
        unsqueeze_598: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_599: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_173: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_599);  mul_298 = unsqueeze_599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_74: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_173, 0)
        mul_299: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_173, 0.01)
        where_74: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_74, add_173, mul_299);  gt_74 = add_173 = mul_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_75: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(where_74, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_74 = arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_174: "f32[128]" = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_75: "f32[128]" = torch.ops.aten.sqrt.default(add_174);  add_174 = None
        reciprocal_75: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_300: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_601: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_603: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_601);  convolution_75 = unsqueeze_601 = None
        mul_301: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_605: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_302: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_605);  mul_301 = unsqueeze_605 = None
        unsqueeze_606: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_607: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_175: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_607);  mul_302 = unsqueeze_607 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_75: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_175, 0)
        mul_303: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_175, 0.01)
        where_75: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_75, add_175, mul_303);  gt_75 = add_175 = mul_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        split_19 = torch.ops.aten.split.Tensor(where_75, 64, 1)
        getitem_39: "f32[8, 64, 64, 64]" = split_19[1];  split_19 = None
        convolution_76: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_39, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_176: "f32[64]" = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_76: "f32[64]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
        reciprocal_76: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_304: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_609: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_304, -1);  mul_304 = None
        unsqueeze_611: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_609);  convolution_76 = unsqueeze_609 = None
        mul_305: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_613: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_306: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_305, unsqueeze_613);  mul_305 = unsqueeze_613 = None
        unsqueeze_614: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_615: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_177: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_306, unsqueeze_615);  mul_306 = unsqueeze_615 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_76: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_177, 0)
        mul_307: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_177, 0.01)
        where_76: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_76, add_177, mul_307);  gt_76 = add_177 = mul_307 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_77: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(where_76, arg51_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_76 = arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_178: "f32[64]" = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_77: "f32[64]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_77: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_308: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_617: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_619: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_617);  convolution_77 = unsqueeze_617 = None
        mul_309: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_621: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_310: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_621);  mul_309 = unsqueeze_621 = None
        unsqueeze_622: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_623: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_179: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_623);  mul_310 = unsqueeze_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_77: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_179, 0)
        mul_311: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_179, 0.01)
        where_77: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_77, add_179, mul_311);  gt_77 = add_179 = mul_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_180: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(where_77, getitem_39);  where_77 = getitem_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_78: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(add_180, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_181: "f32[64]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_78: "f32[64]" = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_78: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_312: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_625: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
        unsqueeze_627: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_625);  convolution_78 = unsqueeze_625 = None
        mul_313: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_629: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_314: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_629);  mul_313 = unsqueeze_629 = None
        unsqueeze_630: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_631: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_182: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_631);  mul_314 = unsqueeze_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_78: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_182, 0)
        mul_315: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_182, 0.01)
        where_78: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_78, add_182, mul_315);  gt_78 = add_182 = mul_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_79: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(where_78, arg61_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_78 = arg61_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_183: "f32[64]" = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_79: "f32[64]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_79: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_316: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_633: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_316, -1);  mul_316 = None
        unsqueeze_635: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_633);  convolution_79 = unsqueeze_633 = None
        mul_317: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_637: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_318: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_637);  mul_317 = unsqueeze_637 = None
        unsqueeze_638: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_639: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_184: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_318, unsqueeze_639);  mul_318 = unsqueeze_639 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_79: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_184, 0)
        mul_319: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_184, 0.01)
        where_79: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_79, add_184, mul_319);  gt_79 = add_184 = mul_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_185: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(where_79, add_180);  where_79 = add_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_80: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(add_185, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_185 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_186: "f32[64]" = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_80: "f32[64]" = torch.ops.aten.sqrt.default(add_186);  add_186 = None
        reciprocal_80: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_320: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_641: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_643: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_641);  convolution_80 = unsqueeze_641 = None
        mul_321: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_645: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_322: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_645);  mul_321 = unsqueeze_645 = None
        unsqueeze_646: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_647: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_187: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_322, unsqueeze_647);  mul_322 = unsqueeze_647 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_80: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_187, 0)
        mul_323: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_187, 0.01)
        where_80: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_80, add_187, mul_323);  gt_80 = add_187 = mul_323 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:339 in forward, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
        split_20 = torch.ops.aten.split.Tensor(where_75, 64, 1);  where_75 = None
        getitem_40: "f32[8, 64, 64, 64]" = split_20[0];  split_20 = None
        cat_6: "f32[8, 128, 64, 64]" = torch.ops.aten.cat.default([getitem_40, where_80], 1);  getitem_40 = where_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_81: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(cat_6, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg71_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_188: "f32[128]" = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_81: "f32[128]" = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_81: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_324: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_649: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
        unsqueeze_651: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_649);  convolution_81 = unsqueeze_649 = None
        mul_325: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_653: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_326: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_653);  mul_325 = unsqueeze_653 = None
        unsqueeze_654: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_655: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_189: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_655);  mul_326 = unsqueeze_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_81: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_189, 0)
        mul_327: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_189, 0.01)
        where_81: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_81, add_189, mul_327);  gt_81 = add_189 = mul_327 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_82: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(where_81, arg76_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_81 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_190: "f32[256]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_82: "f32[256]" = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_82: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_328: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_657: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_328, -1);  mul_328 = None
        unsqueeze_659: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_657);  convolution_82 = unsqueeze_657 = None
        mul_329: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_661: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_330: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_661);  mul_329 = unsqueeze_661 = None
        unsqueeze_662: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_663: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_191: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_330, unsqueeze_663);  mul_330 = unsqueeze_663 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_82: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_191, 0)
        mul_331: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_191, 0.01)
        where_82: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_82, add_191, mul_331);  gt_82 = add_191 = mul_331 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_83: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(where_82, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_82 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_192: "f32[256]" = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_83: "f32[256]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_83: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_332: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_665: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_332, -1);  mul_332 = None
        unsqueeze_667: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_665);  convolution_83 = unsqueeze_665 = None
        mul_333: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_669: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_334: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_333, unsqueeze_669);  mul_333 = unsqueeze_669 = None
        unsqueeze_670: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_671: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_193: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_334, unsqueeze_671);  mul_334 = unsqueeze_671 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_83: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_193, 0)
        mul_335: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_193, 0.01)
        where_83: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_83, add_193, mul_335);  gt_83 = add_193 = mul_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        split_22 = torch.ops.aten.split.Tensor(where_83, 128, 1)
        getitem_45: "f32[8, 128, 32, 32]" = split_22[1];  split_22 = None
        convolution_84: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(getitem_45, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_194: "f32[128]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_84: "f32[128]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_84: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_336: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_673: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_675: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_673);  convolution_84 = unsqueeze_673 = None
        mul_337: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_677: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_338: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_677);  mul_337 = unsqueeze_677 = None
        unsqueeze_678: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_679: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_195: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_679);  mul_338 = unsqueeze_679 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_84: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_195, 0)
        mul_339: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_195, 0.01)
        where_84: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_84, add_195, mul_339);  gt_84 = add_195 = mul_339 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_85: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_84, arg91_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_84 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_196: "f32[128]" = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_85: "f32[128]" = torch.ops.aten.sqrt.default(add_196);  add_196 = None
        reciprocal_85: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_340: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_681: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_340, -1);  mul_340 = None
        unsqueeze_683: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_681);  convolution_85 = unsqueeze_681 = None
        mul_341: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_685: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_342: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_341, unsqueeze_685);  mul_341 = unsqueeze_685 = None
        unsqueeze_686: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_687: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_197: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_687);  mul_342 = unsqueeze_687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_85: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_197, 0)
        mul_343: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_197, 0.01)
        where_85: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_85, add_197, mul_343);  gt_85 = add_197 = mul_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_198: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_85, getitem_45);  where_85 = getitem_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_86: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_198, arg96_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_199: "f32[128]" = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_86: "f32[128]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_86: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_344: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_689: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_344, -1);  mul_344 = None
        unsqueeze_691: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_689);  convolution_86 = unsqueeze_689 = None
        mul_345: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_693: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_346: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_345, unsqueeze_693);  mul_345 = unsqueeze_693 = None
        unsqueeze_694: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_695: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_200: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_346, unsqueeze_695);  mul_346 = unsqueeze_695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_86: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_200, 0)
        mul_347: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_200, 0.01)
        where_86: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_86, add_200, mul_347);  gt_86 = add_200 = mul_347 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_87: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_86, arg101_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_86 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_201: "f32[128]" = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_87: "f32[128]" = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_87: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_348: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_697: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_699: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_697);  convolution_87 = unsqueeze_697 = None
        mul_349: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_701: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_350: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_701);  mul_349 = unsqueeze_701 = None
        unsqueeze_702: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_703: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_202: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_703);  mul_350 = unsqueeze_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_87: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_202, 0)
        mul_351: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_202, 0.01)
        where_87: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_87, add_202, mul_351);  gt_87 = add_202 = mul_351 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_203: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_87, add_198);  where_87 = add_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_88: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_203, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_204: "f32[128]" = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_88: "f32[128]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_88: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_352: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_705: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_352, -1);  mul_352 = None
        unsqueeze_707: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_705);  convolution_88 = unsqueeze_705 = None
        mul_353: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_709: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_354: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_353, unsqueeze_709);  mul_353 = unsqueeze_709 = None
        unsqueeze_710: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_711: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_205: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_354, unsqueeze_711);  mul_354 = unsqueeze_711 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_88: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_205, 0)
        mul_355: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_205, 0.01)
        where_88: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_88, add_205, mul_355);  gt_88 = add_205 = mul_355 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_89: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_88, arg111_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_88 = arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_206: "f32[128]" = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_89: "f32[128]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_89: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_356: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_713: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_356, -1);  mul_356 = None
        unsqueeze_715: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_713);  convolution_89 = unsqueeze_713 = None
        mul_357: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_717: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_358: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_717);  mul_357 = unsqueeze_717 = None
        unsqueeze_718: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_719: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_207: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_719);  mul_358 = unsqueeze_719 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_89: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_207, 0)
        mul_359: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_207, 0.01)
        where_89: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_89, add_207, mul_359);  gt_89 = add_207 = mul_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_208: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_89, add_203);  where_89 = add_203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_90: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_208, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_209: "f32[128]" = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_90: "f32[128]" = torch.ops.aten.sqrt.default(add_209);  add_209 = None
        reciprocal_90: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_360: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_721: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_723: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_721);  convolution_90 = unsqueeze_721 = None
        mul_361: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_725: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_362: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_725);  mul_361 = unsqueeze_725 = None
        unsqueeze_726: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_727: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_210: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_727);  mul_362 = unsqueeze_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_90: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_210, 0)
        mul_363: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_210, 0.01)
        where_90: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_90, add_210, mul_363);  gt_90 = add_210 = mul_363 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_91: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_90, arg121_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_90 = arg121_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_211: "f32[128]" = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_91: "f32[128]" = torch.ops.aten.sqrt.default(add_211);  add_211 = None
        reciprocal_91: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_364: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_729: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_364, -1);  mul_364 = None
        unsqueeze_731: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_729);  convolution_91 = unsqueeze_729 = None
        mul_365: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_733: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_366: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_365, unsqueeze_733);  mul_365 = unsqueeze_733 = None
        unsqueeze_734: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_735: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_212: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_366, unsqueeze_735);  mul_366 = unsqueeze_735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_91: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_212, 0)
        mul_367: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_212, 0.01)
        where_91: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_91, add_212, mul_367);  gt_91 = add_212 = mul_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_213: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_91, add_208);  where_91 = add_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_92: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_213, arg126_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_214: "f32[128]" = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_92: "f32[128]" = torch.ops.aten.sqrt.default(add_214);  add_214 = None
        reciprocal_92: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_368: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_737: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_368, -1);  mul_368 = None
        unsqueeze_739: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_737);  convolution_92 = unsqueeze_737 = None
        mul_369: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_741: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_370: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_369, unsqueeze_741);  mul_369 = unsqueeze_741 = None
        unsqueeze_742: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_743: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_215: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_743);  mul_370 = unsqueeze_743 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_92: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_215, 0)
        mul_371: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_215, 0.01)
        where_92: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_92, add_215, mul_371);  gt_92 = add_215 = mul_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_93: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_92, arg131_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_92 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_216: "f32[128]" = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_93: "f32[128]" = torch.ops.aten.sqrt.default(add_216);  add_216 = None
        reciprocal_93: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_372: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_745: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_747: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_745);  convolution_93 = unsqueeze_745 = None
        mul_373: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_749: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_374: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_749);  mul_373 = unsqueeze_749 = None
        unsqueeze_750: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_751: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_217: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_751);  mul_374 = unsqueeze_751 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_93: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_217, 0)
        mul_375: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_217, 0.01)
        where_93: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_93, add_217, mul_375);  gt_93 = add_217 = mul_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_218: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_93, add_213);  where_93 = add_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_94: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_218, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_219: "f32[128]" = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_94: "f32[128]" = torch.ops.aten.sqrt.default(add_219);  add_219 = None
        reciprocal_94: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_376: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_753: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_376, -1);  mul_376 = None
        unsqueeze_755: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_753);  convolution_94 = unsqueeze_753 = None
        mul_377: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_757: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_378: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_377, unsqueeze_757);  mul_377 = unsqueeze_757 = None
        unsqueeze_758: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_759: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_220: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_378, unsqueeze_759);  mul_378 = unsqueeze_759 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_94: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_220, 0)
        mul_379: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_220, 0.01)
        where_94: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_94, add_220, mul_379);  gt_94 = add_220 = mul_379 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_95: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_94, arg141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_94 = arg141_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_221: "f32[128]" = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_95: "f32[128]" = torch.ops.aten.sqrt.default(add_221);  add_221 = None
        reciprocal_95: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_380: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_761: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_380, -1);  mul_380 = None
        unsqueeze_763: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_761);  convolution_95 = unsqueeze_761 = None
        mul_381: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_765: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_382: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_381, unsqueeze_765);  mul_381 = unsqueeze_765 = None
        unsqueeze_766: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_767: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_222: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_382, unsqueeze_767);  mul_382 = unsqueeze_767 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_95: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_222, 0)
        mul_383: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_222, 0.01)
        where_95: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_95, add_222, mul_383);  gt_95 = add_222 = mul_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_223: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_95, add_218);  where_95 = add_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_96: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_223, arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_224: "f32[128]" = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_96: "f32[128]" = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_96: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_384: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_769: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
        unsqueeze_771: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_769);  convolution_96 = unsqueeze_769 = None
        mul_385: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_773: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_386: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_773);  mul_385 = unsqueeze_773 = None
        unsqueeze_774: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_775: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_225: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_386, unsqueeze_775);  mul_386 = unsqueeze_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_96: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_225, 0)
        mul_387: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_225, 0.01)
        where_96: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_96, add_225, mul_387);  gt_96 = add_225 = mul_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_97: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_96, arg151_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_96 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_226: "f32[128]" = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_97: "f32[128]" = torch.ops.aten.sqrt.default(add_226);  add_226 = None
        reciprocal_97: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_388: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_777: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_388, -1);  mul_388 = None
        unsqueeze_779: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_777);  convolution_97 = unsqueeze_777 = None
        mul_389: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_781: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_390: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_389, unsqueeze_781);  mul_389 = unsqueeze_781 = None
        unsqueeze_782: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_783: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_227: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_390, unsqueeze_783);  mul_390 = unsqueeze_783 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_97: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_227, 0)
        mul_391: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_227, 0.01)
        where_97: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_97, add_227, mul_391);  gt_97 = add_227 = mul_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_228: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_97, add_223);  where_97 = add_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_98: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_228, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_229: "f32[128]" = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_98: "f32[128]" = torch.ops.aten.sqrt.default(add_229);  add_229 = None
        reciprocal_98: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_392: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_785: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_392, -1);  mul_392 = None
        unsqueeze_787: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_785);  convolution_98 = unsqueeze_785 = None
        mul_393: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_789: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_394: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_393, unsqueeze_789);  mul_393 = unsqueeze_789 = None
        unsqueeze_790: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_791: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_230: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_394, unsqueeze_791);  mul_394 = unsqueeze_791 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_98: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_230, 0)
        mul_395: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_230, 0.01)
        where_98: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_98, add_230, mul_395);  gt_98 = add_230 = mul_395 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_99: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_98, arg161_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_98 = arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_231: "f32[128]" = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_99: "f32[128]" = torch.ops.aten.sqrt.default(add_231);  add_231 = None
        reciprocal_99: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_396: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_793: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
        unsqueeze_795: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_793);  convolution_99 = unsqueeze_793 = None
        mul_397: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_797: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_398: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_797);  mul_397 = unsqueeze_797 = None
        unsqueeze_798: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_799: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_232: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_799);  mul_398 = unsqueeze_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_99: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_232, 0)
        mul_399: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_232, 0.01)
        where_99: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_99, add_232, mul_399);  gt_99 = add_232 = mul_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_233: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_99, add_228);  where_99 = add_228 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_100: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_233, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_233 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_234: "f32[128]" = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_100: "f32[128]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_100: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_400: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_801: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_400, -1);  mul_400 = None
        unsqueeze_803: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_801);  convolution_100 = unsqueeze_801 = None
        mul_401: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_805: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_402: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_401, unsqueeze_805);  mul_401 = unsqueeze_805 = None
        unsqueeze_806: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_807: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_235: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_402, unsqueeze_807);  mul_402 = unsqueeze_807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_100: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_235, 0)
        mul_403: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_235, 0.01)
        where_100: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_100, add_235, mul_403);  gt_100 = add_235 = mul_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:339 in forward, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
        split_23 = torch.ops.aten.split.Tensor(where_83, 128, 1);  where_83 = None
        getitem_46: "f32[8, 128, 32, 32]" = split_23[0];  split_23 = None
        cat_7: "f32[8, 256, 32, 32]" = torch.ops.aten.cat.default([getitem_46, where_100], 1);  getitem_46 = where_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_101: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(cat_7, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_236: "f32[256]" = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_101: "f32[256]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_101: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_404: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_809: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_404, -1);  mul_404 = None
        unsqueeze_811: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_809);  convolution_101 = unsqueeze_809 = None
        mul_405: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_813: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_406: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_813);  mul_405 = unsqueeze_813 = None
        unsqueeze_814: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_815: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_237: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_815);  mul_406 = unsqueeze_815 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_101: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_237, 0)
        mul_407: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_237, 0.01)
        where_101: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_101, add_237, mul_407);  gt_101 = add_237 = mul_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_102: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(where_101, arg176_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_101 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_238: "f32[512]" = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_102: "f32[512]" = torch.ops.aten.sqrt.default(add_238);  add_238 = None
        reciprocal_102: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_408: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_817: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
        unsqueeze_819: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_817);  convolution_102 = unsqueeze_817 = None
        mul_409: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_821: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_410: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_821);  mul_409 = unsqueeze_821 = None
        unsqueeze_822: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_823: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_239: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_410, unsqueeze_823);  mul_410 = unsqueeze_823 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_102: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_239, 0)
        mul_411: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_239, 0.01)
        where_102: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_102, add_239, mul_411);  gt_102 = add_239 = mul_411 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_103: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(where_102, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_102 = arg181_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_240: "f32[512]" = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_103: "f32[512]" = torch.ops.aten.sqrt.default(add_240);  add_240 = None
        reciprocal_103: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_412: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_825: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_412, -1);  mul_412 = None
        unsqueeze_827: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_825);  convolution_103 = unsqueeze_825 = None
        mul_413: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_829: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_414: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_829);  mul_413 = unsqueeze_829 = None
        unsqueeze_830: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_831: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_241: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_414, unsqueeze_831);  mul_414 = unsqueeze_831 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_103: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_241, 0)
        mul_415: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_241, 0.01)
        where_103: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_103, add_241, mul_415);  gt_103 = add_241 = mul_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        split_25 = torch.ops.aten.split.Tensor(where_103, 256, 1)
        getitem_51: "f32[8, 256, 16, 16]" = split_25[1];  split_25 = None
        convolution_104: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(getitem_51, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_242: "f32[256]" = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_104: "f32[256]" = torch.ops.aten.sqrt.default(add_242);  add_242 = None
        reciprocal_104: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_416: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_833: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_416, -1);  mul_416 = None
        unsqueeze_835: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_833);  convolution_104 = unsqueeze_833 = None
        mul_417: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_837: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_418: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_417, unsqueeze_837);  mul_417 = unsqueeze_837 = None
        unsqueeze_838: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_839: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_243: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_418, unsqueeze_839);  mul_418 = unsqueeze_839 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_104: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_243, 0)
        mul_419: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_243, 0.01)
        where_104: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_104, add_243, mul_419);  gt_104 = add_243 = mul_419 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_105: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_104, arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_104 = arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_244: "f32[256]" = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_105: "f32[256]" = torch.ops.aten.sqrt.default(add_244);  add_244 = None
        reciprocal_105: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_841: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
        unsqueeze_843: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_841);  convolution_105 = unsqueeze_841 = None
        mul_421: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_845: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_422: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_845);  mul_421 = unsqueeze_845 = None
        unsqueeze_846: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_847: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_245: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_847);  mul_422 = unsqueeze_847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_105: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_245, 0)
        mul_423: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_245, 0.01)
        where_105: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_105, add_245, mul_423);  gt_105 = add_245 = mul_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_246: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_105, getitem_51);  where_105 = getitem_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_106: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_246, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_247: "f32[256]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_106: "f32[256]" = torch.ops.aten.sqrt.default(add_247);  add_247 = None
        reciprocal_106: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_424: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_849: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_424, -1);  mul_424 = None
        unsqueeze_851: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_849);  convolution_106 = unsqueeze_849 = None
        mul_425: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_853: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_426: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_425, unsqueeze_853);  mul_425 = unsqueeze_853 = None
        unsqueeze_854: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_855: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_248: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_855);  mul_426 = unsqueeze_855 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_106: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_248, 0)
        mul_427: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_248, 0.01)
        where_106: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_106, add_248, mul_427);  gt_106 = add_248 = mul_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_107: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_106, arg201_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_106 = arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_249: "f32[256]" = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_107: "f32[256]" = torch.ops.aten.sqrt.default(add_249);  add_249 = None
        reciprocal_107: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_428: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_857: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_428, -1);  mul_428 = None
        unsqueeze_859: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_857);  convolution_107 = unsqueeze_857 = None
        mul_429: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_861: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_430: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_429, unsqueeze_861);  mul_429 = unsqueeze_861 = None
        unsqueeze_862: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_863: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_250: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_430, unsqueeze_863);  mul_430 = unsqueeze_863 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_107: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_250, 0)
        mul_431: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_250, 0.01)
        where_107: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_107, add_250, mul_431);  gt_107 = add_250 = mul_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_251: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_107, add_246);  where_107 = add_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_108: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_251, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_252: "f32[256]" = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_108: "f32[256]" = torch.ops.aten.sqrt.default(add_252);  add_252 = None
        reciprocal_108: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_432: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_865: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
        unsqueeze_867: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_865);  convolution_108 = unsqueeze_865 = None
        mul_433: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_869: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_434: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_869);  mul_433 = unsqueeze_869 = None
        unsqueeze_870: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_871: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_253: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_434, unsqueeze_871);  mul_434 = unsqueeze_871 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_108: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_253, 0)
        mul_435: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_253, 0.01)
        where_108: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_108, add_253, mul_435);  gt_108 = add_253 = mul_435 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_109: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_108, arg211_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_108 = arg211_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_254: "f32[256]" = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_109: "f32[256]" = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_109: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_436: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_873: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_436, -1);  mul_436 = None
        unsqueeze_875: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_873);  convolution_109 = unsqueeze_873 = None
        mul_437: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_877: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_438: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_437, unsqueeze_877);  mul_437 = unsqueeze_877 = None
        unsqueeze_878: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_879: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_255: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_438, unsqueeze_879);  mul_438 = unsqueeze_879 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_109: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_255, 0)
        mul_439: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_255, 0.01)
        where_109: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_109, add_255, mul_439);  gt_109 = add_255 = mul_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_256: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_109, add_251);  where_109 = add_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_110: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_256, arg216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg216_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_257: "f32[256]" = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_110: "f32[256]" = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_110: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_440: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_881: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_440, -1);  mul_440 = None
        unsqueeze_883: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_881);  convolution_110 = unsqueeze_881 = None
        mul_441: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_885: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_442: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_885);  mul_441 = unsqueeze_885 = None
        unsqueeze_886: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_887: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_258: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_442, unsqueeze_887);  mul_442 = unsqueeze_887 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_110: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_258, 0)
        mul_443: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_258, 0.01)
        where_110: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_110, add_258, mul_443);  gt_110 = add_258 = mul_443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_111: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_110, arg221_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_110 = arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_259: "f32[256]" = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_111: "f32[256]" = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_111: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_444: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_889: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_891: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_889);  convolution_111 = unsqueeze_889 = None
        mul_445: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_893: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_446: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_893);  mul_445 = unsqueeze_893 = None
        unsqueeze_894: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_895: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_260: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_895);  mul_446 = unsqueeze_895 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_111: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_260, 0)
        mul_447: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_260, 0.01)
        where_111: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_111, add_260, mul_447);  gt_111 = add_260 = mul_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_261: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_111, add_256);  where_111 = add_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_112: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_261, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg226_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_262: "f32[256]" = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_112: "f32[256]" = torch.ops.aten.sqrt.default(add_262);  add_262 = None
        reciprocal_112: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_448: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_897: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_448, -1);  mul_448 = None
        unsqueeze_899: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_897);  convolution_112 = unsqueeze_897 = None
        mul_449: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_901: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_450: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_449, unsqueeze_901);  mul_449 = unsqueeze_901 = None
        unsqueeze_902: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_903: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_263: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_450, unsqueeze_903);  mul_450 = unsqueeze_903 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_112: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_263, 0)
        mul_451: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_263, 0.01)
        where_112: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_112, add_263, mul_451);  gt_112 = add_263 = mul_451 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_113: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_112, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_112 = arg231_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_264: "f32[256]" = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_113: "f32[256]" = torch.ops.aten.sqrt.default(add_264);  add_264 = None
        reciprocal_113: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_452: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_905: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_452, -1);  mul_452 = None
        unsqueeze_907: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_905);  convolution_113 = unsqueeze_905 = None
        mul_453: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_909: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_454: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_453, unsqueeze_909);  mul_453 = unsqueeze_909 = None
        unsqueeze_910: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_911: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_265: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_911);  mul_454 = unsqueeze_911 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_113: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_265, 0)
        mul_455: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_265, 0.01)
        where_113: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_113, add_265, mul_455);  gt_113 = add_265 = mul_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_266: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_113, add_261);  where_113 = add_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_114: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_266, arg236_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg236_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_267: "f32[256]" = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_114: "f32[256]" = torch.ops.aten.sqrt.default(add_267);  add_267 = None
        reciprocal_114: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_456: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_913: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_915: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_913);  convolution_114 = unsqueeze_913 = None
        mul_457: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_917: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_458: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_917);  mul_457 = unsqueeze_917 = None
        unsqueeze_918: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_919: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_268: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_458, unsqueeze_919);  mul_458 = unsqueeze_919 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_114: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_268, 0)
        mul_459: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_268, 0.01)
        where_114: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_114, add_268, mul_459);  gt_114 = add_268 = mul_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_115: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_114, arg241_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_114 = arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_269: "f32[256]" = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_115: "f32[256]" = torch.ops.aten.sqrt.default(add_269);  add_269 = None
        reciprocal_115: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_460: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_921: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_460, -1);  mul_460 = None
        unsqueeze_923: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_921);  convolution_115 = unsqueeze_921 = None
        mul_461: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_925: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_462: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_461, unsqueeze_925);  mul_461 = unsqueeze_925 = None
        unsqueeze_926: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_927: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_270: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_927);  mul_462 = unsqueeze_927 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_115: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_270, 0)
        mul_463: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_270, 0.01)
        where_115: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_115, add_270, mul_463);  gt_115 = add_270 = mul_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_271: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_115, add_266);  where_115 = add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_116: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_271, arg246_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg246_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_272: "f32[256]" = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_116: "f32[256]" = torch.ops.aten.sqrt.default(add_272);  add_272 = None
        reciprocal_116: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_464: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_929: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_464, -1);  mul_464 = None
        unsqueeze_931: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_929);  convolution_116 = unsqueeze_929 = None
        mul_465: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_933: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_466: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_465, unsqueeze_933);  mul_465 = unsqueeze_933 = None
        unsqueeze_934: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_935: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_273: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_466, unsqueeze_935);  mul_466 = unsqueeze_935 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_116: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_273, 0)
        mul_467: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_273, 0.01)
        where_116: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_116, add_273, mul_467);  gt_116 = add_273 = mul_467 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_117: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_116, arg251_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_116 = arg251_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_274: "f32[256]" = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_117: "f32[256]" = torch.ops.aten.sqrt.default(add_274);  add_274 = None
        reciprocal_117: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_468: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_937: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_468, -1);  mul_468 = None
        unsqueeze_939: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_937);  convolution_117 = unsqueeze_937 = None
        mul_469: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_941: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_470: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_941);  mul_469 = unsqueeze_941 = None
        unsqueeze_942: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_943: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_275: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_470, unsqueeze_943);  mul_470 = unsqueeze_943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_117: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_275, 0)
        mul_471: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_275, 0.01)
        where_117: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_117, add_275, mul_471);  gt_117 = add_275 = mul_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_276: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_117, add_271);  where_117 = add_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_118: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_276, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg256_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_277: "f32[256]" = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_118: "f32[256]" = torch.ops.aten.sqrt.default(add_277);  add_277 = None
        reciprocal_118: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_472: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_945: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_472, -1);  mul_472 = None
        unsqueeze_947: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_945);  convolution_118 = unsqueeze_945 = None
        mul_473: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_949: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_474: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_949);  mul_473 = unsqueeze_949 = None
        unsqueeze_950: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_951: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_278: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_474, unsqueeze_951);  mul_474 = unsqueeze_951 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_118: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_278, 0)
        mul_475: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_278, 0.01)
        where_118: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_118, add_278, mul_475);  gt_118 = add_278 = mul_475 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_119: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_118, arg261_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_118 = arg261_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_279: "f32[256]" = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_119: "f32[256]" = torch.ops.aten.sqrt.default(add_279);  add_279 = None
        reciprocal_119: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_476: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_953: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_476, -1);  mul_476 = None
        unsqueeze_955: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_953);  convolution_119 = unsqueeze_953 = None
        mul_477: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_957: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_478: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_477, unsqueeze_957);  mul_477 = unsqueeze_957 = None
        unsqueeze_958: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_959: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_280: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_478, unsqueeze_959);  mul_478 = unsqueeze_959 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_119: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_280, 0)
        mul_479: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_280, 0.01)
        where_119: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_119, add_280, mul_479);  gt_119 = add_280 = mul_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_281: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_119, add_276);  where_119 = add_276 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_120: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_281, arg266_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_281 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_282: "f32[256]" = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
        sqrt_120: "f32[256]" = torch.ops.aten.sqrt.default(add_282);  add_282 = None
        reciprocal_120: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_480: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_961: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_480, -1);  mul_480 = None
        unsqueeze_963: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_961);  convolution_120 = unsqueeze_961 = None
        mul_481: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_965: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_482: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_965);  mul_481 = unsqueeze_965 = None
        unsqueeze_966: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_967: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_283: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_967);  mul_482 = unsqueeze_967 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_120: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_283, 0)
        mul_483: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_283, 0.01)
        where_120: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_120, add_283, mul_483);  gt_120 = add_283 = mul_483 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:339 in forward, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
        split_26 = torch.ops.aten.split.Tensor(where_103, 256, 1);  where_103 = None
        getitem_52: "f32[8, 256, 16, 16]" = split_26[0];  split_26 = None
        cat_8: "f32[8, 512, 16, 16]" = torch.ops.aten.cat.default([getitem_52, where_120], 1);  getitem_52 = where_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_121: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(cat_8, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_8 = arg271_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_284: "f32[512]" = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_121: "f32[512]" = torch.ops.aten.sqrt.default(add_284);  add_284 = None
        reciprocal_121: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_484: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_969: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_484, -1);  mul_484 = None
        unsqueeze_971: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_969);  convolution_121 = unsqueeze_969 = None
        mul_485: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_973: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_486: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_485, unsqueeze_973);  mul_485 = unsqueeze_973 = None
        unsqueeze_974: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_975: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_285: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_486, unsqueeze_975);  mul_486 = unsqueeze_975 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_121: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_285, 0)
        mul_487: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_285, 0.01)
        where_121: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_121, add_285, mul_487);  gt_121 = add_285 = mul_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_122: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(where_121, arg276_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_121 = arg276_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_286: "f32[1024]" = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
        sqrt_122: "f32[1024]" = torch.ops.aten.sqrt.default(add_286);  add_286 = None
        reciprocal_122: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_488: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_977: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_488, -1);  mul_488 = None
        unsqueeze_979: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_977);  convolution_122 = unsqueeze_977 = None
        mul_489: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_981: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_490: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_489, unsqueeze_981);  mul_489 = unsqueeze_981 = None
        unsqueeze_982: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_983: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_287: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_490, unsqueeze_983);  mul_490 = unsqueeze_983 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_122: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_287, 0)
        mul_491: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_287, 0.01)
        where_122: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_122, add_287, mul_491);  gt_122 = add_287 = mul_491 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_123: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(where_122, arg281_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_122 = arg281_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_288: "f32[1024]" = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_123: "f32[1024]" = torch.ops.aten.sqrt.default(add_288);  add_288 = None
        reciprocal_123: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_492: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_985: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_492, -1);  mul_492 = None
        unsqueeze_987: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_985);  convolution_123 = unsqueeze_985 = None
        mul_493: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_989: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_494: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_989);  mul_493 = unsqueeze_989 = None
        unsqueeze_990: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_991: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_289: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_494, unsqueeze_991);  mul_494 = unsqueeze_991 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_123: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_289, 0)
        mul_495: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_289, 0.01)
        where_123: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_123, add_289, mul_495);  gt_123 = add_289 = mul_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        split_28 = torch.ops.aten.split.Tensor(where_123, 512, 1)
        getitem_57: "f32[8, 512, 8, 8]" = split_28[1];  split_28 = None
        convolution_124: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(getitem_57, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_290: "f32[512]" = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_124: "f32[512]" = torch.ops.aten.sqrt.default(add_290);  add_290 = None
        reciprocal_124: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_496: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_992: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_993: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        unsqueeze_994: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_496, -1);  mul_496 = None
        unsqueeze_995: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        sub_124: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_993);  convolution_124 = unsqueeze_993 = None
        mul_497: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_997: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_498: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_997);  mul_497 = unsqueeze_997 = None
        unsqueeze_998: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_999: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_291: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_498, unsqueeze_999);  mul_498 = unsqueeze_999 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_124: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_291, 0)
        mul_499: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_291, 0.01)
        where_124: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_124, add_291, mul_499);  gt_124 = add_291 = mul_499 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_125: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_124, arg291_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_124 = arg291_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_292: "f32[512]" = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_125: "f32[512]" = torch.ops.aten.sqrt.default(add_292);  add_292 = None
        reciprocal_125: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_500: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1000: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_1001: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        unsqueeze_1002: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_500, -1);  mul_500 = None
        unsqueeze_1003: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        sub_125: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_1001);  convolution_125 = unsqueeze_1001 = None
        mul_501: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_1005: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_502: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_501, unsqueeze_1005);  mul_501 = unsqueeze_1005 = None
        unsqueeze_1006: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_1007: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_293: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_502, unsqueeze_1007);  mul_502 = unsqueeze_1007 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_125: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_293, 0)
        mul_503: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_293, 0.01)
        where_125: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_125, add_293, mul_503);  gt_125 = add_293 = mul_503 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_294: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_125, getitem_57);  where_125 = getitem_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_126: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_294, arg296_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg296_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_295: "f32[512]" = torch.ops.aten.add.Tensor(arg298_1, 1e-05);  arg298_1 = None
        sqrt_126: "f32[512]" = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_126: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_504: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1008: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_1009: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        unsqueeze_1010: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_504, -1);  mul_504 = None
        unsqueeze_1011: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        sub_126: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1009);  convolution_126 = unsqueeze_1009 = None
        mul_505: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_1013: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_506: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_1013);  mul_505 = unsqueeze_1013 = None
        unsqueeze_1014: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_1015: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_296: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_506, unsqueeze_1015);  mul_506 = unsqueeze_1015 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_126: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_296, 0)
        mul_507: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_296, 0.01)
        where_126: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_126, add_296, mul_507);  gt_126 = add_296 = mul_507 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_127: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_126, arg301_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_126 = arg301_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_297: "f32[512]" = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_127: "f32[512]" = torch.ops.aten.sqrt.default(add_297);  add_297 = None
        reciprocal_127: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_508: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1016: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_1017: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        unsqueeze_1018: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_508, -1);  mul_508 = None
        unsqueeze_1019: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        sub_127: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1017);  convolution_127 = unsqueeze_1017 = None
        mul_509: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_1021: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_510: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_509, unsqueeze_1021);  mul_509 = unsqueeze_1021 = None
        unsqueeze_1022: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_1023: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_298: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_1023);  mul_510 = unsqueeze_1023 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_127: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_298, 0)
        mul_511: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_298, 0.01)
        where_127: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_127, add_298, mul_511);  gt_127 = add_298 = mul_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_299: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_127, add_294);  where_127 = add_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_128: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_299, arg306_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg306_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_300: "f32[512]" = torch.ops.aten.add.Tensor(arg308_1, 1e-05);  arg308_1 = None
        sqrt_128: "f32[512]" = torch.ops.aten.sqrt.default(add_300);  add_300 = None
        reciprocal_128: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_512: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1024: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
        unsqueeze_1025: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        unsqueeze_1026: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_512, -1);  mul_512 = None
        unsqueeze_1027: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        sub_128: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_1025);  convolution_128 = unsqueeze_1025 = None
        mul_513: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1029: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_514: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_513, unsqueeze_1029);  mul_513 = unsqueeze_1029 = None
        unsqueeze_1030: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1031: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_301: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_514, unsqueeze_1031);  mul_514 = unsqueeze_1031 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_128: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_301, 0)
        mul_515: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_301, 0.01)
        where_128: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_128, add_301, mul_515);  gt_128 = add_301 = mul_515 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_129: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_128, arg311_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_128 = arg311_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_302: "f32[512]" = torch.ops.aten.add.Tensor(arg313_1, 1e-05);  arg313_1 = None
        sqrt_129: "f32[512]" = torch.ops.aten.sqrt.default(add_302);  add_302 = None
        reciprocal_129: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_516: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1032: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_1033: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        unsqueeze_1034: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_516, -1);  mul_516 = None
        unsqueeze_1035: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        sub_129: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_1033);  convolution_129 = unsqueeze_1033 = None
        mul_517: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1037: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_518: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_1037);  mul_517 = unsqueeze_1037 = None
        unsqueeze_1038: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_1039: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_303: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_518, unsqueeze_1039);  mul_518 = unsqueeze_1039 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_129: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_303, 0)
        mul_519: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_303, 0.01)
        where_129: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_129, add_303, mul_519);  gt_129 = add_303 = mul_519 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_304: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_129, add_299);  where_129 = add_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_130: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_304, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg316_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_305: "f32[512]" = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_130: "f32[512]" = torch.ops.aten.sqrt.default(add_305);  add_305 = None
        reciprocal_130: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_520: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1040: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1041: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        unsqueeze_1042: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_520, -1);  mul_520 = None
        unsqueeze_1043: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        sub_130: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_1041);  convolution_130 = unsqueeze_1041 = None
        mul_521: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1045: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_522: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_521, unsqueeze_1045);  mul_521 = unsqueeze_1045 = None
        unsqueeze_1046: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_1047: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_306: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_522, unsqueeze_1047);  mul_522 = unsqueeze_1047 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_130: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_306, 0)
        mul_523: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_306, 0.01)
        where_130: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_130, add_306, mul_523);  gt_130 = add_306 = mul_523 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_131: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_130, arg321_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_130 = arg321_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_307: "f32[512]" = torch.ops.aten.add.Tensor(arg323_1, 1e-05);  arg323_1 = None
        sqrt_131: "f32[512]" = torch.ops.aten.sqrt.default(add_307);  add_307 = None
        reciprocal_131: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_524: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1048: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, -1);  arg322_1 = None
        unsqueeze_1049: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        unsqueeze_1050: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_524, -1);  mul_524 = None
        unsqueeze_1051: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        sub_131: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_1049);  convolution_131 = unsqueeze_1049 = None
        mul_525: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_1053: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_526: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_1053);  mul_525 = unsqueeze_1053 = None
        unsqueeze_1054: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_1055: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_308: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_526, unsqueeze_1055);  mul_526 = unsqueeze_1055 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_131: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_308, 0)
        mul_527: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_308, 0.01)
        where_131: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_131, add_308, mul_527);  gt_131 = add_308 = mul_527 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:222 in forward, code: x = self.drop_path(x) + shortcut
        add_309: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_131, add_304);  where_131 = add_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_132: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_309, arg326_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_309 = arg326_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_310: "f32[512]" = torch.ops.aten.add.Tensor(arg328_1, 1e-05);  arg328_1 = None
        sqrt_132: "f32[512]" = torch.ops.aten.sqrt.default(add_310);  add_310 = None
        reciprocal_132: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_528: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1056: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_1057: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        unsqueeze_1058: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_528, -1);  mul_528 = None
        unsqueeze_1059: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        sub_132: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_1057);  convolution_132 = unsqueeze_1057 = None
        mul_529: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_1061: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_530: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_529, unsqueeze_1061);  mul_529 = unsqueeze_1061 = None
        unsqueeze_1062: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_1063: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_311: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_530, unsqueeze_1063);  mul_530 = unsqueeze_1063 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_132: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_311, 0)
        mul_531: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_311, 0.01)
        where_132: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_132, add_311, mul_531);  gt_132 = add_311 = mul_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/cspnet.py:339 in forward, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
        split_29 = torch.ops.aten.split.Tensor(where_123, 512, 1);  where_123 = None
        getitem_58: "f32[8, 512, 8, 8]" = split_29[0];  split_29 = None
        cat_9: "f32[8, 1024, 8, 8]" = torch.ops.aten.cat.default([getitem_58, where_132], 1);  getitem_58 = where_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:83 in forward, code: x = self.conv(x)
        convolution_133: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(cat_9, arg331_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_9 = arg331_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:115 in forward, code: x = F.batch_norm(
        add_312: "f32[1024]" = torch.ops.aten.add.Tensor(arg333_1, 1e-05);  arg333_1 = None
        sqrt_133: "f32[1024]" = torch.ops.aten.sqrt.default(add_312);  add_312 = None
        reciprocal_133: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_532: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1064: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg332_1, -1);  arg332_1 = None
        unsqueeze_1065: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        unsqueeze_1066: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_532, -1);  mul_532 = None
        unsqueeze_1067: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        sub_133: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_1065);  convolution_133 = unsqueeze_1065 = None
        mul_533: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_1069: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_534: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_533, unsqueeze_1069);  mul_533 = unsqueeze_1069 = None
        unsqueeze_1070: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_1071: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_313: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_534, unsqueeze_1071);  mul_534 = unsqueeze_1071 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm_act.py:127 in forward, code: x = self.act(x)
        gt_133: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_313, 0)
        mul_535: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_313, 0.01)
        where_133: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_133, add_313, mul_535);  gt_133 = add_313 = mul_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(where_133, [-1, -2], True);  where_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_1: "f32[8, 1024]" = torch.ops.aten.view.default(mean_1, [8, 1024]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_1: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg337_1, view_1, permute_1);  arg337_1 = view_1 = permute_1 = None
        return (addmm_1,)
        