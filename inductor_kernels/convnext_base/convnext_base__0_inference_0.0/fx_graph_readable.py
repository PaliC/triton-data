class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[128, 3, 4, 4]", arg1_1: "f32[128]", arg2_1: "f32[8, 3, 288, 288]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128, 1, 7, 7]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[512, 128]", arg10_1: "f32[512]", arg11_1: "f32[128, 512]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128, 1, 7, 7]", arg15_1: "f32[128]", arg16_1: "f32[128]", arg17_1: "f32[128]", arg18_1: "f32[512, 128]", arg19_1: "f32[512]", arg20_1: "f32[128, 512]", arg21_1: "f32[128]", arg22_1: "f32[128]", arg23_1: "f32[128, 1, 7, 7]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[512, 128]", arg28_1: "f32[512]", arg29_1: "f32[128, 512]", arg30_1: "f32[128]", arg31_1: "f32[128]", arg32_1: "f32[128]", arg33_1: "f32[128]", arg34_1: "f32[256, 128, 2, 2]", arg35_1: "f32[256]", arg36_1: "f32[256, 1, 7, 7]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[256]", arg40_1: "f32[1024, 256]", arg41_1: "f32[1024]", arg42_1: "f32[256, 1024]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[256, 1, 7, 7]", arg46_1: "f32[256]", arg47_1: "f32[256]", arg48_1: "f32[256]", arg49_1: "f32[1024, 256]", arg50_1: "f32[1024]", arg51_1: "f32[256, 1024]", arg52_1: "f32[256]", arg53_1: "f32[256]", arg54_1: "f32[256, 1, 7, 7]", arg55_1: "f32[256]", arg56_1: "f32[256]", arg57_1: "f32[256]", arg58_1: "f32[1024, 256]", arg59_1: "f32[1024]", arg60_1: "f32[256, 1024]", arg61_1: "f32[256]", arg62_1: "f32[256]", arg63_1: "f32[256]", arg64_1: "f32[256]", arg65_1: "f32[512, 256, 2, 2]", arg66_1: "f32[512]", arg67_1: "f32[512, 1, 7, 7]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[512]", arg71_1: "f32[2048, 512]", arg72_1: "f32[2048]", arg73_1: "f32[512, 2048]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[512, 1, 7, 7]", arg77_1: "f32[512]", arg78_1: "f32[512]", arg79_1: "f32[512]", arg80_1: "f32[2048, 512]", arg81_1: "f32[2048]", arg82_1: "f32[512, 2048]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[512, 1, 7, 7]", arg86_1: "f32[512]", arg87_1: "f32[512]", arg88_1: "f32[512]", arg89_1: "f32[2048, 512]", arg90_1: "f32[2048]", arg91_1: "f32[512, 2048]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[512, 1, 7, 7]", arg95_1: "f32[512]", arg96_1: "f32[512]", arg97_1: "f32[512]", arg98_1: "f32[2048, 512]", arg99_1: "f32[2048]", arg100_1: "f32[512, 2048]", arg101_1: "f32[512]", arg102_1: "f32[512]", arg103_1: "f32[512, 1, 7, 7]", arg104_1: "f32[512]", arg105_1: "f32[512]", arg106_1: "f32[512]", arg107_1: "f32[2048, 512]", arg108_1: "f32[2048]", arg109_1: "f32[512, 2048]", arg110_1: "f32[512]", arg111_1: "f32[512]", arg112_1: "f32[512, 1, 7, 7]", arg113_1: "f32[512]", arg114_1: "f32[512]", arg115_1: "f32[512]", arg116_1: "f32[2048, 512]", arg117_1: "f32[2048]", arg118_1: "f32[512, 2048]", arg119_1: "f32[512]", arg120_1: "f32[512]", arg121_1: "f32[512, 1, 7, 7]", arg122_1: "f32[512]", arg123_1: "f32[512]", arg124_1: "f32[512]", arg125_1: "f32[2048, 512]", arg126_1: "f32[2048]", arg127_1: "f32[512, 2048]", arg128_1: "f32[512]", arg129_1: "f32[512]", arg130_1: "f32[512, 1, 7, 7]", arg131_1: "f32[512]", arg132_1: "f32[512]", arg133_1: "f32[512]", arg134_1: "f32[2048, 512]", arg135_1: "f32[2048]", arg136_1: "f32[512, 2048]", arg137_1: "f32[512]", arg138_1: "f32[512]", arg139_1: "f32[512, 1, 7, 7]", arg140_1: "f32[512]", arg141_1: "f32[512]", arg142_1: "f32[512]", arg143_1: "f32[2048, 512]", arg144_1: "f32[2048]", arg145_1: "f32[512, 2048]", arg146_1: "f32[512]", arg147_1: "f32[512]", arg148_1: "f32[512, 1, 7, 7]", arg149_1: "f32[512]", arg150_1: "f32[512]", arg151_1: "f32[512]", arg152_1: "f32[2048, 512]", arg153_1: "f32[2048]", arg154_1: "f32[512, 2048]", arg155_1: "f32[512]", arg156_1: "f32[512]", arg157_1: "f32[512, 1, 7, 7]", arg158_1: "f32[512]", arg159_1: "f32[512]", arg160_1: "f32[512]", arg161_1: "f32[2048, 512]", arg162_1: "f32[2048]", arg163_1: "f32[512, 2048]", arg164_1: "f32[512]", arg165_1: "f32[512]", arg166_1: "f32[512, 1, 7, 7]", arg167_1: "f32[512]", arg168_1: "f32[512]", arg169_1: "f32[512]", arg170_1: "f32[2048, 512]", arg171_1: "f32[2048]", arg172_1: "f32[512, 2048]", arg173_1: "f32[512]", arg174_1: "f32[512]", arg175_1: "f32[512, 1, 7, 7]", arg176_1: "f32[512]", arg177_1: "f32[512]", arg178_1: "f32[512]", arg179_1: "f32[2048, 512]", arg180_1: "f32[2048]", arg181_1: "f32[512, 2048]", arg182_1: "f32[512]", arg183_1: "f32[512]", arg184_1: "f32[512, 1, 7, 7]", arg185_1: "f32[512]", arg186_1: "f32[512]", arg187_1: "f32[512]", arg188_1: "f32[2048, 512]", arg189_1: "f32[2048]", arg190_1: "f32[512, 2048]", arg191_1: "f32[512]", arg192_1: "f32[512]", arg193_1: "f32[512, 1, 7, 7]", arg194_1: "f32[512]", arg195_1: "f32[512]", arg196_1: "f32[512]", arg197_1: "f32[2048, 512]", arg198_1: "f32[2048]", arg199_1: "f32[512, 2048]", arg200_1: "f32[512]", arg201_1: "f32[512]", arg202_1: "f32[512, 1, 7, 7]", arg203_1: "f32[512]", arg204_1: "f32[512]", arg205_1: "f32[512]", arg206_1: "f32[2048, 512]", arg207_1: "f32[2048]", arg208_1: "f32[512, 2048]", arg209_1: "f32[512]", arg210_1: "f32[512]", arg211_1: "f32[512, 1, 7, 7]", arg212_1: "f32[512]", arg213_1: "f32[512]", arg214_1: "f32[512]", arg215_1: "f32[2048, 512]", arg216_1: "f32[2048]", arg217_1: "f32[512, 2048]", arg218_1: "f32[512]", arg219_1: "f32[512]", arg220_1: "f32[512, 1, 7, 7]", arg221_1: "f32[512]", arg222_1: "f32[512]", arg223_1: "f32[512]", arg224_1: "f32[2048, 512]", arg225_1: "f32[2048]", arg226_1: "f32[512, 2048]", arg227_1: "f32[512]", arg228_1: "f32[512]", arg229_1: "f32[512, 1, 7, 7]", arg230_1: "f32[512]", arg231_1: "f32[512]", arg232_1: "f32[512]", arg233_1: "f32[2048, 512]", arg234_1: "f32[2048]", arg235_1: "f32[512, 2048]", arg236_1: "f32[512]", arg237_1: "f32[512]", arg238_1: "f32[512, 1, 7, 7]", arg239_1: "f32[512]", arg240_1: "f32[512]", arg241_1: "f32[512]", arg242_1: "f32[2048, 512]", arg243_1: "f32[2048]", arg244_1: "f32[512, 2048]", arg245_1: "f32[512]", arg246_1: "f32[512]", arg247_1: "f32[512, 1, 7, 7]", arg248_1: "f32[512]", arg249_1: "f32[512]", arg250_1: "f32[512]", arg251_1: "f32[2048, 512]", arg252_1: "f32[2048]", arg253_1: "f32[512, 2048]", arg254_1: "f32[512]", arg255_1: "f32[512]", arg256_1: "f32[512, 1, 7, 7]", arg257_1: "f32[512]", arg258_1: "f32[512]", arg259_1: "f32[512]", arg260_1: "f32[2048, 512]", arg261_1: "f32[2048]", arg262_1: "f32[512, 2048]", arg263_1: "f32[512]", arg264_1: "f32[512]", arg265_1: "f32[512, 1, 7, 7]", arg266_1: "f32[512]", arg267_1: "f32[512]", arg268_1: "f32[512]", arg269_1: "f32[2048, 512]", arg270_1: "f32[2048]", arg271_1: "f32[512, 2048]", arg272_1: "f32[512]", arg273_1: "f32[512]", arg274_1: "f32[512, 1, 7, 7]", arg275_1: "f32[512]", arg276_1: "f32[512]", arg277_1: "f32[512]", arg278_1: "f32[2048, 512]", arg279_1: "f32[2048]", arg280_1: "f32[512, 2048]", arg281_1: "f32[512]", arg282_1: "f32[512]", arg283_1: "f32[512, 1, 7, 7]", arg284_1: "f32[512]", arg285_1: "f32[512]", arg286_1: "f32[512]", arg287_1: "f32[2048, 512]", arg288_1: "f32[2048]", arg289_1: "f32[512, 2048]", arg290_1: "f32[512]", arg291_1: "f32[512]", arg292_1: "f32[512, 1, 7, 7]", arg293_1: "f32[512]", arg294_1: "f32[512]", arg295_1: "f32[512]", arg296_1: "f32[2048, 512]", arg297_1: "f32[2048]", arg298_1: "f32[512, 2048]", arg299_1: "f32[512]", arg300_1: "f32[512]", arg301_1: "f32[512, 1, 7, 7]", arg302_1: "f32[512]", arg303_1: "f32[512]", arg304_1: "f32[512]", arg305_1: "f32[2048, 512]", arg306_1: "f32[2048]", arg307_1: "f32[512, 2048]", arg308_1: "f32[512]", arg309_1: "f32[512]", arg310_1: "f32[512]", arg311_1: "f32[512]", arg312_1: "f32[1024, 512, 2, 2]", arg313_1: "f32[1024]", arg314_1: "f32[1024, 1, 7, 7]", arg315_1: "f32[1024]", arg316_1: "f32[1024]", arg317_1: "f32[1024]", arg318_1: "f32[4096, 1024]", arg319_1: "f32[4096]", arg320_1: "f32[1024, 4096]", arg321_1: "f32[1024]", arg322_1: "f32[1024]", arg323_1: "f32[1024, 1, 7, 7]", arg324_1: "f32[1024]", arg325_1: "f32[1024]", arg326_1: "f32[1024]", arg327_1: "f32[4096, 1024]", arg328_1: "f32[4096]", arg329_1: "f32[1024, 4096]", arg330_1: "f32[1024]", arg331_1: "f32[1024]", arg332_1: "f32[1024, 1, 7, 7]", arg333_1: "f32[1024]", arg334_1: "f32[1024]", arg335_1: "f32[1024]", arg336_1: "f32[4096, 1024]", arg337_1: "f32[4096]", arg338_1: "f32[1024, 4096]", arg339_1: "f32[1024]", arg340_1: "f32[1024]", arg341_1: "f32[1024]", arg342_1: "f32[1024]", arg343_1: "f32[1000, 1024]", arg344_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:482 in forward_features, code: x = self.stem(x)
        convolution_40: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:68 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_155: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1]);  convolution_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:72 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        clone_74: "f32[8, 72, 72, 128]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(clone_74, [3], correction = 0, keepdim = True)
        getitem_82: "f32[8, 72, 72, 1]" = var_mean_41[0]
        getitem_83: "f32[8, 72, 72, 1]" = var_mean_41[1];  var_mean_41 = None
        add_154: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
        rsqrt_41: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_41: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(clone_74, getitem_83);  clone_74 = getitem_83 = None
        mul_226: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_227: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_226, arg3_1);  mul_226 = arg3_1 = None
        add_155: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_227, arg4_1);  mul_227 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:73 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_156: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(add_155, [0, 3, 1, 2]);  add_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_41: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(permute_156, arg5_1, arg6_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg5_1 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_157: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(convolution_41, [0, 2, 3, 1]);  convolution_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_42 = torch.ops.aten.var_mean.correction(permute_157, [3], correction = 0, keepdim = True)
        getitem_84: "f32[8, 72, 72, 1]" = var_mean_42[0]
        getitem_85: "f32[8, 72, 72, 1]" = var_mean_42[1];  var_mean_42 = None
        add_156: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
        rsqrt_42: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_42: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(permute_157, getitem_85);  permute_157 = getitem_85 = None
        mul_228: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_229: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_228, arg7_1);  mul_228 = arg7_1 = None
        add_157: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_229, arg8_1);  mul_229 = arg8_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_181: "f32[41472, 128]" = torch.ops.aten.view.default(add_157, [41472, 128]);  add_157 = None
        permute_158: "f32[128, 512]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_73: "f32[41472, 512]" = torch.ops.aten.addmm.default(arg10_1, view_181, permute_158);  arg10_1 = view_181 = permute_158 = None
        view_182: "f32[8, 72, 72, 512]" = torch.ops.aten.view.default(addmm_73, [8, 72, 72, 512]);  addmm_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_230: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_182, 0.5)
        mul_231: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_182, 0.7071067811865476);  view_182 = None
        erf_36: "f32[8, 72, 72, 512]" = torch.ops.aten.erf.default(mul_231);  mul_231 = None
        add_158: "f32[8, 72, 72, 512]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_232: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(mul_230, add_158);  mul_230 = add_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_183: "f32[41472, 512]" = torch.ops.aten.view.default(mul_232, [41472, 512]);  mul_232 = None
        permute_159: "f32[512, 128]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_74: "f32[41472, 128]" = torch.ops.aten.addmm.default(arg12_1, view_183, permute_159);  arg12_1 = view_183 = permute_159 = None
        view_184: "f32[8, 72, 72, 128]" = torch.ops.aten.view.default(addmm_74, [8, 72, 72, 128]);  addmm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_160: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(view_184, [0, 3, 1, 2]);  view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_185: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(arg13_1, [1, -1, 1, 1]);  arg13_1 = None
        mul_233: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(permute_160, view_185);  permute_160 = view_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_159: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_233, permute_156);  mul_233 = permute_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_42: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(add_159, arg14_1, arg15_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg14_1 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_161: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(convolution_42, [0, 2, 3, 1]);  convolution_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_43 = torch.ops.aten.var_mean.correction(permute_161, [3], correction = 0, keepdim = True)
        getitem_86: "f32[8, 72, 72, 1]" = var_mean_43[0]
        getitem_87: "f32[8, 72, 72, 1]" = var_mean_43[1];  var_mean_43 = None
        add_160: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
        rsqrt_43: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        sub_43: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(permute_161, getitem_87);  permute_161 = getitem_87 = None
        mul_234: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_235: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_234, arg16_1);  mul_234 = arg16_1 = None
        add_161: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_235, arg17_1);  mul_235 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_186: "f32[41472, 128]" = torch.ops.aten.view.default(add_161, [41472, 128]);  add_161 = None
        permute_162: "f32[128, 512]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_75: "f32[41472, 512]" = torch.ops.aten.addmm.default(arg19_1, view_186, permute_162);  arg19_1 = view_186 = permute_162 = None
        view_187: "f32[8, 72, 72, 512]" = torch.ops.aten.view.default(addmm_75, [8, 72, 72, 512]);  addmm_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_236: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_187, 0.5)
        mul_237: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_187, 0.7071067811865476);  view_187 = None
        erf_37: "f32[8, 72, 72, 512]" = torch.ops.aten.erf.default(mul_237);  mul_237 = None
        add_162: "f32[8, 72, 72, 512]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_238: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(mul_236, add_162);  mul_236 = add_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_188: "f32[41472, 512]" = torch.ops.aten.view.default(mul_238, [41472, 512]);  mul_238 = None
        permute_163: "f32[512, 128]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_76: "f32[41472, 128]" = torch.ops.aten.addmm.default(arg21_1, view_188, permute_163);  arg21_1 = view_188 = permute_163 = None
        view_189: "f32[8, 72, 72, 128]" = torch.ops.aten.view.default(addmm_76, [8, 72, 72, 128]);  addmm_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_164: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(view_189, [0, 3, 1, 2]);  view_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_190: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(arg22_1, [1, -1, 1, 1]);  arg22_1 = None
        mul_239: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(permute_164, view_190);  permute_164 = view_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_163: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_239, add_159);  mul_239 = add_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_43: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(add_163, arg23_1, arg24_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg23_1 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_165: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_44 = torch.ops.aten.var_mean.correction(permute_165, [3], correction = 0, keepdim = True)
        getitem_88: "f32[8, 72, 72, 1]" = var_mean_44[0]
        getitem_89: "f32[8, 72, 72, 1]" = var_mean_44[1];  var_mean_44 = None
        add_164: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_44: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_44: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(permute_165, getitem_89);  permute_165 = getitem_89 = None
        mul_240: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_241: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_240, arg25_1);  mul_240 = arg25_1 = None
        add_165: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_241, arg26_1);  mul_241 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_191: "f32[41472, 128]" = torch.ops.aten.view.default(add_165, [41472, 128]);  add_165 = None
        permute_166: "f32[128, 512]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_77: "f32[41472, 512]" = torch.ops.aten.addmm.default(arg28_1, view_191, permute_166);  arg28_1 = view_191 = permute_166 = None
        view_192: "f32[8, 72, 72, 512]" = torch.ops.aten.view.default(addmm_77, [8, 72, 72, 512]);  addmm_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_242: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_192, 0.5)
        mul_243: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476);  view_192 = None
        erf_38: "f32[8, 72, 72, 512]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_166: "f32[8, 72, 72, 512]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_244: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(mul_242, add_166);  mul_242 = add_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_193: "f32[41472, 512]" = torch.ops.aten.view.default(mul_244, [41472, 512]);  mul_244 = None
        permute_167: "f32[512, 128]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_78: "f32[41472, 128]" = torch.ops.aten.addmm.default(arg30_1, view_193, permute_167);  arg30_1 = view_193 = permute_167 = None
        view_194: "f32[8, 72, 72, 128]" = torch.ops.aten.view.default(addmm_78, [8, 72, 72, 128]);  addmm_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_168: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(view_194, [0, 3, 1, 2]);  view_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_195: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(arg31_1, [1, -1, 1, 1]);  arg31_1 = None
        mul_245: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(permute_168, view_195);  permute_168 = view_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_167: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_245, add_163);  mul_245 = add_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:68 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_169: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(add_167, [0, 2, 3, 1]);  add_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:72 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_45 = torch.ops.aten.var_mean.correction(permute_169, [3], correction = 0, keepdim = True)
        getitem_90: "f32[8, 72, 72, 1]" = var_mean_45[0]
        getitem_91: "f32[8, 72, 72, 1]" = var_mean_45[1];  var_mean_45 = None
        add_168: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
        rsqrt_45: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        sub_45: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(permute_169, getitem_91);  permute_169 = getitem_91 = None
        mul_246: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_247: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_246, arg32_1);  mul_246 = arg32_1 = None
        add_169: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_247, arg33_1);  mul_247 = arg33_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:73 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_170: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(add_169, [0, 3, 1, 2]);  add_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:229 in forward, code: x = self.downsample(x)
        convolution_44: "f32[8, 256, 36, 36]" = torch.ops.aten.convolution.default(permute_170, arg34_1, arg35_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_170 = arg34_1 = arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_45: "f32[8, 256, 36, 36]" = torch.ops.aten.convolution.default(convolution_44, arg36_1, arg37_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg36_1 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_171: "f32[8, 36, 36, 256]" = torch.ops.aten.permute.default(convolution_45, [0, 2, 3, 1]);  convolution_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_46 = torch.ops.aten.var_mean.correction(permute_171, [3], correction = 0, keepdim = True)
        getitem_92: "f32[8, 36, 36, 1]" = var_mean_46[0]
        getitem_93: "f32[8, 36, 36, 1]" = var_mean_46[1];  var_mean_46 = None
        add_170: "f32[8, 36, 36, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
        rsqrt_46: "f32[8, 36, 36, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_46: "f32[8, 36, 36, 256]" = torch.ops.aten.sub.Tensor(permute_171, getitem_93);  permute_171 = getitem_93 = None
        mul_248: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_249: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(mul_248, arg38_1);  mul_248 = arg38_1 = None
        add_171: "f32[8, 36, 36, 256]" = torch.ops.aten.add.Tensor(mul_249, arg39_1);  mul_249 = arg39_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_196: "f32[10368, 256]" = torch.ops.aten.view.default(add_171, [10368, 256]);  add_171 = None
        permute_172: "f32[256, 1024]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_79: "f32[10368, 1024]" = torch.ops.aten.addmm.default(arg41_1, view_196, permute_172);  arg41_1 = view_196 = permute_172 = None
        view_197: "f32[8, 36, 36, 1024]" = torch.ops.aten.view.default(addmm_79, [8, 36, 36, 1024]);  addmm_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_250: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_251: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
        erf_39: "f32[8, 36, 36, 1024]" = torch.ops.aten.erf.default(mul_251);  mul_251 = None
        add_172: "f32[8, 36, 36, 1024]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_252: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(mul_250, add_172);  mul_250 = add_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_198: "f32[10368, 1024]" = torch.ops.aten.view.default(mul_252, [10368, 1024]);  mul_252 = None
        permute_173: "f32[1024, 256]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_80: "f32[10368, 256]" = torch.ops.aten.addmm.default(arg43_1, view_198, permute_173);  arg43_1 = view_198 = permute_173 = None
        view_199: "f32[8, 36, 36, 256]" = torch.ops.aten.view.default(addmm_80, [8, 36, 36, 256]);  addmm_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_174: "f32[8, 256, 36, 36]" = torch.ops.aten.permute.default(view_199, [0, 3, 1, 2]);  view_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_200: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(arg44_1, [1, -1, 1, 1]);  arg44_1 = None
        mul_253: "f32[8, 256, 36, 36]" = torch.ops.aten.mul.Tensor(permute_174, view_200);  permute_174 = view_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_173: "f32[8, 256, 36, 36]" = torch.ops.aten.add.Tensor(mul_253, convolution_44);  mul_253 = convolution_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_46: "f32[8, 256, 36, 36]" = torch.ops.aten.convolution.default(add_173, arg45_1, arg46_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg45_1 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_175: "f32[8, 36, 36, 256]" = torch.ops.aten.permute.default(convolution_46, [0, 2, 3, 1]);  convolution_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_47 = torch.ops.aten.var_mean.correction(permute_175, [3], correction = 0, keepdim = True)
        getitem_94: "f32[8, 36, 36, 1]" = var_mean_47[0]
        getitem_95: "f32[8, 36, 36, 1]" = var_mean_47[1];  var_mean_47 = None
        add_174: "f32[8, 36, 36, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_47: "f32[8, 36, 36, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_47: "f32[8, 36, 36, 256]" = torch.ops.aten.sub.Tensor(permute_175, getitem_95);  permute_175 = getitem_95 = None
        mul_254: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_255: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(mul_254, arg47_1);  mul_254 = arg47_1 = None
        add_175: "f32[8, 36, 36, 256]" = torch.ops.aten.add.Tensor(mul_255, arg48_1);  mul_255 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_201: "f32[10368, 256]" = torch.ops.aten.view.default(add_175, [10368, 256]);  add_175 = None
        permute_176: "f32[256, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_81: "f32[10368, 1024]" = torch.ops.aten.addmm.default(arg50_1, view_201, permute_176);  arg50_1 = view_201 = permute_176 = None
        view_202: "f32[8, 36, 36, 1024]" = torch.ops.aten.view.default(addmm_81, [8, 36, 36, 1024]);  addmm_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_256: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_202, 0.5)
        mul_257: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_202, 0.7071067811865476);  view_202 = None
        erf_40: "f32[8, 36, 36, 1024]" = torch.ops.aten.erf.default(mul_257);  mul_257 = None
        add_176: "f32[8, 36, 36, 1024]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_258: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(mul_256, add_176);  mul_256 = add_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_203: "f32[10368, 1024]" = torch.ops.aten.view.default(mul_258, [10368, 1024]);  mul_258 = None
        permute_177: "f32[1024, 256]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_82: "f32[10368, 256]" = torch.ops.aten.addmm.default(arg52_1, view_203, permute_177);  arg52_1 = view_203 = permute_177 = None
        view_204: "f32[8, 36, 36, 256]" = torch.ops.aten.view.default(addmm_82, [8, 36, 36, 256]);  addmm_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_178: "f32[8, 256, 36, 36]" = torch.ops.aten.permute.default(view_204, [0, 3, 1, 2]);  view_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_205: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(arg53_1, [1, -1, 1, 1]);  arg53_1 = None
        mul_259: "f32[8, 256, 36, 36]" = torch.ops.aten.mul.Tensor(permute_178, view_205);  permute_178 = view_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_177: "f32[8, 256, 36, 36]" = torch.ops.aten.add.Tensor(mul_259, add_173);  mul_259 = add_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_47: "f32[8, 256, 36, 36]" = torch.ops.aten.convolution.default(add_177, arg54_1, arg55_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg54_1 = arg55_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_179: "f32[8, 36, 36, 256]" = torch.ops.aten.permute.default(convolution_47, [0, 2, 3, 1]);  convolution_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_48 = torch.ops.aten.var_mean.correction(permute_179, [3], correction = 0, keepdim = True)
        getitem_96: "f32[8, 36, 36, 1]" = var_mean_48[0]
        getitem_97: "f32[8, 36, 36, 1]" = var_mean_48[1];  var_mean_48 = None
        add_178: "f32[8, 36, 36, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_48: "f32[8, 36, 36, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_48: "f32[8, 36, 36, 256]" = torch.ops.aten.sub.Tensor(permute_179, getitem_97);  permute_179 = getitem_97 = None
        mul_260: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_261: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(mul_260, arg56_1);  mul_260 = arg56_1 = None
        add_179: "f32[8, 36, 36, 256]" = torch.ops.aten.add.Tensor(mul_261, arg57_1);  mul_261 = arg57_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_206: "f32[10368, 256]" = torch.ops.aten.view.default(add_179, [10368, 256]);  add_179 = None
        permute_180: "f32[256, 1024]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_83: "f32[10368, 1024]" = torch.ops.aten.addmm.default(arg59_1, view_206, permute_180);  arg59_1 = view_206 = permute_180 = None
        view_207: "f32[8, 36, 36, 1024]" = torch.ops.aten.view.default(addmm_83, [8, 36, 36, 1024]);  addmm_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_262: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_207, 0.5)
        mul_263: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_207, 0.7071067811865476);  view_207 = None
        erf_41: "f32[8, 36, 36, 1024]" = torch.ops.aten.erf.default(mul_263);  mul_263 = None
        add_180: "f32[8, 36, 36, 1024]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_264: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(mul_262, add_180);  mul_262 = add_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_208: "f32[10368, 1024]" = torch.ops.aten.view.default(mul_264, [10368, 1024]);  mul_264 = None
        permute_181: "f32[1024, 256]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_84: "f32[10368, 256]" = torch.ops.aten.addmm.default(arg61_1, view_208, permute_181);  arg61_1 = view_208 = permute_181 = None
        view_209: "f32[8, 36, 36, 256]" = torch.ops.aten.view.default(addmm_84, [8, 36, 36, 256]);  addmm_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_182: "f32[8, 256, 36, 36]" = torch.ops.aten.permute.default(view_209, [0, 3, 1, 2]);  view_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_210: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(arg62_1, [1, -1, 1, 1]);  arg62_1 = None
        mul_265: "f32[8, 256, 36, 36]" = torch.ops.aten.mul.Tensor(permute_182, view_210);  permute_182 = view_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_181: "f32[8, 256, 36, 36]" = torch.ops.aten.add.Tensor(mul_265, add_177);  mul_265 = add_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:68 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_183: "f32[8, 36, 36, 256]" = torch.ops.aten.permute.default(add_181, [0, 2, 3, 1]);  add_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:72 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_49 = torch.ops.aten.var_mean.correction(permute_183, [3], correction = 0, keepdim = True)
        getitem_98: "f32[8, 36, 36, 1]" = var_mean_49[0]
        getitem_99: "f32[8, 36, 36, 1]" = var_mean_49[1];  var_mean_49 = None
        add_182: "f32[8, 36, 36, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-06);  getitem_98 = None
        rsqrt_49: "f32[8, 36, 36, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_49: "f32[8, 36, 36, 256]" = torch.ops.aten.sub.Tensor(permute_183, getitem_99);  permute_183 = getitem_99 = None
        mul_266: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_267: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(mul_266, arg63_1);  mul_266 = arg63_1 = None
        add_183: "f32[8, 36, 36, 256]" = torch.ops.aten.add.Tensor(mul_267, arg64_1);  mul_267 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:73 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_184: "f32[8, 256, 36, 36]" = torch.ops.aten.permute.default(add_183, [0, 3, 1, 2]);  add_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:229 in forward, code: x = self.downsample(x)
        convolution_48: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(permute_184, arg65_1, arg66_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_184 = arg65_1 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_49: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(convolution_48, arg67_1, arg68_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg67_1 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_185: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_49, [0, 2, 3, 1]);  convolution_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_50 = torch.ops.aten.var_mean.correction(permute_185, [3], correction = 0, keepdim = True)
        getitem_100: "f32[8, 18, 18, 1]" = var_mean_50[0]
        getitem_101: "f32[8, 18, 18, 1]" = var_mean_50[1];  var_mean_50 = None
        add_184: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
        rsqrt_50: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        sub_50: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_185, getitem_101);  permute_185 = getitem_101 = None
        mul_268: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        mul_269: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_268, arg69_1);  mul_268 = arg69_1 = None
        add_185: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_269, arg70_1);  mul_269 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_211: "f32[2592, 512]" = torch.ops.aten.view.default(add_185, [2592, 512]);  add_185 = None
        permute_186: "f32[512, 2048]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_85: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg72_1, view_211, permute_186);  arg72_1 = view_211 = permute_186 = None
        view_212: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_85, [8, 18, 18, 2048]);  addmm_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_270: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_212, 0.5)
        mul_271: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_212, 0.7071067811865476);  view_212 = None
        erf_42: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_271);  mul_271 = None
        add_186: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_272: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_270, add_186);  mul_270 = add_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_213: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_272, [2592, 2048]);  mul_272 = None
        permute_187: "f32[2048, 512]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_86: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg74_1, view_213, permute_187);  arg74_1 = view_213 = permute_187 = None
        view_214: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_86, [8, 18, 18, 512]);  addmm_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_188: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_214, [0, 3, 1, 2]);  view_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_215: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg75_1, [1, -1, 1, 1]);  arg75_1 = None
        mul_273: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_188, view_215);  permute_188 = view_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_187: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_273, convolution_48);  mul_273 = convolution_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_50: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_187, arg76_1, arg77_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg76_1 = arg77_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_189: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_50, [0, 2, 3, 1]);  convolution_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_51 = torch.ops.aten.var_mean.correction(permute_189, [3], correction = 0, keepdim = True)
        getitem_102: "f32[8, 18, 18, 1]" = var_mean_51[0]
        getitem_103: "f32[8, 18, 18, 1]" = var_mean_51[1];  var_mean_51 = None
        add_188: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
        rsqrt_51: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_51: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_189, getitem_103);  permute_189 = getitem_103 = None
        mul_274: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        mul_275: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_274, arg78_1);  mul_274 = arg78_1 = None
        add_189: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_275, arg79_1);  mul_275 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_216: "f32[2592, 512]" = torch.ops.aten.view.default(add_189, [2592, 512]);  add_189 = None
        permute_190: "f32[512, 2048]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_87: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg81_1, view_216, permute_190);  arg81_1 = view_216 = permute_190 = None
        view_217: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_87, [8, 18, 18, 2048]);  addmm_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_276: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_277: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_43: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_277);  mul_277 = None
        add_190: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_278: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_276, add_190);  mul_276 = add_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_218: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_278, [2592, 2048]);  mul_278 = None
        permute_191: "f32[2048, 512]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_88: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg83_1, view_218, permute_191);  arg83_1 = view_218 = permute_191 = None
        view_219: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_88, [8, 18, 18, 512]);  addmm_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_192: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_219, [0, 3, 1, 2]);  view_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_220: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg84_1, [1, -1, 1, 1]);  arg84_1 = None
        mul_279: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_192, view_220);  permute_192 = view_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_191: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_279, add_187);  mul_279 = add_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_51: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_191, arg85_1, arg86_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg85_1 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_193: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_51, [0, 2, 3, 1]);  convolution_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_52 = torch.ops.aten.var_mean.correction(permute_193, [3], correction = 0, keepdim = True)
        getitem_104: "f32[8, 18, 18, 1]" = var_mean_52[0]
        getitem_105: "f32[8, 18, 18, 1]" = var_mean_52[1];  var_mean_52 = None
        add_192: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
        rsqrt_52: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_52: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_193, getitem_105);  permute_193 = getitem_105 = None
        mul_280: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        mul_281: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_280, arg87_1);  mul_280 = arg87_1 = None
        add_193: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_281, arg88_1);  mul_281 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_221: "f32[2592, 512]" = torch.ops.aten.view.default(add_193, [2592, 512]);  add_193 = None
        permute_194: "f32[512, 2048]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_89: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg90_1, view_221, permute_194);  arg90_1 = view_221 = permute_194 = None
        view_222: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_89, [8, 18, 18, 2048]);  addmm_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_282: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_222, 0.5)
        mul_283: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_222, 0.7071067811865476);  view_222 = None
        erf_44: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_283);  mul_283 = None
        add_194: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
        mul_284: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_282, add_194);  mul_282 = add_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_223: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_284, [2592, 2048]);  mul_284 = None
        permute_195: "f32[2048, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_90: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg92_1, view_223, permute_195);  arg92_1 = view_223 = permute_195 = None
        view_224: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_90, [8, 18, 18, 512]);  addmm_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_196: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_224, [0, 3, 1, 2]);  view_224 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_225: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg93_1, [1, -1, 1, 1]);  arg93_1 = None
        mul_285: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_196, view_225);  permute_196 = view_225 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_195: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_285, add_191);  mul_285 = add_191 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_52: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_195, arg94_1, arg95_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg94_1 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_197: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_52, [0, 2, 3, 1]);  convolution_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_53 = torch.ops.aten.var_mean.correction(permute_197, [3], correction = 0, keepdim = True)
        getitem_106: "f32[8, 18, 18, 1]" = var_mean_53[0]
        getitem_107: "f32[8, 18, 18, 1]" = var_mean_53[1];  var_mean_53 = None
        add_196: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
        rsqrt_53: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_53: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_197, getitem_107);  permute_197 = getitem_107 = None
        mul_286: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
        mul_287: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_286, arg96_1);  mul_286 = arg96_1 = None
        add_197: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_287, arg97_1);  mul_287 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_226: "f32[2592, 512]" = torch.ops.aten.view.default(add_197, [2592, 512]);  add_197 = None
        permute_198: "f32[512, 2048]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_91: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg99_1, view_226, permute_198);  arg99_1 = view_226 = permute_198 = None
        view_227: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_91, [8, 18, 18, 2048]);  addmm_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_288: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_227, 0.5)
        mul_289: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476);  view_227 = None
        erf_45: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_289);  mul_289 = None
        add_198: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
        mul_290: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_288, add_198);  mul_288 = add_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_228: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_290, [2592, 2048]);  mul_290 = None
        permute_199: "f32[2048, 512]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_92: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg101_1, view_228, permute_199);  arg101_1 = view_228 = permute_199 = None
        view_229: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_92, [8, 18, 18, 512]);  addmm_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_200: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_229, [0, 3, 1, 2]);  view_229 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_230: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg102_1, [1, -1, 1, 1]);  arg102_1 = None
        mul_291: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_200, view_230);  permute_200 = view_230 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_199: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_291, add_195);  mul_291 = add_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_53: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_199, arg103_1, arg104_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg103_1 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_201: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_53, [0, 2, 3, 1]);  convolution_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_54 = torch.ops.aten.var_mean.correction(permute_201, [3], correction = 0, keepdim = True)
        getitem_108: "f32[8, 18, 18, 1]" = var_mean_54[0]
        getitem_109: "f32[8, 18, 18, 1]" = var_mean_54[1];  var_mean_54 = None
        add_200: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
        rsqrt_54: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        sub_54: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_201, getitem_109);  permute_201 = getitem_109 = None
        mul_292: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
        mul_293: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_292, arg105_1);  mul_292 = arg105_1 = None
        add_201: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_293, arg106_1);  mul_293 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_231: "f32[2592, 512]" = torch.ops.aten.view.default(add_201, [2592, 512]);  add_201 = None
        permute_202: "f32[512, 2048]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_93: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg108_1, view_231, permute_202);  arg108_1 = view_231 = permute_202 = None
        view_232: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_93, [8, 18, 18, 2048]);  addmm_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_294: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_232, 0.5)
        mul_295: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476);  view_232 = None
        erf_46: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_295);  mul_295 = None
        add_202: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
        mul_296: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_294, add_202);  mul_294 = add_202 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_233: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_296, [2592, 2048]);  mul_296 = None
        permute_203: "f32[2048, 512]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_94: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg110_1, view_233, permute_203);  arg110_1 = view_233 = permute_203 = None
        view_234: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_94, [8, 18, 18, 512]);  addmm_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_204: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_234, [0, 3, 1, 2]);  view_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_235: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg111_1, [1, -1, 1, 1]);  arg111_1 = None
        mul_297: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_204, view_235);  permute_204 = view_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_203: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_297, add_199);  mul_297 = add_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_54: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_203, arg112_1, arg113_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg112_1 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_205: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_54, [0, 2, 3, 1]);  convolution_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_55 = torch.ops.aten.var_mean.correction(permute_205, [3], correction = 0, keepdim = True)
        getitem_110: "f32[8, 18, 18, 1]" = var_mean_55[0]
        getitem_111: "f32[8, 18, 18, 1]" = var_mean_55[1];  var_mean_55 = None
        add_204: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_55: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        sub_55: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_205, getitem_111);  permute_205 = getitem_111 = None
        mul_298: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
        mul_299: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_298, arg114_1);  mul_298 = arg114_1 = None
        add_205: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_299, arg115_1);  mul_299 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_236: "f32[2592, 512]" = torch.ops.aten.view.default(add_205, [2592, 512]);  add_205 = None
        permute_206: "f32[512, 2048]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_95: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg117_1, view_236, permute_206);  arg117_1 = view_236 = permute_206 = None
        view_237: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_95, [8, 18, 18, 2048]);  addmm_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_300: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_237, 0.5)
        mul_301: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_237, 0.7071067811865476);  view_237 = None
        erf_47: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_301);  mul_301 = None
        add_206: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
        mul_302: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_300, add_206);  mul_300 = add_206 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_238: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_302, [2592, 2048]);  mul_302 = None
        permute_207: "f32[2048, 512]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_96: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg119_1, view_238, permute_207);  arg119_1 = view_238 = permute_207 = None
        view_239: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_96, [8, 18, 18, 512]);  addmm_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_208: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_239, [0, 3, 1, 2]);  view_239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_240: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg120_1, [1, -1, 1, 1]);  arg120_1 = None
        mul_303: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_208, view_240);  permute_208 = view_240 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_207: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_303, add_203);  mul_303 = add_203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_55: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_207, arg121_1, arg122_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg121_1 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_209: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_55, [0, 2, 3, 1]);  convolution_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_56 = torch.ops.aten.var_mean.correction(permute_209, [3], correction = 0, keepdim = True)
        getitem_112: "f32[8, 18, 18, 1]" = var_mean_56[0]
        getitem_113: "f32[8, 18, 18, 1]" = var_mean_56[1];  var_mean_56 = None
        add_208: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
        rsqrt_56: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        sub_56: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_209, getitem_113);  permute_209 = getitem_113 = None
        mul_304: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
        mul_305: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_304, arg123_1);  mul_304 = arg123_1 = None
        add_209: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_305, arg124_1);  mul_305 = arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_241: "f32[2592, 512]" = torch.ops.aten.view.default(add_209, [2592, 512]);  add_209 = None
        permute_210: "f32[512, 2048]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_97: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg126_1, view_241, permute_210);  arg126_1 = view_241 = permute_210 = None
        view_242: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_97, [8, 18, 18, 2048]);  addmm_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_306: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_242, 0.5)
        mul_307: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_242, 0.7071067811865476);  view_242 = None
        erf_48: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_307);  mul_307 = None
        add_210: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
        mul_308: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_306, add_210);  mul_306 = add_210 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_243: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_308, [2592, 2048]);  mul_308 = None
        permute_211: "f32[2048, 512]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_98: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg128_1, view_243, permute_211);  arg128_1 = view_243 = permute_211 = None
        view_244: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_98, [8, 18, 18, 512]);  addmm_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_212: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_244, [0, 3, 1, 2]);  view_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_245: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg129_1, [1, -1, 1, 1]);  arg129_1 = None
        mul_309: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_212, view_245);  permute_212 = view_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_211: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_309, add_207);  mul_309 = add_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_56: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_211, arg130_1, arg131_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg130_1 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_213: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_56, [0, 2, 3, 1]);  convolution_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_57 = torch.ops.aten.var_mean.correction(permute_213, [3], correction = 0, keepdim = True)
        getitem_114: "f32[8, 18, 18, 1]" = var_mean_57[0]
        getitem_115: "f32[8, 18, 18, 1]" = var_mean_57[1];  var_mean_57 = None
        add_212: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
        rsqrt_57: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_57: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_213, getitem_115);  permute_213 = getitem_115 = None
        mul_310: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        mul_311: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_310, arg132_1);  mul_310 = arg132_1 = None
        add_213: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_311, arg133_1);  mul_311 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_246: "f32[2592, 512]" = torch.ops.aten.view.default(add_213, [2592, 512]);  add_213 = None
        permute_214: "f32[512, 2048]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_99: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg135_1, view_246, permute_214);  arg135_1 = view_246 = permute_214 = None
        view_247: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_99, [8, 18, 18, 2048]);  addmm_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_312: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.5)
        mul_313: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.7071067811865476);  view_247 = None
        erf_49: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_313);  mul_313 = None
        add_214: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
        mul_314: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_312, add_214);  mul_312 = add_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_248: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_314, [2592, 2048]);  mul_314 = None
        permute_215: "f32[2048, 512]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_100: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg137_1, view_248, permute_215);  arg137_1 = view_248 = permute_215 = None
        view_249: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_100, [8, 18, 18, 512]);  addmm_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_216: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_249, [0, 3, 1, 2]);  view_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_250: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg138_1, [1, -1, 1, 1]);  arg138_1 = None
        mul_315: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_216, view_250);  permute_216 = view_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_215: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_315, add_211);  mul_315 = add_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_57: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_215, arg139_1, arg140_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg139_1 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_217: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_57, [0, 2, 3, 1]);  convolution_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_58 = torch.ops.aten.var_mean.correction(permute_217, [3], correction = 0, keepdim = True)
        getitem_116: "f32[8, 18, 18, 1]" = var_mean_58[0]
        getitem_117: "f32[8, 18, 18, 1]" = var_mean_58[1];  var_mean_58 = None
        add_216: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
        rsqrt_58: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_58: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_217, getitem_117);  permute_217 = getitem_117 = None
        mul_316: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        mul_317: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_316, arg141_1);  mul_316 = arg141_1 = None
        add_217: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_317, arg142_1);  mul_317 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_251: "f32[2592, 512]" = torch.ops.aten.view.default(add_217, [2592, 512]);  add_217 = None
        permute_218: "f32[512, 2048]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_101: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg144_1, view_251, permute_218);  arg144_1 = view_251 = permute_218 = None
        view_252: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_101, [8, 18, 18, 2048]);  addmm_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_318: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_252, 0.5)
        mul_319: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_252, 0.7071067811865476);  view_252 = None
        erf_50: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_319);  mul_319 = None
        add_218: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
        mul_320: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_318, add_218);  mul_318 = add_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_253: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_320, [2592, 2048]);  mul_320 = None
        permute_219: "f32[2048, 512]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_102: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg146_1, view_253, permute_219);  arg146_1 = view_253 = permute_219 = None
        view_254: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_102, [8, 18, 18, 512]);  addmm_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_220: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_254, [0, 3, 1, 2]);  view_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_255: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg147_1, [1, -1, 1, 1]);  arg147_1 = None
        mul_321: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_220, view_255);  permute_220 = view_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_219: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_321, add_215);  mul_321 = add_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_58: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_219, arg148_1, arg149_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg148_1 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_221: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_58, [0, 2, 3, 1]);  convolution_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_59 = torch.ops.aten.var_mean.correction(permute_221, [3], correction = 0, keepdim = True)
        getitem_118: "f32[8, 18, 18, 1]" = var_mean_59[0]
        getitem_119: "f32[8, 18, 18, 1]" = var_mean_59[1];  var_mean_59 = None
        add_220: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
        rsqrt_59: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        sub_59: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_221, getitem_119);  permute_221 = getitem_119 = None
        mul_322: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        mul_323: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_322, arg150_1);  mul_322 = arg150_1 = None
        add_221: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_323, arg151_1);  mul_323 = arg151_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_256: "f32[2592, 512]" = torch.ops.aten.view.default(add_221, [2592, 512]);  add_221 = None
        permute_222: "f32[512, 2048]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_103: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg153_1, view_256, permute_222);  arg153_1 = view_256 = permute_222 = None
        view_257: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_103, [8, 18, 18, 2048]);  addmm_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_324: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_257, 0.5)
        mul_325: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_257, 0.7071067811865476);  view_257 = None
        erf_51: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_325);  mul_325 = None
        add_222: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
        mul_326: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_324, add_222);  mul_324 = add_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_258: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_326, [2592, 2048]);  mul_326 = None
        permute_223: "f32[2048, 512]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_104: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg155_1, view_258, permute_223);  arg155_1 = view_258 = permute_223 = None
        view_259: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_104, [8, 18, 18, 512]);  addmm_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_224: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_259, [0, 3, 1, 2]);  view_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_260: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg156_1, [1, -1, 1, 1]);  arg156_1 = None
        mul_327: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_224, view_260);  permute_224 = view_260 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_223: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_327, add_219);  mul_327 = add_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_59: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_223, arg157_1, arg158_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg157_1 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_225: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_59, [0, 2, 3, 1]);  convolution_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_60 = torch.ops.aten.var_mean.correction(permute_225, [3], correction = 0, keepdim = True)
        getitem_120: "f32[8, 18, 18, 1]" = var_mean_60[0]
        getitem_121: "f32[8, 18, 18, 1]" = var_mean_60[1];  var_mean_60 = None
        add_224: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
        rsqrt_60: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        sub_60: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_225, getitem_121);  permute_225 = getitem_121 = None
        mul_328: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        mul_329: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_328, arg159_1);  mul_328 = arg159_1 = None
        add_225: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_329, arg160_1);  mul_329 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_261: "f32[2592, 512]" = torch.ops.aten.view.default(add_225, [2592, 512]);  add_225 = None
        permute_226: "f32[512, 2048]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_105: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg162_1, view_261, permute_226);  arg162_1 = view_261 = permute_226 = None
        view_262: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_105, [8, 18, 18, 2048]);  addmm_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_330: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_262, 0.5)
        mul_331: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_262, 0.7071067811865476);  view_262 = None
        erf_52: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_331);  mul_331 = None
        add_226: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_332: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_330, add_226);  mul_330 = add_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_263: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_332, [2592, 2048]);  mul_332 = None
        permute_227: "f32[2048, 512]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_106: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg164_1, view_263, permute_227);  arg164_1 = view_263 = permute_227 = None
        view_264: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_106, [8, 18, 18, 512]);  addmm_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_228: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_264, [0, 3, 1, 2]);  view_264 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_265: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg165_1, [1, -1, 1, 1]);  arg165_1 = None
        mul_333: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_228, view_265);  permute_228 = view_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_227: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_333, add_223);  mul_333 = add_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_60: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_227, arg166_1, arg167_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg166_1 = arg167_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_229: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_60, [0, 2, 3, 1]);  convolution_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_61 = torch.ops.aten.var_mean.correction(permute_229, [3], correction = 0, keepdim = True)
        getitem_122: "f32[8, 18, 18, 1]" = var_mean_61[0]
        getitem_123: "f32[8, 18, 18, 1]" = var_mean_61[1];  var_mean_61 = None
        add_228: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
        rsqrt_61: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        sub_61: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_229, getitem_123);  permute_229 = getitem_123 = None
        mul_334: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        mul_335: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_334, arg168_1);  mul_334 = arg168_1 = None
        add_229: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_335, arg169_1);  mul_335 = arg169_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_266: "f32[2592, 512]" = torch.ops.aten.view.default(add_229, [2592, 512]);  add_229 = None
        permute_230: "f32[512, 2048]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_107: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg171_1, view_266, permute_230);  arg171_1 = view_266 = permute_230 = None
        view_267: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_107, [8, 18, 18, 2048]);  addmm_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_336: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
        mul_337: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476);  view_267 = None
        erf_53: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_337);  mul_337 = None
        add_230: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_338: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_336, add_230);  mul_336 = add_230 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_268: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_338, [2592, 2048]);  mul_338 = None
        permute_231: "f32[2048, 512]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_108: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg173_1, view_268, permute_231);  arg173_1 = view_268 = permute_231 = None
        view_269: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_108, [8, 18, 18, 512]);  addmm_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_232: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_269, [0, 3, 1, 2]);  view_269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_270: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg174_1, [1, -1, 1, 1]);  arg174_1 = None
        mul_339: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_232, view_270);  permute_232 = view_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_231: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_339, add_227);  mul_339 = add_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_61: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_231, arg175_1, arg176_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg175_1 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_233: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_61, [0, 2, 3, 1]);  convolution_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_62 = torch.ops.aten.var_mean.correction(permute_233, [3], correction = 0, keepdim = True)
        getitem_124: "f32[8, 18, 18, 1]" = var_mean_62[0]
        getitem_125: "f32[8, 18, 18, 1]" = var_mean_62[1];  var_mean_62 = None
        add_232: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
        rsqrt_62: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        sub_62: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_233, getitem_125);  permute_233 = getitem_125 = None
        mul_340: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        mul_341: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_340, arg177_1);  mul_340 = arg177_1 = None
        add_233: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_341, arg178_1);  mul_341 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_271: "f32[2592, 512]" = torch.ops.aten.view.default(add_233, [2592, 512]);  add_233 = None
        permute_234: "f32[512, 2048]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_109: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg180_1, view_271, permute_234);  arg180_1 = view_271 = permute_234 = None
        view_272: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_109, [8, 18, 18, 2048]);  addmm_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_342: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_272, 0.5)
        mul_343: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_272, 0.7071067811865476);  view_272 = None
        erf_54: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_343);  mul_343 = None
        add_234: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_344: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_342, add_234);  mul_342 = add_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_273: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_344, [2592, 2048]);  mul_344 = None
        permute_235: "f32[2048, 512]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_110: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg182_1, view_273, permute_235);  arg182_1 = view_273 = permute_235 = None
        view_274: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_110, [8, 18, 18, 512]);  addmm_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_236: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_274, [0, 3, 1, 2]);  view_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_275: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg183_1, [1, -1, 1, 1]);  arg183_1 = None
        mul_345: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_236, view_275);  permute_236 = view_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_235: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_345, add_231);  mul_345 = add_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_62: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_235, arg184_1, arg185_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg184_1 = arg185_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_237: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_62, [0, 2, 3, 1]);  convolution_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_63 = torch.ops.aten.var_mean.correction(permute_237, [3], correction = 0, keepdim = True)
        getitem_126: "f32[8, 18, 18, 1]" = var_mean_63[0]
        getitem_127: "f32[8, 18, 18, 1]" = var_mean_63[1];  var_mean_63 = None
        add_236: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
        rsqrt_63: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_63: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_237, getitem_127);  permute_237 = getitem_127 = None
        mul_346: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        mul_347: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_346, arg186_1);  mul_346 = arg186_1 = None
        add_237: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_347, arg187_1);  mul_347 = arg187_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_276: "f32[2592, 512]" = torch.ops.aten.view.default(add_237, [2592, 512]);  add_237 = None
        permute_238: "f32[512, 2048]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_111: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg189_1, view_276, permute_238);  arg189_1 = view_276 = permute_238 = None
        view_277: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_111, [8, 18, 18, 2048]);  addmm_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_348: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_277, 0.5)
        mul_349: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_277, 0.7071067811865476);  view_277 = None
        erf_55: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_349);  mul_349 = None
        add_238: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_350: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_348, add_238);  mul_348 = add_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_278: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_350, [2592, 2048]);  mul_350 = None
        permute_239: "f32[2048, 512]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_112: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg191_1, view_278, permute_239);  arg191_1 = view_278 = permute_239 = None
        view_279: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_112, [8, 18, 18, 512]);  addmm_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_240: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_279, [0, 3, 1, 2]);  view_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_280: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg192_1, [1, -1, 1, 1]);  arg192_1 = None
        mul_351: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_240, view_280);  permute_240 = view_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_239: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_351, add_235);  mul_351 = add_235 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_63: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_239, arg193_1, arg194_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg193_1 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_241: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_63, [0, 2, 3, 1]);  convolution_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_64 = torch.ops.aten.var_mean.correction(permute_241, [3], correction = 0, keepdim = True)
        getitem_128: "f32[8, 18, 18, 1]" = var_mean_64[0]
        getitem_129: "f32[8, 18, 18, 1]" = var_mean_64[1];  var_mean_64 = None
        add_240: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
        rsqrt_64: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_64: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_241, getitem_129);  permute_241 = getitem_129 = None
        mul_352: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        mul_353: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_352, arg195_1);  mul_352 = arg195_1 = None
        add_241: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_353, arg196_1);  mul_353 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_281: "f32[2592, 512]" = torch.ops.aten.view.default(add_241, [2592, 512]);  add_241 = None
        permute_242: "f32[512, 2048]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_113: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg198_1, view_281, permute_242);  arg198_1 = view_281 = permute_242 = None
        view_282: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_113, [8, 18, 18, 2048]);  addmm_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_354: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_282, 0.5)
        mul_355: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_282, 0.7071067811865476);  view_282 = None
        erf_56: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_355);  mul_355 = None
        add_242: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_356: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_354, add_242);  mul_354 = add_242 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_283: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_356, [2592, 2048]);  mul_356 = None
        permute_243: "f32[2048, 512]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_114: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg200_1, view_283, permute_243);  arg200_1 = view_283 = permute_243 = None
        view_284: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_114, [8, 18, 18, 512]);  addmm_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_244: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_284, [0, 3, 1, 2]);  view_284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_285: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg201_1, [1, -1, 1, 1]);  arg201_1 = None
        mul_357: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_244, view_285);  permute_244 = view_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_243: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_357, add_239);  mul_357 = add_239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_64: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_243, arg202_1, arg203_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg202_1 = arg203_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_245: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_64, [0, 2, 3, 1]);  convolution_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_65 = torch.ops.aten.var_mean.correction(permute_245, [3], correction = 0, keepdim = True)
        getitem_130: "f32[8, 18, 18, 1]" = var_mean_65[0]
        getitem_131: "f32[8, 18, 18, 1]" = var_mean_65[1];  var_mean_65 = None
        add_244: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
        rsqrt_65: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_65: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_245, getitem_131);  permute_245 = getitem_131 = None
        mul_358: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        mul_359: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_358, arg204_1);  mul_358 = arg204_1 = None
        add_245: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_359, arg205_1);  mul_359 = arg205_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_286: "f32[2592, 512]" = torch.ops.aten.view.default(add_245, [2592, 512]);  add_245 = None
        permute_246: "f32[512, 2048]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_115: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg207_1, view_286, permute_246);  arg207_1 = view_286 = permute_246 = None
        view_287: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_115, [8, 18, 18, 2048]);  addmm_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_360: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_287, 0.5)
        mul_361: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_287, 0.7071067811865476);  view_287 = None
        erf_57: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_361);  mul_361 = None
        add_246: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_362: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_360, add_246);  mul_360 = add_246 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_288: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_362, [2592, 2048]);  mul_362 = None
        permute_247: "f32[2048, 512]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_116: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg209_1, view_288, permute_247);  arg209_1 = view_288 = permute_247 = None
        view_289: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_116, [8, 18, 18, 512]);  addmm_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_248: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_289, [0, 3, 1, 2]);  view_289 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_290: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg210_1, [1, -1, 1, 1]);  arg210_1 = None
        mul_363: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_248, view_290);  permute_248 = view_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_247: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_363, add_243);  mul_363 = add_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_65: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_247, arg211_1, arg212_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg211_1 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_249: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_65, [0, 2, 3, 1]);  convolution_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_66 = torch.ops.aten.var_mean.correction(permute_249, [3], correction = 0, keepdim = True)
        getitem_132: "f32[8, 18, 18, 1]" = var_mean_66[0]
        getitem_133: "f32[8, 18, 18, 1]" = var_mean_66[1];  var_mean_66 = None
        add_248: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_66: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
        sub_66: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_249, getitem_133);  permute_249 = getitem_133 = None
        mul_364: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        mul_365: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_364, arg213_1);  mul_364 = arg213_1 = None
        add_249: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_365, arg214_1);  mul_365 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_291: "f32[2592, 512]" = torch.ops.aten.view.default(add_249, [2592, 512]);  add_249 = None
        permute_250: "f32[512, 2048]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_117: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg216_1, view_291, permute_250);  arg216_1 = view_291 = permute_250 = None
        view_292: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_117, [8, 18, 18, 2048]);  addmm_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_366: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_292, 0.5)
        mul_367: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_292, 0.7071067811865476);  view_292 = None
        erf_58: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_367);  mul_367 = None
        add_250: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_368: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_366, add_250);  mul_366 = add_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_293: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_368, [2592, 2048]);  mul_368 = None
        permute_251: "f32[2048, 512]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_118: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg218_1, view_293, permute_251);  arg218_1 = view_293 = permute_251 = None
        view_294: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_118, [8, 18, 18, 512]);  addmm_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_252: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_294, [0, 3, 1, 2]);  view_294 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_295: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg219_1, [1, -1, 1, 1]);  arg219_1 = None
        mul_369: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_252, view_295);  permute_252 = view_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_251: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_369, add_247);  mul_369 = add_247 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_66: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_251, arg220_1, arg221_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg220_1 = arg221_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_253: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_66, [0, 2, 3, 1]);  convolution_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_67 = torch.ops.aten.var_mean.correction(permute_253, [3], correction = 0, keepdim = True)
        getitem_134: "f32[8, 18, 18, 1]" = var_mean_67[0]
        getitem_135: "f32[8, 18, 18, 1]" = var_mean_67[1];  var_mean_67 = None
        add_252: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
        rsqrt_67: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        sub_67: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_253, getitem_135);  permute_253 = getitem_135 = None
        mul_370: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        mul_371: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_370, arg222_1);  mul_370 = arg222_1 = None
        add_253: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_371, arg223_1);  mul_371 = arg223_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_296: "f32[2592, 512]" = torch.ops.aten.view.default(add_253, [2592, 512]);  add_253 = None
        permute_254: "f32[512, 2048]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_119: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg225_1, view_296, permute_254);  arg225_1 = view_296 = permute_254 = None
        view_297: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_119, [8, 18, 18, 2048]);  addmm_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_372: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
        mul_373: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_297, 0.7071067811865476);  view_297 = None
        erf_59: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_373);  mul_373 = None
        add_254: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_374: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_372, add_254);  mul_372 = add_254 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_298: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_374, [2592, 2048]);  mul_374 = None
        permute_255: "f32[2048, 512]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
        addmm_120: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg227_1, view_298, permute_255);  arg227_1 = view_298 = permute_255 = None
        view_299: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_120, [8, 18, 18, 512]);  addmm_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_256: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_299, [0, 3, 1, 2]);  view_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_300: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg228_1, [1, -1, 1, 1]);  arg228_1 = None
        mul_375: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_256, view_300);  permute_256 = view_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_255: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_375, add_251);  mul_375 = add_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_67: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_255, arg229_1, arg230_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg229_1 = arg230_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_257: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_67, [0, 2, 3, 1]);  convolution_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_68 = torch.ops.aten.var_mean.correction(permute_257, [3], correction = 0, keepdim = True)
        getitem_136: "f32[8, 18, 18, 1]" = var_mean_68[0]
        getitem_137: "f32[8, 18, 18, 1]" = var_mean_68[1];  var_mean_68 = None
        add_256: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
        rsqrt_68: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        sub_68: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_257, getitem_137);  permute_257 = getitem_137 = None
        mul_376: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        mul_377: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_376, arg231_1);  mul_376 = arg231_1 = None
        add_257: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_377, arg232_1);  mul_377 = arg232_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_301: "f32[2592, 512]" = torch.ops.aten.view.default(add_257, [2592, 512]);  add_257 = None
        permute_258: "f32[512, 2048]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_121: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg234_1, view_301, permute_258);  arg234_1 = view_301 = permute_258 = None
        view_302: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_121, [8, 18, 18, 2048]);  addmm_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_378: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_302, 0.5)
        mul_379: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_302, 0.7071067811865476);  view_302 = None
        erf_60: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_379);  mul_379 = None
        add_258: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_380: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_378, add_258);  mul_378 = add_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_303: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_380, [2592, 2048]);  mul_380 = None
        permute_259: "f32[2048, 512]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_122: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg236_1, view_303, permute_259);  arg236_1 = view_303 = permute_259 = None
        view_304: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_122, [8, 18, 18, 512]);  addmm_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_260: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_304, [0, 3, 1, 2]);  view_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_305: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg237_1, [1, -1, 1, 1]);  arg237_1 = None
        mul_381: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_260, view_305);  permute_260 = view_305 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_259: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_381, add_255);  mul_381 = add_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_68: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_259, arg238_1, arg239_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg238_1 = arg239_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_261: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_68, [0, 2, 3, 1]);  convolution_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_69 = torch.ops.aten.var_mean.correction(permute_261, [3], correction = 0, keepdim = True)
        getitem_138: "f32[8, 18, 18, 1]" = var_mean_69[0]
        getitem_139: "f32[8, 18, 18, 1]" = var_mean_69[1];  var_mean_69 = None
        add_260: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
        rsqrt_69: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        sub_69: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_261, getitem_139);  permute_261 = getitem_139 = None
        mul_382: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        mul_383: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_382, arg240_1);  mul_382 = arg240_1 = None
        add_261: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_383, arg241_1);  mul_383 = arg241_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_306: "f32[2592, 512]" = torch.ops.aten.view.default(add_261, [2592, 512]);  add_261 = None
        permute_262: "f32[512, 2048]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_123: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg243_1, view_306, permute_262);  arg243_1 = view_306 = permute_262 = None
        view_307: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_123, [8, 18, 18, 2048]);  addmm_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_384: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
        mul_385: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_307, 0.7071067811865476);  view_307 = None
        erf_61: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_385);  mul_385 = None
        add_262: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_386: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_384, add_262);  mul_384 = add_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_308: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_386, [2592, 2048]);  mul_386 = None
        permute_263: "f32[2048, 512]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_124: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg245_1, view_308, permute_263);  arg245_1 = view_308 = permute_263 = None
        view_309: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_124, [8, 18, 18, 512]);  addmm_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_264: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_309, [0, 3, 1, 2]);  view_309 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_310: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg246_1, [1, -1, 1, 1]);  arg246_1 = None
        mul_387: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_264, view_310);  permute_264 = view_310 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_263: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_387, add_259);  mul_387 = add_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_69: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_263, arg247_1, arg248_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg247_1 = arg248_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_265: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_69, [0, 2, 3, 1]);  convolution_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_70 = torch.ops.aten.var_mean.correction(permute_265, [3], correction = 0, keepdim = True)
        getitem_140: "f32[8, 18, 18, 1]" = var_mean_70[0]
        getitem_141: "f32[8, 18, 18, 1]" = var_mean_70[1];  var_mean_70 = None
        add_264: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
        rsqrt_70: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
        sub_70: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_265, getitem_141);  permute_265 = getitem_141 = None
        mul_388: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        mul_389: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_388, arg249_1);  mul_388 = arg249_1 = None
        add_265: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_389, arg250_1);  mul_389 = arg250_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_311: "f32[2592, 512]" = torch.ops.aten.view.default(add_265, [2592, 512]);  add_265 = None
        permute_266: "f32[512, 2048]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_125: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg252_1, view_311, permute_266);  arg252_1 = view_311 = permute_266 = None
        view_312: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_125, [8, 18, 18, 2048]);  addmm_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_390: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_312, 0.5)
        mul_391: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_312, 0.7071067811865476);  view_312 = None
        erf_62: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_391);  mul_391 = None
        add_266: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_392: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_390, add_266);  mul_390 = add_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_313: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_392, [2592, 2048]);  mul_392 = None
        permute_267: "f32[2048, 512]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_126: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg254_1, view_313, permute_267);  arg254_1 = view_313 = permute_267 = None
        view_314: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_126, [8, 18, 18, 512]);  addmm_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_268: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_314, [0, 3, 1, 2]);  view_314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_315: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg255_1, [1, -1, 1, 1]);  arg255_1 = None
        mul_393: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_268, view_315);  permute_268 = view_315 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_267: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_393, add_263);  mul_393 = add_263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_70: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_267, arg256_1, arg257_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg256_1 = arg257_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_269: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_70, [0, 2, 3, 1]);  convolution_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_71 = torch.ops.aten.var_mean.correction(permute_269, [3], correction = 0, keepdim = True)
        getitem_142: "f32[8, 18, 18, 1]" = var_mean_71[0]
        getitem_143: "f32[8, 18, 18, 1]" = var_mean_71[1];  var_mean_71 = None
        add_268: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
        rsqrt_71: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_71: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_269, getitem_143);  permute_269 = getitem_143 = None
        mul_394: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        mul_395: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_394, arg258_1);  mul_394 = arg258_1 = None
        add_269: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_395, arg259_1);  mul_395 = arg259_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_316: "f32[2592, 512]" = torch.ops.aten.view.default(add_269, [2592, 512]);  add_269 = None
        permute_270: "f32[512, 2048]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        addmm_127: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg261_1, view_316, permute_270);  arg261_1 = view_316 = permute_270 = None
        view_317: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_127, [8, 18, 18, 2048]);  addmm_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_396: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_317, 0.5)
        mul_397: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_317, 0.7071067811865476);  view_317 = None
        erf_63: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_397);  mul_397 = None
        add_270: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_398: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_396, add_270);  mul_396 = add_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_318: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_398, [2592, 2048]);  mul_398 = None
        permute_271: "f32[2048, 512]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_128: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg263_1, view_318, permute_271);  arg263_1 = view_318 = permute_271 = None
        view_319: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_128, [8, 18, 18, 512]);  addmm_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_272: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_319, [0, 3, 1, 2]);  view_319 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_320: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg264_1, [1, -1, 1, 1]);  arg264_1 = None
        mul_399: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_272, view_320);  permute_272 = view_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_271: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_399, add_267);  mul_399 = add_267 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_71: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_271, arg265_1, arg266_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg265_1 = arg266_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_273: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_71, [0, 2, 3, 1]);  convolution_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_72 = torch.ops.aten.var_mean.correction(permute_273, [3], correction = 0, keepdim = True)
        getitem_144: "f32[8, 18, 18, 1]" = var_mean_72[0]
        getitem_145: "f32[8, 18, 18, 1]" = var_mean_72[1];  var_mean_72 = None
        add_272: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_72: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        sub_72: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_273, getitem_145);  permute_273 = getitem_145 = None
        mul_400: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        mul_401: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_400, arg267_1);  mul_400 = arg267_1 = None
        add_273: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_401, arg268_1);  mul_401 = arg268_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_321: "f32[2592, 512]" = torch.ops.aten.view.default(add_273, [2592, 512]);  add_273 = None
        permute_274: "f32[512, 2048]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_129: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg270_1, view_321, permute_274);  arg270_1 = view_321 = permute_274 = None
        view_322: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_129, [8, 18, 18, 2048]);  addmm_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_402: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_322, 0.5)
        mul_403: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_322, 0.7071067811865476);  view_322 = None
        erf_64: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_403);  mul_403 = None
        add_274: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_404: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_402, add_274);  mul_402 = add_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_323: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_404, [2592, 2048]);  mul_404 = None
        permute_275: "f32[2048, 512]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_130: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg272_1, view_323, permute_275);  arg272_1 = view_323 = permute_275 = None
        view_324: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_130, [8, 18, 18, 512]);  addmm_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_276: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_324, [0, 3, 1, 2]);  view_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_325: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg273_1, [1, -1, 1, 1]);  arg273_1 = None
        mul_405: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_276, view_325);  permute_276 = view_325 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_275: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_405, add_271);  mul_405 = add_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_72: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_275, arg274_1, arg275_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg274_1 = arg275_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_277: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_72, [0, 2, 3, 1]);  convolution_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_73 = torch.ops.aten.var_mean.correction(permute_277, [3], correction = 0, keepdim = True)
        getitem_146: "f32[8, 18, 18, 1]" = var_mean_73[0]
        getitem_147: "f32[8, 18, 18, 1]" = var_mean_73[1];  var_mean_73 = None
        add_276: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
        rsqrt_73: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_73: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_277, getitem_147);  permute_277 = getitem_147 = None
        mul_406: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        mul_407: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_406, arg276_1);  mul_406 = arg276_1 = None
        add_277: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_407, arg277_1);  mul_407 = arg277_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_326: "f32[2592, 512]" = torch.ops.aten.view.default(add_277, [2592, 512]);  add_277 = None
        permute_278: "f32[512, 2048]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_131: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg279_1, view_326, permute_278);  arg279_1 = view_326 = permute_278 = None
        view_327: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_131, [8, 18, 18, 2048]);  addmm_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_408: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_409: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
        erf_65: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_409);  mul_409 = None
        add_278: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_410: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_408, add_278);  mul_408 = add_278 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_328: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_410, [2592, 2048]);  mul_410 = None
        permute_279: "f32[2048, 512]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_132: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg281_1, view_328, permute_279);  arg281_1 = view_328 = permute_279 = None
        view_329: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_132, [8, 18, 18, 512]);  addmm_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_280: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_329, [0, 3, 1, 2]);  view_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_330: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg282_1, [1, -1, 1, 1]);  arg282_1 = None
        mul_411: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_280, view_330);  permute_280 = view_330 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_279: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_411, add_275);  mul_411 = add_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_73: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_279, arg283_1, arg284_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg283_1 = arg284_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_281: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_73, [0, 2, 3, 1]);  convolution_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_74 = torch.ops.aten.var_mean.correction(permute_281, [3], correction = 0, keepdim = True)
        getitem_148: "f32[8, 18, 18, 1]" = var_mean_74[0]
        getitem_149: "f32[8, 18, 18, 1]" = var_mean_74[1];  var_mean_74 = None
        add_280: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
        rsqrt_74: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        sub_74: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_281, getitem_149);  permute_281 = getitem_149 = None
        mul_412: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        mul_413: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_412, arg285_1);  mul_412 = arg285_1 = None
        add_281: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_413, arg286_1);  mul_413 = arg286_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_331: "f32[2592, 512]" = torch.ops.aten.view.default(add_281, [2592, 512]);  add_281 = None
        permute_282: "f32[512, 2048]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_133: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg288_1, view_331, permute_282);  arg288_1 = view_331 = permute_282 = None
        view_332: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_133, [8, 18, 18, 2048]);  addmm_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_414: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_332, 0.5)
        mul_415: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476);  view_332 = None
        erf_66: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_415);  mul_415 = None
        add_282: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_416: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_414, add_282);  mul_414 = add_282 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_333: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_416, [2592, 2048]);  mul_416 = None
        permute_283: "f32[2048, 512]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_134: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg290_1, view_333, permute_283);  arg290_1 = view_333 = permute_283 = None
        view_334: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_134, [8, 18, 18, 512]);  addmm_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_284: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_334, [0, 3, 1, 2]);  view_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_335: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg291_1, [1, -1, 1, 1]);  arg291_1 = None
        mul_417: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_284, view_335);  permute_284 = view_335 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_283: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_417, add_279);  mul_417 = add_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_74: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_283, arg292_1, arg293_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg292_1 = arg293_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_285: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_74, [0, 2, 3, 1]);  convolution_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_75 = torch.ops.aten.var_mean.correction(permute_285, [3], correction = 0, keepdim = True)
        getitem_150: "f32[8, 18, 18, 1]" = var_mean_75[0]
        getitem_151: "f32[8, 18, 18, 1]" = var_mean_75[1];  var_mean_75 = None
        add_284: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
        rsqrt_75: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
        sub_75: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_285, getitem_151);  permute_285 = getitem_151 = None
        mul_418: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        mul_419: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_418, arg294_1);  mul_418 = arg294_1 = None
        add_285: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_419, arg295_1);  mul_419 = arg295_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_336: "f32[2592, 512]" = torch.ops.aten.view.default(add_285, [2592, 512]);  add_285 = None
        permute_286: "f32[512, 2048]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_135: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg297_1, view_336, permute_286);  arg297_1 = view_336 = permute_286 = None
        view_337: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_135, [8, 18, 18, 2048]);  addmm_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_420: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_337, 0.5)
        mul_421: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_337, 0.7071067811865476);  view_337 = None
        erf_67: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_421);  mul_421 = None
        add_286: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_422: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_420, add_286);  mul_420 = add_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_338: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_422, [2592, 2048]);  mul_422 = None
        permute_287: "f32[2048, 512]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_136: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg299_1, view_338, permute_287);  arg299_1 = view_338 = permute_287 = None
        view_339: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_136, [8, 18, 18, 512]);  addmm_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_288: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_339, [0, 3, 1, 2]);  view_339 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_340: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg300_1, [1, -1, 1, 1]);  arg300_1 = None
        mul_423: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_288, view_340);  permute_288 = view_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_287: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_423, add_283);  mul_423 = add_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_75: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_287, arg301_1, arg302_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg301_1 = arg302_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_289: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_75, [0, 2, 3, 1]);  convolution_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_76 = torch.ops.aten.var_mean.correction(permute_289, [3], correction = 0, keepdim = True)
        getitem_152: "f32[8, 18, 18, 1]" = var_mean_76[0]
        getitem_153: "f32[8, 18, 18, 1]" = var_mean_76[1];  var_mean_76 = None
        add_288: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-06);  getitem_152 = None
        rsqrt_76: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
        sub_76: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_289, getitem_153);  permute_289 = getitem_153 = None
        mul_424: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        mul_425: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_424, arg303_1);  mul_424 = arg303_1 = None
        add_289: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_425, arg304_1);  mul_425 = arg304_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_341: "f32[2592, 512]" = torch.ops.aten.view.default(add_289, [2592, 512]);  add_289 = None
        permute_290: "f32[512, 2048]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_137: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg306_1, view_341, permute_290);  arg306_1 = view_341 = permute_290 = None
        view_342: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_137, [8, 18, 18, 2048]);  addmm_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_426: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_342, 0.5)
        mul_427: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_342, 0.7071067811865476);  view_342 = None
        erf_68: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_427);  mul_427 = None
        add_290: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_428: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_426, add_290);  mul_426 = add_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_343: "f32[2592, 2048]" = torch.ops.aten.view.default(mul_428, [2592, 2048]);  mul_428 = None
        permute_291: "f32[2048, 512]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_138: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg308_1, view_343, permute_291);  arg308_1 = view_343 = permute_291 = None
        view_344: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_138, [8, 18, 18, 512]);  addmm_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_292: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(view_344, [0, 3, 1, 2]);  view_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_345: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg309_1, [1, -1, 1, 1]);  arg309_1 = None
        mul_429: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_292, view_345);  permute_292 = view_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_291: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_429, add_287);  mul_429 = add_287 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:68 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_293: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(add_291, [0, 2, 3, 1]);  add_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:72 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_77 = torch.ops.aten.var_mean.correction(permute_293, [3], correction = 0, keepdim = True)
        getitem_154: "f32[8, 18, 18, 1]" = var_mean_77[0]
        getitem_155: "f32[8, 18, 18, 1]" = var_mean_77[1];  var_mean_77 = None
        add_292: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
        rsqrt_77: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_77: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_293, getitem_155);  permute_293 = getitem_155 = None
        mul_430: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        mul_431: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_430, arg310_1);  mul_430 = arg310_1 = None
        add_293: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_431, arg311_1);  mul_431 = arg311_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:73 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_294: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(add_293, [0, 3, 1, 2]);  add_293 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:229 in forward, code: x = self.downsample(x)
        convolution_76: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(permute_294, arg312_1, arg313_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_294 = arg312_1 = arg313_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_77: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(convolution_76, arg314_1, arg315_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg314_1 = arg315_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_295: "f32[8, 9, 9, 1024]" = torch.ops.aten.permute.default(convolution_77, [0, 2, 3, 1]);  convolution_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_78 = torch.ops.aten.var_mean.correction(permute_295, [3], correction = 0, keepdim = True)
        getitem_156: "f32[8, 9, 9, 1]" = var_mean_78[0]
        getitem_157: "f32[8, 9, 9, 1]" = var_mean_78[1];  var_mean_78 = None
        add_294: "f32[8, 9, 9, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
        rsqrt_78: "f32[8, 9, 9, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
        sub_78: "f32[8, 9, 9, 1024]" = torch.ops.aten.sub.Tensor(permute_295, getitem_157);  permute_295 = getitem_157 = None
        mul_432: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        mul_433: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(mul_432, arg316_1);  mul_432 = arg316_1 = None
        add_295: "f32[8, 9, 9, 1024]" = torch.ops.aten.add.Tensor(mul_433, arg317_1);  mul_433 = arg317_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_346: "f32[648, 1024]" = torch.ops.aten.view.default(add_295, [648, 1024]);  add_295 = None
        permute_296: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        addmm_139: "f32[648, 4096]" = torch.ops.aten.addmm.default(arg319_1, view_346, permute_296);  arg319_1 = view_346 = permute_296 = None
        view_347: "f32[8, 9, 9, 4096]" = torch.ops.aten.view.default(addmm_139, [8, 9, 9, 4096]);  addmm_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_434: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_347, 0.5)
        mul_435: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_347, 0.7071067811865476);  view_347 = None
        erf_69: "f32[8, 9, 9, 4096]" = torch.ops.aten.erf.default(mul_435);  mul_435 = None
        add_296: "f32[8, 9, 9, 4096]" = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_436: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(mul_434, add_296);  mul_434 = add_296 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_348: "f32[648, 4096]" = torch.ops.aten.view.default(mul_436, [648, 4096]);  mul_436 = None
        permute_297: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
        addmm_140: "f32[648, 1024]" = torch.ops.aten.addmm.default(arg321_1, view_348, permute_297);  arg321_1 = view_348 = permute_297 = None
        view_349: "f32[8, 9, 9, 1024]" = torch.ops.aten.view.default(addmm_140, [8, 9, 9, 1024]);  addmm_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_298: "f32[8, 1024, 9, 9]" = torch.ops.aten.permute.default(view_349, [0, 3, 1, 2]);  view_349 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_350: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(arg322_1, [1, -1, 1, 1]);  arg322_1 = None
        mul_437: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(permute_298, view_350);  permute_298 = view_350 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_297: "f32[8, 1024, 9, 9]" = torch.ops.aten.add.Tensor(mul_437, convolution_76);  mul_437 = convolution_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_78: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(add_297, arg323_1, arg324_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg323_1 = arg324_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_299: "f32[8, 9, 9, 1024]" = torch.ops.aten.permute.default(convolution_78, [0, 2, 3, 1]);  convolution_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_79 = torch.ops.aten.var_mean.correction(permute_299, [3], correction = 0, keepdim = True)
        getitem_158: "f32[8, 9, 9, 1]" = var_mean_79[0]
        getitem_159: "f32[8, 9, 9, 1]" = var_mean_79[1];  var_mean_79 = None
        add_298: "f32[8, 9, 9, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_79: "f32[8, 9, 9, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
        sub_79: "f32[8, 9, 9, 1024]" = torch.ops.aten.sub.Tensor(permute_299, getitem_159);  permute_299 = getitem_159 = None
        mul_438: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        mul_439: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(mul_438, arg325_1);  mul_438 = arg325_1 = None
        add_299: "f32[8, 9, 9, 1024]" = torch.ops.aten.add.Tensor(mul_439, arg326_1);  mul_439 = arg326_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_351: "f32[648, 1024]" = torch.ops.aten.view.default(add_299, [648, 1024]);  add_299 = None
        permute_300: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_141: "f32[648, 4096]" = torch.ops.aten.addmm.default(arg328_1, view_351, permute_300);  arg328_1 = view_351 = permute_300 = None
        view_352: "f32[8, 9, 9, 4096]" = torch.ops.aten.view.default(addmm_141, [8, 9, 9, 4096]);  addmm_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_440: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_352, 0.5)
        mul_441: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_352, 0.7071067811865476);  view_352 = None
        erf_70: "f32[8, 9, 9, 4096]" = torch.ops.aten.erf.default(mul_441);  mul_441 = None
        add_300: "f32[8, 9, 9, 4096]" = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_442: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(mul_440, add_300);  mul_440 = add_300 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_353: "f32[648, 4096]" = torch.ops.aten.view.default(mul_442, [648, 4096]);  mul_442 = None
        permute_301: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_142: "f32[648, 1024]" = torch.ops.aten.addmm.default(arg330_1, view_353, permute_301);  arg330_1 = view_353 = permute_301 = None
        view_354: "f32[8, 9, 9, 1024]" = torch.ops.aten.view.default(addmm_142, [8, 9, 9, 1024]);  addmm_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_302: "f32[8, 1024, 9, 9]" = torch.ops.aten.permute.default(view_354, [0, 3, 1, 2]);  view_354 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_355: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(arg331_1, [1, -1, 1, 1]);  arg331_1 = None
        mul_443: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(permute_302, view_355);  permute_302 = view_355 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_301: "f32[8, 1024, 9, 9]" = torch.ops.aten.add.Tensor(mul_443, add_297);  mul_443 = add_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:152 in forward, code: x = self.conv_dw(x)
        convolution_79: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(add_301, arg332_1, arg333_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg332_1 = arg333_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:157 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_303: "f32[8, 9, 9, 1024]" = torch.ops.aten.permute.default(convolution_79, [0, 2, 3, 1]);  convolution_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_80 = torch.ops.aten.var_mean.correction(permute_303, [3], correction = 0, keepdim = True)
        getitem_160: "f32[8, 9, 9, 1]" = var_mean_80[0]
        getitem_161: "f32[8, 9, 9, 1]" = var_mean_80[1];  var_mean_80 = None
        add_302: "f32[8, 9, 9, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
        rsqrt_80: "f32[8, 9, 9, 1]" = torch.ops.aten.rsqrt.default(add_302);  add_302 = None
        sub_80: "f32[8, 9, 9, 1024]" = torch.ops.aten.sub.Tensor(permute_303, getitem_161);  permute_303 = getitem_161 = None
        mul_444: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        mul_445: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(mul_444, arg334_1);  mul_444 = arg334_1 = None
        add_303: "f32[8, 9, 9, 1024]" = torch.ops.aten.add.Tensor(mul_445, arg335_1);  mul_445 = arg335_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:42 in forward, code: x = self.fc1(x)
        view_356: "f32[648, 1024]" = torch.ops.aten.view.default(add_303, [648, 1024]);  add_303 = None
        permute_304: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
        addmm_143: "f32[648, 4096]" = torch.ops.aten.addmm.default(arg337_1, view_356, permute_304);  arg337_1 = view_356 = permute_304 = None
        view_357: "f32[8, 9, 9, 4096]" = torch.ops.aten.view.default(addmm_143, [8, 9, 9, 4096]);  addmm_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_446: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_357, 0.5)
        mul_447: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
        erf_71: "f32[8, 9, 9, 4096]" = torch.ops.aten.erf.default(mul_447);  mul_447 = None
        add_304: "f32[8, 9, 9, 4096]" = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_448: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(mul_446, add_304);  mul_446 = add_304 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/mlp.py:46 in forward, code: x = self.fc2(x)
        view_358: "f32[648, 4096]" = torch.ops.aten.view.default(mul_448, [648, 4096]);  mul_448 = None
        permute_305: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
        addmm_144: "f32[648, 1024]" = torch.ops.aten.addmm.default(arg339_1, view_358, permute_305);  arg339_1 = view_358 = permute_305 = None
        view_359: "f32[8, 9, 9, 1024]" = torch.ops.aten.view.default(addmm_144, [8, 9, 9, 1024]);  addmm_144 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:160 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_306: "f32[8, 1024, 9, 9]" = torch.ops.aten.permute.default(view_359, [0, 3, 1, 2]);  view_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:162 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        view_360: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(arg340_1, [1, -1, 1, 1]);  arg340_1 = None
        mul_449: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(permute_306, view_360);  permute_306 = view_360 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/convnext.py:164 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_305: "f32[8, 1024, 9, 9]" = torch.ops.aten.add.Tensor(mul_449, add_301);  mul_449 = add_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_1: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(add_305, [-1, -2], True);  add_305 = None
        as_strided_1: "f32[8, 1024, 1, 1]" = torch.ops.aten.as_strided.default(mean_1, [8, 1024, 1, 1], [1024, 1, 1024, 1024]);  mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:68 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_307: "f32[8, 1, 1, 1024]" = torch.ops.aten.permute.default(as_strided_1, [0, 2, 3, 1]);  as_strided_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:72 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_81 = torch.ops.aten.var_mean.correction(permute_307, [3], correction = 0, keepdim = True)
        getitem_162: "f32[8, 1, 1, 1]" = var_mean_81[0]
        getitem_163: "f32[8, 1, 1, 1]" = var_mean_81[1];  var_mean_81 = None
        add_306: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
        rsqrt_81: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_81: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(permute_307, getitem_163);  permute_307 = getitem_163 = None
        mul_450: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        mul_451: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_450, arg341_1);  mul_450 = arg341_1 = None
        add_307: "f32[8, 1, 1, 1024]" = torch.ops.aten.add.Tensor(mul_451, arg342_1);  mul_451 = arg342_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/norm.py:73 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_308: "f32[8, 1024, 1, 1]" = torch.ops.aten.permute.default(add_307, [0, 3, 1, 2]);  add_307 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:201 in forward, code: x = self.flatten(x)
        view_361: "f32[8, 1024]" = torch.ops.aten.view.default(permute_308, [8, 1024]);  permute_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:206 in forward, code: x = self.fc(x)
        permute_309: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_145: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg344_1, view_361, permute_309);  arg344_1 = view_361 = permute_309 = None
        return (addmm_145,)
        