class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 3, 3, 3]", arg1_1: "f32[16, 1, 1, 1]", arg2_1: "f32[16]", arg3_1: "f32[8, 3, 288, 288]", arg4_1: "f32[32, 16, 3, 3]", arg5_1: "f32[32, 1, 1, 1]", arg6_1: "f32[32]", arg7_1: "f32[64, 32, 3, 3]", arg8_1: "f32[64, 1, 1, 1]", arg9_1: "f32[64]", arg10_1: "f32[128, 64, 3, 3]", arg11_1: "f32[128, 1, 1, 1]", arg12_1: "f32[128]", arg13_1: "f32[256, 128, 1, 1]", arg14_1: "f32[256, 1, 1, 1]", arg15_1: "f32[256]", arg16_1: "f32[64, 128, 1, 1]", arg17_1: "f32[64, 1, 1, 1]", arg18_1: "f32[64]", arg19_1: "f32[64, 64, 3, 3]", arg20_1: "f32[64, 1, 1, 1]", arg21_1: "f32[64]", arg22_1: "f32[64, 64, 3, 3]", arg23_1: "f32[64, 1, 1, 1]", arg24_1: "f32[64]", arg25_1: "f32[256, 64, 1, 1]", arg26_1: "f32[256, 1, 1, 1]", arg27_1: "f32[256]", arg28_1: "f32[64, 256, 1, 1]", arg29_1: "f32[64]", arg30_1: "f32[256, 64, 1, 1]", arg31_1: "f32[256]", arg32_1: "f32[512, 256, 1, 1]", arg33_1: "f32[512, 1, 1, 1]", arg34_1: "f32[512]", arg35_1: "f32[128, 256, 1, 1]", arg36_1: "f32[128, 1, 1, 1]", arg37_1: "f32[128]", arg38_1: "f32[128, 64, 3, 3]", arg39_1: "f32[128, 1, 1, 1]", arg40_1: "f32[128]", arg41_1: "f32[128, 64, 3, 3]", arg42_1: "f32[128, 1, 1, 1]", arg43_1: "f32[128]", arg44_1: "f32[512, 128, 1, 1]", arg45_1: "f32[512, 1, 1, 1]", arg46_1: "f32[512]", arg47_1: "f32[128, 512, 1, 1]", arg48_1: "f32[128]", arg49_1: "f32[512, 128, 1, 1]", arg50_1: "f32[512]", arg51_1: "f32[128, 512, 1, 1]", arg52_1: "f32[128, 1, 1, 1]", arg53_1: "f32[128]", arg54_1: "f32[128, 64, 3, 3]", arg55_1: "f32[128, 1, 1, 1]", arg56_1: "f32[128]", arg57_1: "f32[128, 64, 3, 3]", arg58_1: "f32[128, 1, 1, 1]", arg59_1: "f32[128]", arg60_1: "f32[512, 128, 1, 1]", arg61_1: "f32[512, 1, 1, 1]", arg62_1: "f32[512]", arg63_1: "f32[128, 512, 1, 1]", arg64_1: "f32[128]", arg65_1: "f32[512, 128, 1, 1]", arg66_1: "f32[512]", arg67_1: "f32[1536, 512, 1, 1]", arg68_1: "f32[1536, 1, 1, 1]", arg69_1: "f32[1536]", arg70_1: "f32[384, 512, 1, 1]", arg71_1: "f32[384, 1, 1, 1]", arg72_1: "f32[384]", arg73_1: "f32[384, 64, 3, 3]", arg74_1: "f32[384, 1, 1, 1]", arg75_1: "f32[384]", arg76_1: "f32[384, 64, 3, 3]", arg77_1: "f32[384, 1, 1, 1]", arg78_1: "f32[384]", arg79_1: "f32[1536, 384, 1, 1]", arg80_1: "f32[1536, 1, 1, 1]", arg81_1: "f32[1536]", arg82_1: "f32[384, 1536, 1, 1]", arg83_1: "f32[384]", arg84_1: "f32[1536, 384, 1, 1]", arg85_1: "f32[1536]", arg86_1: "f32[384, 1536, 1, 1]", arg87_1: "f32[384, 1, 1, 1]", arg88_1: "f32[384]", arg89_1: "f32[384, 64, 3, 3]", arg90_1: "f32[384, 1, 1, 1]", arg91_1: "f32[384]", arg92_1: "f32[384, 64, 3, 3]", arg93_1: "f32[384, 1, 1, 1]", arg94_1: "f32[384]", arg95_1: "f32[1536, 384, 1, 1]", arg96_1: "f32[1536, 1, 1, 1]", arg97_1: "f32[1536]", arg98_1: "f32[384, 1536, 1, 1]", arg99_1: "f32[384]", arg100_1: "f32[1536, 384, 1, 1]", arg101_1: "f32[1536]", arg102_1: "f32[384, 1536, 1, 1]", arg103_1: "f32[384, 1, 1, 1]", arg104_1: "f32[384]", arg105_1: "f32[384, 64, 3, 3]", arg106_1: "f32[384, 1, 1, 1]", arg107_1: "f32[384]", arg108_1: "f32[384, 64, 3, 3]", arg109_1: "f32[384, 1, 1, 1]", arg110_1: "f32[384]", arg111_1: "f32[1536, 384, 1, 1]", arg112_1: "f32[1536, 1, 1, 1]", arg113_1: "f32[1536]", arg114_1: "f32[384, 1536, 1, 1]", arg115_1: "f32[384]", arg116_1: "f32[1536, 384, 1, 1]", arg117_1: "f32[1536]", arg118_1: "f32[384, 1536, 1, 1]", arg119_1: "f32[384, 1, 1, 1]", arg120_1: "f32[384]", arg121_1: "f32[384, 64, 3, 3]", arg122_1: "f32[384, 1, 1, 1]", arg123_1: "f32[384]", arg124_1: "f32[384, 64, 3, 3]", arg125_1: "f32[384, 1, 1, 1]", arg126_1: "f32[384]", arg127_1: "f32[1536, 384, 1, 1]", arg128_1: "f32[1536, 1, 1, 1]", arg129_1: "f32[1536]", arg130_1: "f32[384, 1536, 1, 1]", arg131_1: "f32[384]", arg132_1: "f32[1536, 384, 1, 1]", arg133_1: "f32[1536]", arg134_1: "f32[384, 1536, 1, 1]", arg135_1: "f32[384, 1, 1, 1]", arg136_1: "f32[384]", arg137_1: "f32[384, 64, 3, 3]", arg138_1: "f32[384, 1, 1, 1]", arg139_1: "f32[384]", arg140_1: "f32[384, 64, 3, 3]", arg141_1: "f32[384, 1, 1, 1]", arg142_1: "f32[384]", arg143_1: "f32[1536, 384, 1, 1]", arg144_1: "f32[1536, 1, 1, 1]", arg145_1: "f32[1536]", arg146_1: "f32[384, 1536, 1, 1]", arg147_1: "f32[384]", arg148_1: "f32[1536, 384, 1, 1]", arg149_1: "f32[1536]", arg150_1: "f32[384, 1536, 1, 1]", arg151_1: "f32[384, 1, 1, 1]", arg152_1: "f32[384]", arg153_1: "f32[384, 64, 3, 3]", arg154_1: "f32[384, 1, 1, 1]", arg155_1: "f32[384]", arg156_1: "f32[384, 64, 3, 3]", arg157_1: "f32[384, 1, 1, 1]", arg158_1: "f32[384]", arg159_1: "f32[1536, 384, 1, 1]", arg160_1: "f32[1536, 1, 1, 1]", arg161_1: "f32[1536]", arg162_1: "f32[384, 1536, 1, 1]", arg163_1: "f32[384]", arg164_1: "f32[1536, 384, 1, 1]", arg165_1: "f32[1536]", arg166_1: "f32[1536, 1536, 1, 1]", arg167_1: "f32[1536, 1, 1, 1]", arg168_1: "f32[1536]", arg169_1: "f32[384, 1536, 1, 1]", arg170_1: "f32[384, 1, 1, 1]", arg171_1: "f32[384]", arg172_1: "f32[384, 64, 3, 3]", arg173_1: "f32[384, 1, 1, 1]", arg174_1: "f32[384]", arg175_1: "f32[384, 64, 3, 3]", arg176_1: "f32[384, 1, 1, 1]", arg177_1: "f32[384]", arg178_1: "f32[1536, 384, 1, 1]", arg179_1: "f32[1536, 1, 1, 1]", arg180_1: "f32[1536]", arg181_1: "f32[384, 1536, 1, 1]", arg182_1: "f32[384]", arg183_1: "f32[1536, 384, 1, 1]", arg184_1: "f32[1536]", arg185_1: "f32[384, 1536, 1, 1]", arg186_1: "f32[384, 1, 1, 1]", arg187_1: "f32[384]", arg188_1: "f32[384, 64, 3, 3]", arg189_1: "f32[384, 1, 1, 1]", arg190_1: "f32[384]", arg191_1: "f32[384, 64, 3, 3]", arg192_1: "f32[384, 1, 1, 1]", arg193_1: "f32[384]", arg194_1: "f32[1536, 384, 1, 1]", arg195_1: "f32[1536, 1, 1, 1]", arg196_1: "f32[1536]", arg197_1: "f32[384, 1536, 1, 1]", arg198_1: "f32[384]", arg199_1: "f32[1536, 384, 1, 1]", arg200_1: "f32[1536]", arg201_1: "f32[384, 1536, 1, 1]", arg202_1: "f32[384, 1, 1, 1]", arg203_1: "f32[384]", arg204_1: "f32[384, 64, 3, 3]", arg205_1: "f32[384, 1, 1, 1]", arg206_1: "f32[384]", arg207_1: "f32[384, 64, 3, 3]", arg208_1: "f32[384, 1, 1, 1]", arg209_1: "f32[384]", arg210_1: "f32[1536, 384, 1, 1]", arg211_1: "f32[1536, 1, 1, 1]", arg212_1: "f32[1536]", arg213_1: "f32[384, 1536, 1, 1]", arg214_1: "f32[384]", arg215_1: "f32[1536, 384, 1, 1]", arg216_1: "f32[1536]", arg217_1: "f32[2304, 1536, 1, 1]", arg218_1: "f32[2304, 1, 1, 1]", arg219_1: "f32[2304]", arg220_1: "f32[1000, 2304]", arg221_1: "f32[1000]"):
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_172: "f32[1, 16, 27]" = torch.ops.aten.view.default(arg0_1, [1, 16, -1]);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_271: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg1_1, 0.34412564994580647);  arg1_1 = None
        view_173: "f32[16]" = torch.ops.aten.view.default(mul_271, [-1]);  mul_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_57 = torch.ops.aten.var_mean.correction(view_172, [0, 2], correction = 0, keepdim = True)
        getitem_114: "f32[1, 16, 1]" = var_mean_57[0]
        getitem_115: "f32[1, 16, 1]" = var_mean_57[1];  var_mean_57 = None
        add_69: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_57: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_57: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_172, getitem_115);  view_172 = getitem_115 = None
        mul_272: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        unsqueeze_57: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(view_173, -1);  view_173 = None
        mul_273: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_57);  mul_272 = unsqueeze_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_174: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_273, [16, 3, 3, 3]);  mul_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_81: "f32[8, 16, 144, 144]" = torch.ops.aten.convolution.default(arg3_1, view_174, arg2_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg3_1 = view_174 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:440 in forward_features, code: x = self.stem(x)
        sigmoid_64: "f32[8, 16, 144, 144]" = torch.ops.aten.sigmoid.default(convolution_81)
        mul_274: "f32[8, 16, 144, 144]" = torch.ops.aten.mul.Tensor(convolution_81, sigmoid_64);  convolution_81 = sigmoid_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_175: "f32[1, 32, 144]" = torch.ops.aten.view.default(arg4_1, [1, 32, -1]);  arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_275: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg5_1, 0.1490107774734497);  arg5_1 = None
        view_176: "f32[32]" = torch.ops.aten.view.default(mul_275, [-1]);  mul_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_58 = torch.ops.aten.var_mean.correction(view_175, [0, 2], correction = 0, keepdim = True)
        getitem_116: "f32[1, 32, 1]" = var_mean_58[0]
        getitem_117: "f32[1, 32, 1]" = var_mean_58[1];  var_mean_58 = None
        add_70: "f32[1, 32, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_58: "f32[1, 32, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_58: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(view_175, getitem_117);  view_175 = getitem_117 = None
        mul_276: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        unsqueeze_58: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(view_176, -1);  view_176 = None
        mul_277: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_58);  mul_276 = unsqueeze_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_177: "f32[32, 16, 3, 3]" = torch.ops.aten.view.default(mul_277, [32, 16, 3, 3]);  mul_277 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_82: "f32[8, 32, 144, 144]" = torch.ops.aten.convolution.default(mul_274, view_177, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_274 = view_177 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:440 in forward_features, code: x = self.stem(x)
        sigmoid_65: "f32[8, 32, 144, 144]" = torch.ops.aten.sigmoid.default(convolution_82)
        mul_278: "f32[8, 32, 144, 144]" = torch.ops.aten.mul.Tensor(convolution_82, sigmoid_65);  convolution_82 = sigmoid_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_178: "f32[1, 64, 288]" = torch.ops.aten.view.default(arg7_1, [1, 64, -1]);  arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_279: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg8_1, 0.10536653122135592);  arg8_1 = None
        view_179: "f32[64]" = torch.ops.aten.view.default(mul_279, [-1]);  mul_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_59 = torch.ops.aten.var_mean.correction(view_178, [0, 2], correction = 0, keepdim = True)
        getitem_118: "f32[1, 64, 1]" = var_mean_59[0]
        getitem_119: "f32[1, 64, 1]" = var_mean_59[1];  var_mean_59 = None
        add_71: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_59: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_59: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(view_178, getitem_119);  view_178 = getitem_119 = None
        mul_280: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        unsqueeze_59: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_179, -1);  view_179 = None
        mul_281: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_59);  mul_280 = unsqueeze_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_180: "f32[64, 32, 3, 3]" = torch.ops.aten.view.default(mul_281, [64, 32, 3, 3]);  mul_281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_83: "f32[8, 64, 144, 144]" = torch.ops.aten.convolution.default(mul_278, view_180, arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_278 = view_180 = arg9_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:440 in forward_features, code: x = self.stem(x)
        sigmoid_66: "f32[8, 64, 144, 144]" = torch.ops.aten.sigmoid.default(convolution_83)
        mul_282: "f32[8, 64, 144, 144]" = torch.ops.aten.mul.Tensor(convolution_83, sigmoid_66);  convolution_83 = sigmoid_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_181: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg10_1, [1, 128, -1]);  arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_283: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg11_1, 0.07450538873672485);  arg11_1 = None
        view_182: "f32[128]" = torch.ops.aten.view.default(mul_283, [-1]);  mul_283 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_60 = torch.ops.aten.var_mean.correction(view_181, [0, 2], correction = 0, keepdim = True)
        getitem_120: "f32[1, 128, 1]" = var_mean_60[0]
        getitem_121: "f32[1, 128, 1]" = var_mean_60[1];  var_mean_60 = None
        add_72: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
        rsqrt_60: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_60: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_181, getitem_121);  view_181 = getitem_121 = None
        mul_284: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_182, -1);  view_182 = None
        mul_285: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_60);  mul_284 = unsqueeze_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_183: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_285, [128, 64, 3, 3]);  mul_285 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_84: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(mul_282, view_183, arg12_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_282 = view_183 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_67: "f32[8, 128, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_84)
        mul_286: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_84, sigmoid_67);  convolution_84 = sigmoid_67 = None
        mul_287: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(mul_286, 1.0);  mul_286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_184: "f32[1, 256, 128]" = torch.ops.aten.view.default(arg13_1, [1, 256, -1]);  arg13_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_288: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg14_1, 0.1580497968320339);  arg14_1 = None
        view_185: "f32[256]" = torch.ops.aten.view.default(mul_288, [-1]);  mul_288 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_61 = torch.ops.aten.var_mean.correction(view_184, [0, 2], correction = 0, keepdim = True)
        getitem_122: "f32[1, 256, 1]" = var_mean_61[0]
        getitem_123: "f32[1, 256, 1]" = var_mean_61[1];  var_mean_61 = None
        add_73: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
        rsqrt_61: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_61: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(view_184, getitem_123);  view_184 = getitem_123 = None
        mul_289: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        unsqueeze_61: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_185, -1);  view_185 = None
        mul_290: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_61);  mul_289 = unsqueeze_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_186: "f32[256, 128, 1, 1]" = torch.ops.aten.view.default(mul_290, [256, 128, 1, 1]);  mul_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_85: "f32[8, 256, 72, 72]" = torch.ops.aten.convolution.default(mul_287, view_186, arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_186 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_187: "f32[1, 64, 128]" = torch.ops.aten.view.default(arg16_1, [1, 64, -1]);  arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_291: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg17_1, 0.1580497968320339);  arg17_1 = None
        view_188: "f32[64]" = torch.ops.aten.view.default(mul_291, [-1]);  mul_291 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_62 = torch.ops.aten.var_mean.correction(view_187, [0, 2], correction = 0, keepdim = True)
        getitem_124: "f32[1, 64, 1]" = var_mean_62[0]
        getitem_125: "f32[1, 64, 1]" = var_mean_62[1];  var_mean_62 = None
        add_74: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
        rsqrt_62: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_62: "f32[1, 64, 128]" = torch.ops.aten.sub.Tensor(view_187, getitem_125);  view_187 = getitem_125 = None
        mul_292: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        unsqueeze_62: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_188, -1);  view_188 = None
        mul_293: "f32[1, 64, 128]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_62);  mul_292 = unsqueeze_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_189: "f32[64, 128, 1, 1]" = torch.ops.aten.view.default(mul_293, [64, 128, 1, 1]);  mul_293 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_86: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(mul_287, view_189, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_287 = view_189 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_68: "f32[8, 64, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_86)
        mul_294: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_86, sigmoid_68);  convolution_86 = sigmoid_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_190: "f32[1, 64, 576]" = torch.ops.aten.view.default(arg19_1, [1, 64, -1]);  arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_295: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg20_1, 0.07450538873672485);  arg20_1 = None
        view_191: "f32[64]" = torch.ops.aten.view.default(mul_295, [-1]);  mul_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_63 = torch.ops.aten.var_mean.correction(view_190, [0, 2], correction = 0, keepdim = True)
        getitem_126: "f32[1, 64, 1]" = var_mean_63[0]
        getitem_127: "f32[1, 64, 1]" = var_mean_63[1];  var_mean_63 = None
        add_75: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_63: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_63: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_190, getitem_127);  view_190 = getitem_127 = None
        mul_296: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        unsqueeze_63: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_191, -1);  view_191 = None
        mul_297: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_63);  mul_296 = unsqueeze_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_192: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_297, [64, 64, 3, 3]);  mul_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_87: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(mul_294, view_192, arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_294 = view_192 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_69: "f32[8, 64, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_87)
        mul_298: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_87, sigmoid_69);  convolution_87 = sigmoid_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_193: "f32[1, 64, 576]" = torch.ops.aten.view.default(arg22_1, [1, 64, -1]);  arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_299: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg23_1, 0.07450538873672485);  arg23_1 = None
        view_194: "f32[64]" = torch.ops.aten.view.default(mul_299, [-1]);  mul_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_64 = torch.ops.aten.var_mean.correction(view_193, [0, 2], correction = 0, keepdim = True)
        getitem_128: "f32[1, 64, 1]" = var_mean_64[0]
        getitem_129: "f32[1, 64, 1]" = var_mean_64[1];  var_mean_64 = None
        add_76: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_64: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_64: "f32[1, 64, 576]" = torch.ops.aten.sub.Tensor(view_193, getitem_129);  view_193 = getitem_129 = None
        mul_300: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(view_194, -1);  view_194 = None
        mul_301: "f32[1, 64, 576]" = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_64);  mul_300 = unsqueeze_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_195: "f32[64, 64, 3, 3]" = torch.ops.aten.view.default(mul_301, [64, 64, 3, 3]);  mul_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_88: "f32[8, 64, 72, 72]" = torch.ops.aten.convolution.default(mul_298, view_195, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_298 = view_195 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_70: "f32[8, 64, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_88)
        mul_302: "f32[8, 64, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_88, sigmoid_70);  convolution_88 = sigmoid_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_196: "f32[1, 256, 64]" = torch.ops.aten.view.default(arg25_1, [1, 256, -1]);  arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_303: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg26_1, 0.22351616621017456);  arg26_1 = None
        view_197: "f32[256]" = torch.ops.aten.view.default(mul_303, [-1]);  mul_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_65 = torch.ops.aten.var_mean.correction(view_196, [0, 2], correction = 0, keepdim = True)
        getitem_130: "f32[1, 256, 1]" = var_mean_65[0]
        getitem_131: "f32[1, 256, 1]" = var_mean_65[1];  var_mean_65 = None
        add_77: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_65: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_65: "f32[1, 256, 64]" = torch.ops.aten.sub.Tensor(view_196, getitem_131);  view_196 = getitem_131 = None
        mul_304: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        unsqueeze_65: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(view_197, -1);  view_197 = None
        mul_305: "f32[1, 256, 64]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_65);  mul_304 = unsqueeze_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_198: "f32[256, 64, 1, 1]" = torch.ops.aten.view.default(mul_305, [256, 64, 1, 1]);  mul_305 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_89: "f32[8, 256, 72, 72]" = torch.ops.aten.convolution.default(mul_302, view_198, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_302 = view_198 = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_13: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(convolution_89, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_90: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_13, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg28_1 = arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_12: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(convolution_90);  convolution_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_91: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_12, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_12 = arg30_1 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_71: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_91);  convolution_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_306: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_89, sigmoid_71);  convolution_89 = sigmoid_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_307: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(mul_306, 2.0);  mul_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_308: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(mul_307, 0.2);  mul_307 = None
        add_78: "f32[8, 256, 72, 72]" = torch.ops.aten.add.Tensor(mul_308, convolution_85);  mul_308 = convolution_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_72: "f32[8, 256, 72, 72]" = torch.ops.aten.sigmoid.default(add_78)
        mul_309: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(add_78, sigmoid_72);  add_78 = sigmoid_72 = None
        mul_310: "f32[8, 256, 72, 72]" = torch.ops.aten.mul.Tensor(mul_309, 0.9805806756909201);  mul_309 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:101 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_3: "f32[8, 256, 36, 36]" = torch.ops.aten.avg_pool2d.default(mul_310, [2, 2], [2, 2], [0, 0], True, False)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_199: "f32[1, 512, 256]" = torch.ops.aten.view.default(arg32_1, [1, 512, -1]);  arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_311: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg33_1, 0.11175808310508728);  arg33_1 = None
        view_200: "f32[512]" = torch.ops.aten.view.default(mul_311, [-1]);  mul_311 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_66 = torch.ops.aten.var_mean.correction(view_199, [0, 2], correction = 0, keepdim = True)
        getitem_132: "f32[1, 512, 1]" = var_mean_66[0]
        getitem_133: "f32[1, 512, 1]" = var_mean_66[1];  var_mean_66 = None
        add_79: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_66: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_66: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(view_199, getitem_133);  view_199 = getitem_133 = None
        mul_312: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        unsqueeze_66: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_200, -1);  view_200 = None
        mul_313: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_312, unsqueeze_66);  mul_312 = unsqueeze_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_201: "f32[512, 256, 1, 1]" = torch.ops.aten.view.default(mul_313, [512, 256, 1, 1]);  mul_313 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_92: "f32[8, 512, 36, 36]" = torch.ops.aten.convolution.default(avg_pool2d_3, view_201, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_3 = view_201 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_202: "f32[1, 128, 256]" = torch.ops.aten.view.default(arg35_1, [1, 128, -1]);  arg35_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_314: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg36_1, 0.11175808310508728);  arg36_1 = None
        view_203: "f32[128]" = torch.ops.aten.view.default(mul_314, [-1]);  mul_314 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_67 = torch.ops.aten.var_mean.correction(view_202, [0, 2], correction = 0, keepdim = True)
        getitem_134: "f32[1, 128, 1]" = var_mean_67[0]
        getitem_135: "f32[1, 128, 1]" = var_mean_67[1];  var_mean_67 = None
        add_80: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
        rsqrt_67: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_67: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(view_202, getitem_135);  view_202 = getitem_135 = None
        mul_315: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        unsqueeze_67: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_203, -1);  view_203 = None
        mul_316: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_67);  mul_315 = unsqueeze_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_204: "f32[128, 256, 1, 1]" = torch.ops.aten.view.default(mul_316, [128, 256, 1, 1]);  mul_316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_93: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(mul_310, view_204, arg37_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_310 = view_204 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_73: "f32[8, 128, 72, 72]" = torch.ops.aten.sigmoid.default(convolution_93)
        mul_317: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(convolution_93, sigmoid_73);  convolution_93 = sigmoid_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_205: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg38_1, [1, 128, -1]);  arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_318: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg39_1, 0.07450538873672485);  arg39_1 = None
        view_206: "f32[128]" = torch.ops.aten.view.default(mul_318, [-1]);  mul_318 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_68 = torch.ops.aten.var_mean.correction(view_205, [0, 2], correction = 0, keepdim = True)
        getitem_136: "f32[1, 128, 1]" = var_mean_68[0]
        getitem_137: "f32[1, 128, 1]" = var_mean_68[1];  var_mean_68 = None
        add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
        rsqrt_68: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_68: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_205, getitem_137);  view_205 = getitem_137 = None
        mul_319: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_206, -1);  view_206 = None
        mul_320: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_68);  mul_319 = unsqueeze_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_207: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_320, [128, 64, 3, 3]);  mul_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_94: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_317, view_207, arg40_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 2);  mul_317 = view_207 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_74: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_94)
        mul_321: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_94, sigmoid_74);  convolution_94 = sigmoid_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_208: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg41_1, [1, 128, -1]);  arg41_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_322: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg42_1, 0.07450538873672485);  arg42_1 = None
        view_209: "f32[128]" = torch.ops.aten.view.default(mul_322, [-1]);  mul_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_69 = torch.ops.aten.var_mean.correction(view_208, [0, 2], correction = 0, keepdim = True)
        getitem_138: "f32[1, 128, 1]" = var_mean_69[0]
        getitem_139: "f32[1, 128, 1]" = var_mean_69[1];  var_mean_69 = None
        add_82: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
        rsqrt_69: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_69: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_208, getitem_139);  view_208 = getitem_139 = None
        mul_323: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        unsqueeze_69: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_209, -1);  view_209 = None
        mul_324: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_323, unsqueeze_69);  mul_323 = unsqueeze_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_210: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_324, [128, 64, 3, 3]);  mul_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_95: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_321, view_210, arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_321 = view_210 = arg43_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_75: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_95)
        mul_325: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_95, sigmoid_75);  convolution_95 = sigmoid_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_211: "f32[1, 512, 128]" = torch.ops.aten.view.default(arg44_1, [1, 512, -1]);  arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_326: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg45_1, 0.1580497968320339);  arg45_1 = None
        view_212: "f32[512]" = torch.ops.aten.view.default(mul_326, [-1]);  mul_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_70 = torch.ops.aten.var_mean.correction(view_211, [0, 2], correction = 0, keepdim = True)
        getitem_140: "f32[1, 512, 1]" = var_mean_70[0]
        getitem_141: "f32[1, 512, 1]" = var_mean_70[1];  var_mean_70 = None
        add_83: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_70: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_70: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_211, getitem_141);  view_211 = getitem_141 = None
        mul_327: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        unsqueeze_70: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_212, -1);  view_212 = None
        mul_328: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_327, unsqueeze_70);  mul_327 = unsqueeze_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_213: "f32[512, 128, 1, 1]" = torch.ops.aten.view.default(mul_328, [512, 128, 1, 1]);  mul_328 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_96: "f32[8, 512, 36, 36]" = torch.ops.aten.convolution.default(mul_325, view_213, arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_325 = view_213 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_96, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_97: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_14, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg47_1 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_13: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_97);  convolution_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_98: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_13, arg49_1, arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg49_1 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_76: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_98);  convolution_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_329: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_96, sigmoid_76);  convolution_96 = sigmoid_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_330: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_329, 2.0);  mul_329 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_331: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_330, 0.2);  mul_330 = None
        add_84: "f32[8, 512, 36, 36]" = torch.ops.aten.add.Tensor(mul_331, convolution_92);  mul_331 = convolution_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_77: "f32[8, 512, 36, 36]" = torch.ops.aten.sigmoid.default(add_84)
        mul_332: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(add_84, sigmoid_77);  sigmoid_77 = None
        mul_333: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_332, 0.9805806756909201);  mul_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_214: "f32[1, 128, 512]" = torch.ops.aten.view.default(arg51_1, [1, 128, -1]);  arg51_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_334: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg52_1, 0.07902489841601695);  arg52_1 = None
        view_215: "f32[128]" = torch.ops.aten.view.default(mul_334, [-1]);  mul_334 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_71 = torch.ops.aten.var_mean.correction(view_214, [0, 2], correction = 0, keepdim = True)
        getitem_142: "f32[1, 128, 1]" = var_mean_71[0]
        getitem_143: "f32[1, 128, 1]" = var_mean_71[1];  var_mean_71 = None
        add_85: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
        rsqrt_71: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_71: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(view_214, getitem_143);  view_214 = getitem_143 = None
        mul_335: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        unsqueeze_71: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_215, -1);  view_215 = None
        mul_336: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_335, unsqueeze_71);  mul_335 = unsqueeze_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_216: "f32[128, 512, 1, 1]" = torch.ops.aten.view.default(mul_336, [128, 512, 1, 1]);  mul_336 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_99: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_333, view_216, arg53_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_333 = view_216 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_78: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_99)
        mul_337: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_99, sigmoid_78);  convolution_99 = sigmoid_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_217: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg54_1, [1, 128, -1]);  arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_338: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg55_1, 0.07450538873672485);  arg55_1 = None
        view_218: "f32[128]" = torch.ops.aten.view.default(mul_338, [-1]);  mul_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_72 = torch.ops.aten.var_mean.correction(view_217, [0, 2], correction = 0, keepdim = True)
        getitem_144: "f32[1, 128, 1]" = var_mean_72[0]
        getitem_145: "f32[1, 128, 1]" = var_mean_72[1];  var_mean_72 = None
        add_86: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_72: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_72: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_217, getitem_145);  view_217 = getitem_145 = None
        mul_339: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_218, -1);  view_218 = None
        mul_340: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_339, unsqueeze_72);  mul_339 = unsqueeze_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_219: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_340, [128, 64, 3, 3]);  mul_340 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_100: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_337, view_219, arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_337 = view_219 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_79: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_100)
        mul_341: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_100, sigmoid_79);  convolution_100 = sigmoid_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_220: "f32[1, 128, 576]" = torch.ops.aten.view.default(arg57_1, [1, 128, -1]);  arg57_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_342: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg58_1, 0.07450538873672485);  arg58_1 = None
        view_221: "f32[128]" = torch.ops.aten.view.default(mul_342, [-1]);  mul_342 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_73 = torch.ops.aten.var_mean.correction(view_220, [0, 2], correction = 0, keepdim = True)
        getitem_146: "f32[1, 128, 1]" = var_mean_73[0]
        getitem_147: "f32[1, 128, 1]" = var_mean_73[1];  var_mean_73 = None
        add_87: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_73: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_73: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(view_220, getitem_147);  view_220 = getitem_147 = None
        mul_343: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        unsqueeze_73: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(view_221, -1);  view_221 = None
        mul_344: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_73);  mul_343 = unsqueeze_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_222: "f32[128, 64, 3, 3]" = torch.ops.aten.view.default(mul_344, [128, 64, 3, 3]);  mul_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_101: "f32[8, 128, 36, 36]" = torch.ops.aten.convolution.default(mul_341, view_222, arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_341 = view_222 = arg59_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_80: "f32[8, 128, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_101)
        mul_345: "f32[8, 128, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_101, sigmoid_80);  convolution_101 = sigmoid_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_223: "f32[1, 512, 128]" = torch.ops.aten.view.default(arg60_1, [1, 512, -1]);  arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_346: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg61_1, 0.1580497968320339);  arg61_1 = None
        view_224: "f32[512]" = torch.ops.aten.view.default(mul_346, [-1]);  mul_346 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_74 = torch.ops.aten.var_mean.correction(view_223, [0, 2], correction = 0, keepdim = True)
        getitem_148: "f32[1, 512, 1]" = var_mean_74[0]
        getitem_149: "f32[1, 512, 1]" = var_mean_74[1];  var_mean_74 = None
        add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
        rsqrt_74: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_74: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(view_223, getitem_149);  view_223 = getitem_149 = None
        mul_347: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        unsqueeze_74: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(view_224, -1);  view_224 = None
        mul_348: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_347, unsqueeze_74);  mul_347 = unsqueeze_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_225: "f32[512, 128, 1, 1]" = torch.ops.aten.view.default(mul_348, [512, 128, 1, 1]);  mul_348 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_102: "f32[8, 512, 36, 36]" = torch.ops.aten.convolution.default(mul_345, view_225, arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_345 = view_225 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_15: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(convolution_102, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_103: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_15, arg63_1, arg64_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg63_1 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_14: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(convolution_103);  convolution_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_104: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_14, arg65_1, arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg65_1 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_81: "f32[8, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_104);  convolution_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_349: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_102, sigmoid_81);  convolution_102 = sigmoid_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_350: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_349, 2.0);  mul_349 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_351: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_350, 0.2);  mul_350 = None
        add_89: "f32[8, 512, 36, 36]" = torch.ops.aten.add.Tensor(mul_351, add_84);  mul_351 = add_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_82: "f32[8, 512, 36, 36]" = torch.ops.aten.sigmoid.default(add_89)
        mul_352: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(add_89, sigmoid_82);  add_89 = sigmoid_82 = None
        mul_353: "f32[8, 512, 36, 36]" = torch.ops.aten.mul.Tensor(mul_352, 0.9622504486493761);  mul_352 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:101 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_4: "f32[8, 512, 18, 18]" = torch.ops.aten.avg_pool2d.default(mul_353, [2, 2], [2, 2], [0, 0], True, False)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_226: "f32[1, 1536, 512]" = torch.ops.aten.view.default(arg67_1, [1, 1536, -1]);  arg67_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_354: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg68_1, 0.07902489841601695);  arg68_1 = None
        view_227: "f32[1536]" = torch.ops.aten.view.default(mul_354, [-1]);  mul_354 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_75 = torch.ops.aten.var_mean.correction(view_226, [0, 2], correction = 0, keepdim = True)
        getitem_150: "f32[1, 1536, 1]" = var_mean_75[0]
        getitem_151: "f32[1, 1536, 1]" = var_mean_75[1];  var_mean_75 = None
        add_90: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
        rsqrt_75: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_75: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(view_226, getitem_151);  view_226 = getitem_151 = None
        mul_355: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        unsqueeze_75: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_227, -1);  view_227 = None
        mul_356: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_75);  mul_355 = unsqueeze_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_228: "f32[1536, 512, 1, 1]" = torch.ops.aten.view.default(mul_356, [1536, 512, 1, 1]);  mul_356 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_105: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(avg_pool2d_4, view_228, arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_4 = view_228 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_229: "f32[1, 384, 512]" = torch.ops.aten.view.default(arg70_1, [1, 384, -1]);  arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_357: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg71_1, 0.07902489841601695);  arg71_1 = None
        view_230: "f32[384]" = torch.ops.aten.view.default(mul_357, [-1]);  mul_357 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_76 = torch.ops.aten.var_mean.correction(view_229, [0, 2], correction = 0, keepdim = True)
        getitem_152: "f32[1, 384, 1]" = var_mean_76[0]
        getitem_153: "f32[1, 384, 1]" = var_mean_76[1];  var_mean_76 = None
        add_91: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
        rsqrt_76: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_76: "f32[1, 384, 512]" = torch.ops.aten.sub.Tensor(view_229, getitem_153);  view_229 = getitem_153 = None
        mul_358: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        unsqueeze_76: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_230, -1);  view_230 = None
        mul_359: "f32[1, 384, 512]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_76);  mul_358 = unsqueeze_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_231: "f32[384, 512, 1, 1]" = torch.ops.aten.view.default(mul_359, [384, 512, 1, 1]);  mul_359 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_106: "f32[8, 384, 36, 36]" = torch.ops.aten.convolution.default(mul_353, view_231, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_353 = view_231 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_83: "f32[8, 384, 36, 36]" = torch.ops.aten.sigmoid.default(convolution_106)
        mul_360: "f32[8, 384, 36, 36]" = torch.ops.aten.mul.Tensor(convolution_106, sigmoid_83);  convolution_106 = sigmoid_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_232: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg73_1, [1, 384, -1]);  arg73_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_361: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg74_1, 0.07450538873672485);  arg74_1 = None
        view_233: "f32[384]" = torch.ops.aten.view.default(mul_361, [-1]);  mul_361 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_77 = torch.ops.aten.var_mean.correction(view_232, [0, 2], correction = 0, keepdim = True)
        getitem_154: "f32[1, 384, 1]" = var_mean_77[0]
        getitem_155: "f32[1, 384, 1]" = var_mean_77[1];  var_mean_77 = None
        add_92: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_77: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_77: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_232, getitem_155);  view_232 = getitem_155 = None
        mul_362: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        unsqueeze_77: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_233, -1);  view_233 = None
        mul_363: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_362, unsqueeze_77);  mul_362 = unsqueeze_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_234: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_363, [384, 64, 3, 3]);  mul_363 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_107: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_360, view_234, arg75_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  mul_360 = view_234 = arg75_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_84: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_107)
        mul_364: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_107, sigmoid_84);  convolution_107 = sigmoid_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_235: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg76_1, [1, 384, -1]);  arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_365: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg77_1, 0.07450538873672485);  arg77_1 = None
        view_236: "f32[384]" = torch.ops.aten.view.default(mul_365, [-1]);  mul_365 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_78 = torch.ops.aten.var_mean.correction(view_235, [0, 2], correction = 0, keepdim = True)
        getitem_156: "f32[1, 384, 1]" = var_mean_78[0]
        getitem_157: "f32[1, 384, 1]" = var_mean_78[1];  var_mean_78 = None
        add_93: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_78: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_78: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_235, getitem_157);  view_235 = getitem_157 = None
        mul_366: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        unsqueeze_78: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_236, -1);  view_236 = None
        mul_367: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_366, unsqueeze_78);  mul_366 = unsqueeze_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_237: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_367, [384, 64, 3, 3]);  mul_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_108: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_364, view_237, arg78_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_364 = view_237 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_85: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_108)
        mul_368: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_108, sigmoid_85);  convolution_108 = sigmoid_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_238: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg79_1, [1, 1536, -1]);  arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_369: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg80_1, 0.09125009274634042);  arg80_1 = None
        view_239: "f32[1536]" = torch.ops.aten.view.default(mul_369, [-1]);  mul_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_79 = torch.ops.aten.var_mean.correction(view_238, [0, 2], correction = 0, keepdim = True)
        getitem_158: "f32[1, 1536, 1]" = var_mean_79[0]
        getitem_159: "f32[1, 1536, 1]" = var_mean_79[1];  var_mean_79 = None
        add_94: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_79: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_79: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_238, getitem_159);  view_238 = getitem_159 = None
        mul_370: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        unsqueeze_79: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_239, -1);  view_239 = None
        mul_371: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_79);  mul_370 = unsqueeze_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_240: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_371, [1536, 384, 1, 1]);  mul_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_109: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_368, view_240, arg81_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_368 = view_240 = arg81_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_16: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_109, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_110: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_16, arg82_1, arg83_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_16 = arg82_1 = arg83_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_15: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_110);  convolution_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_111: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_15, arg84_1, arg85_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg84_1 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_86: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_111);  convolution_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_372: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_109, sigmoid_86);  convolution_109 = sigmoid_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_373: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_372, 2.0);  mul_372 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_374: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_373, 0.2);  mul_373 = None
        add_95: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_374, convolution_105);  mul_374 = convolution_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_87: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_95)
        mul_375: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_95, sigmoid_87);  sigmoid_87 = None
        mul_376: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_375, 0.9805806756909201);  mul_375 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_241: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg86_1, [1, 384, -1]);  arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_377: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg87_1, 0.04562504637317021);  arg87_1 = None
        view_242: "f32[384]" = torch.ops.aten.view.default(mul_377, [-1]);  mul_377 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_80 = torch.ops.aten.var_mean.correction(view_241, [0, 2], correction = 0, keepdim = True)
        getitem_160: "f32[1, 384, 1]" = var_mean_80[0]
        getitem_161: "f32[1, 384, 1]" = var_mean_80[1];  var_mean_80 = None
        add_96: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_80: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_80: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_241, getitem_161);  view_241 = getitem_161 = None
        mul_378: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        unsqueeze_80: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_242, -1);  view_242 = None
        mul_379: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_80);  mul_378 = unsqueeze_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_243: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_379, [384, 1536, 1, 1]);  mul_379 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_112: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_376, view_243, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_376 = view_243 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_88: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_112)
        mul_380: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_112, sigmoid_88);  convolution_112 = sigmoid_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_244: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg89_1, [1, 384, -1]);  arg89_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_381: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg90_1, 0.07450538873672485);  arg90_1 = None
        view_245: "f32[384]" = torch.ops.aten.view.default(mul_381, [-1]);  mul_381 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_81 = torch.ops.aten.var_mean.correction(view_244, [0, 2], correction = 0, keepdim = True)
        getitem_162: "f32[1, 384, 1]" = var_mean_81[0]
        getitem_163: "f32[1, 384, 1]" = var_mean_81[1];  var_mean_81 = None
        add_97: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_81: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_81: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_244, getitem_163);  view_244 = getitem_163 = None
        mul_382: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        unsqueeze_81: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_245, -1);  view_245 = None
        mul_383: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_81);  mul_382 = unsqueeze_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_246: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_383, [384, 64, 3, 3]);  mul_383 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_113: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_380, view_246, arg91_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_380 = view_246 = arg91_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_89: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_113)
        mul_384: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_113, sigmoid_89);  convolution_113 = sigmoid_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_247: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg92_1, [1, 384, -1]);  arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_385: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg93_1, 0.07450538873672485);  arg93_1 = None
        view_248: "f32[384]" = torch.ops.aten.view.default(mul_385, [-1]);  mul_385 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_82 = torch.ops.aten.var_mean.correction(view_247, [0, 2], correction = 0, keepdim = True)
        getitem_164: "f32[1, 384, 1]" = var_mean_82[0]
        getitem_165: "f32[1, 384, 1]" = var_mean_82[1];  var_mean_82 = None
        add_98: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05);  getitem_164 = None
        rsqrt_82: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_82: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_247, getitem_165);  view_247 = getitem_165 = None
        mul_386: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = rsqrt_82 = None
        unsqueeze_82: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_248, -1);  view_248 = None
        mul_387: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_386, unsqueeze_82);  mul_386 = unsqueeze_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_249: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_387, [384, 64, 3, 3]);  mul_387 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_114: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_384, view_249, arg94_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_384 = view_249 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_90: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_114)
        mul_388: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_114, sigmoid_90);  convolution_114 = sigmoid_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_250: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg95_1, [1, 1536, -1]);  arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_389: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg96_1, 0.09125009274634042);  arg96_1 = None
        view_251: "f32[1536]" = torch.ops.aten.view.default(mul_389, [-1]);  mul_389 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_83 = torch.ops.aten.var_mean.correction(view_250, [0, 2], correction = 0, keepdim = True)
        getitem_166: "f32[1, 1536, 1]" = var_mean_83[0]
        getitem_167: "f32[1, 1536, 1]" = var_mean_83[1];  var_mean_83 = None
        add_99: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
        rsqrt_83: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_83: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_250, getitem_167);  view_250 = getitem_167 = None
        mul_390: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = rsqrt_83 = None
        unsqueeze_83: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_251, -1);  view_251 = None
        mul_391: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_390, unsqueeze_83);  mul_390 = unsqueeze_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_252: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_391, [1536, 384, 1, 1]);  mul_391 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_115: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_388, view_252, arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_388 = view_252 = arg97_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_17: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_115, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_116: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_17, arg98_1, arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg98_1 = arg99_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_16: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_116);  convolution_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_117: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_16, arg100_1, arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_16 = arg100_1 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_91: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_117);  convolution_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_392: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_115, sigmoid_91);  convolution_115 = sigmoid_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_393: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_392, 2.0);  mul_392 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_394: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_393, 0.2);  mul_393 = None
        add_100: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_394, add_95);  mul_394 = add_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_92: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_100)
        mul_395: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_100, sigmoid_92);  sigmoid_92 = None
        mul_396: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_395, 0.9622504486493761);  mul_395 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_253: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg102_1, [1, 384, -1]);  arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_397: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg103_1, 0.04562504637317021);  arg103_1 = None
        view_254: "f32[384]" = torch.ops.aten.view.default(mul_397, [-1]);  mul_397 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_84 = torch.ops.aten.var_mean.correction(view_253, [0, 2], correction = 0, keepdim = True)
        getitem_168: "f32[1, 384, 1]" = var_mean_84[0]
        getitem_169: "f32[1, 384, 1]" = var_mean_84[1];  var_mean_84 = None
        add_101: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05);  getitem_168 = None
        rsqrt_84: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_84: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_253, getitem_169);  view_253 = getitem_169 = None
        mul_398: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = rsqrt_84 = None
        unsqueeze_84: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_254, -1);  view_254 = None
        mul_399: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_398, unsqueeze_84);  mul_398 = unsqueeze_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_255: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_399, [384, 1536, 1, 1]);  mul_399 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_118: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_396, view_255, arg104_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_396 = view_255 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_93: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_118)
        mul_400: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_118, sigmoid_93);  convolution_118 = sigmoid_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_256: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg105_1, [1, 384, -1]);  arg105_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_401: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg106_1, 0.07450538873672485);  arg106_1 = None
        view_257: "f32[384]" = torch.ops.aten.view.default(mul_401, [-1]);  mul_401 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_85 = torch.ops.aten.var_mean.correction(view_256, [0, 2], correction = 0, keepdim = True)
        getitem_170: "f32[1, 384, 1]" = var_mean_85[0]
        getitem_171: "f32[1, 384, 1]" = var_mean_85[1];  var_mean_85 = None
        add_102: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
        rsqrt_85: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_85: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_256, getitem_171);  view_256 = getitem_171 = None
        mul_402: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = rsqrt_85 = None
        unsqueeze_85: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_257, -1);  view_257 = None
        mul_403: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_402, unsqueeze_85);  mul_402 = unsqueeze_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_258: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_403, [384, 64, 3, 3]);  mul_403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_119: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_400, view_258, arg107_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_400 = view_258 = arg107_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_94: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_119)
        mul_404: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_119, sigmoid_94);  convolution_119 = sigmoid_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_259: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg108_1, [1, 384, -1]);  arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_405: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg109_1, 0.07450538873672485);  arg109_1 = None
        view_260: "f32[384]" = torch.ops.aten.view.default(mul_405, [-1]);  mul_405 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_86 = torch.ops.aten.var_mean.correction(view_259, [0, 2], correction = 0, keepdim = True)
        getitem_172: "f32[1, 384, 1]" = var_mean_86[0]
        getitem_173: "f32[1, 384, 1]" = var_mean_86[1];  var_mean_86 = None
        add_103: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
        rsqrt_86: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_86: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_259, getitem_173);  view_259 = getitem_173 = None
        mul_406: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        unsqueeze_86: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_260, -1);  view_260 = None
        mul_407: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_86);  mul_406 = unsqueeze_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_261: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_407, [384, 64, 3, 3]);  mul_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_120: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_404, view_261, arg110_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_404 = view_261 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_95: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_120)
        mul_408: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_120, sigmoid_95);  convolution_120 = sigmoid_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_262: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg111_1, [1, 1536, -1]);  arg111_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_409: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg112_1, 0.09125009274634042);  arg112_1 = None
        view_263: "f32[1536]" = torch.ops.aten.view.default(mul_409, [-1]);  mul_409 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_87 = torch.ops.aten.var_mean.correction(view_262, [0, 2], correction = 0, keepdim = True)
        getitem_174: "f32[1, 1536, 1]" = var_mean_87[0]
        getitem_175: "f32[1, 1536, 1]" = var_mean_87[1];  var_mean_87 = None
        add_104: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_87: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_87: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_262, getitem_175);  view_262 = getitem_175 = None
        mul_410: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        unsqueeze_87: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_263, -1);  view_263 = None
        mul_411: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_410, unsqueeze_87);  mul_410 = unsqueeze_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_264: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_411, [1536, 384, 1, 1]);  mul_411 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_121: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_408, view_264, arg113_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_408 = view_264 = arg113_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_18: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_121, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_122: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_18, arg114_1, arg115_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg114_1 = arg115_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_17: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_122);  convolution_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_123: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_17, arg116_1, arg117_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg116_1 = arg117_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_96: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_123);  convolution_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_412: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_121, sigmoid_96);  convolution_121 = sigmoid_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_413: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_412, 2.0);  mul_412 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_414: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_413, 0.2);  mul_413 = None
        add_105: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_414, add_100);  mul_414 = add_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_97: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_105)
        mul_415: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_105, sigmoid_97);  sigmoid_97 = None
        mul_416: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_415, 0.9449111825230679);  mul_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_265: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg118_1, [1, 384, -1]);  arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_417: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg119_1, 0.04562504637317021);  arg119_1 = None
        view_266: "f32[384]" = torch.ops.aten.view.default(mul_417, [-1]);  mul_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_88 = torch.ops.aten.var_mean.correction(view_265, [0, 2], correction = 0, keepdim = True)
        getitem_176: "f32[1, 384, 1]" = var_mean_88[0]
        getitem_177: "f32[1, 384, 1]" = var_mean_88[1];  var_mean_88 = None
        add_106: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_88: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_88: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_265, getitem_177);  view_265 = getitem_177 = None
        mul_418: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        unsqueeze_88: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_266, -1);  view_266 = None
        mul_419: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_88);  mul_418 = unsqueeze_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_267: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_419, [384, 1536, 1, 1]);  mul_419 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_124: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_416, view_267, arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_416 = view_267 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_98: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_124)
        mul_420: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_124, sigmoid_98);  convolution_124 = sigmoid_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_268: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg121_1, [1, 384, -1]);  arg121_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_421: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg122_1, 0.07450538873672485);  arg122_1 = None
        view_269: "f32[384]" = torch.ops.aten.view.default(mul_421, [-1]);  mul_421 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_89 = torch.ops.aten.var_mean.correction(view_268, [0, 2], correction = 0, keepdim = True)
        getitem_178: "f32[1, 384, 1]" = var_mean_89[0]
        getitem_179: "f32[1, 384, 1]" = var_mean_89[1];  var_mean_89 = None
        add_107: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_89: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_89: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_268, getitem_179);  view_268 = getitem_179 = None
        mul_422: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        unsqueeze_89: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_269, -1);  view_269 = None
        mul_423: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_422, unsqueeze_89);  mul_422 = unsqueeze_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_270: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_423, [384, 64, 3, 3]);  mul_423 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_125: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_420, view_270, arg123_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_420 = view_270 = arg123_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_99: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_125)
        mul_424: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_125, sigmoid_99);  convolution_125 = sigmoid_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_271: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg124_1, [1, 384, -1]);  arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_425: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg125_1, 0.07450538873672485);  arg125_1 = None
        view_272: "f32[384]" = torch.ops.aten.view.default(mul_425, [-1]);  mul_425 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_90 = torch.ops.aten.var_mean.correction(view_271, [0, 2], correction = 0, keepdim = True)
        getitem_180: "f32[1, 384, 1]" = var_mean_90[0]
        getitem_181: "f32[1, 384, 1]" = var_mean_90[1];  var_mean_90 = None
        add_108: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_90: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_90: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_271, getitem_181);  view_271 = getitem_181 = None
        mul_426: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        unsqueeze_90: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_272, -1);  view_272 = None
        mul_427: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_90);  mul_426 = unsqueeze_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_273: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_427, [384, 64, 3, 3]);  mul_427 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_126: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_424, view_273, arg126_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_424 = view_273 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_100: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_126)
        mul_428: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_126, sigmoid_100);  convolution_126 = sigmoid_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_274: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg127_1, [1, 1536, -1]);  arg127_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_429: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg128_1, 0.09125009274634042);  arg128_1 = None
        view_275: "f32[1536]" = torch.ops.aten.view.default(mul_429, [-1]);  mul_429 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_91 = torch.ops.aten.var_mean.correction(view_274, [0, 2], correction = 0, keepdim = True)
        getitem_182: "f32[1, 1536, 1]" = var_mean_91[0]
        getitem_183: "f32[1, 1536, 1]" = var_mean_91[1];  var_mean_91 = None
        add_109: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_91: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_91: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_274, getitem_183);  view_274 = getitem_183 = None
        mul_430: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        unsqueeze_91: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_275, -1);  view_275 = None
        mul_431: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_91);  mul_430 = unsqueeze_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_276: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_431, [1536, 384, 1, 1]);  mul_431 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_127: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_428, view_276, arg129_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_428 = view_276 = arg129_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_19: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_127, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_128: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_19, arg130_1, arg131_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg130_1 = arg131_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_18: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_128);  convolution_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_129: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_18, arg132_1, arg133_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_18 = arg132_1 = arg133_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_101: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_129);  convolution_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_432: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_127, sigmoid_101);  convolution_127 = sigmoid_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_433: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_432, 2.0);  mul_432 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_434: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_433, 0.2);  mul_433 = None
        add_110: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_434, add_105);  mul_434 = add_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_102: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_110)
        mul_435: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_110, sigmoid_102);  sigmoid_102 = None
        mul_436: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_435, 0.9284766908852592);  mul_435 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_277: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg134_1, [1, 384, -1]);  arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_437: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg135_1, 0.04562504637317021);  arg135_1 = None
        view_278: "f32[384]" = torch.ops.aten.view.default(mul_437, [-1]);  mul_437 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_92 = torch.ops.aten.var_mean.correction(view_277, [0, 2], correction = 0, keepdim = True)
        getitem_184: "f32[1, 384, 1]" = var_mean_92[0]
        getitem_185: "f32[1, 384, 1]" = var_mean_92[1];  var_mean_92 = None
        add_111: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_92: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_92: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_277, getitem_185);  view_277 = getitem_185 = None
        mul_438: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        unsqueeze_92: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_278, -1);  view_278 = None
        mul_439: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_438, unsqueeze_92);  mul_438 = unsqueeze_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_279: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_439, [384, 1536, 1, 1]);  mul_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_130: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_436, view_279, arg136_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_436 = view_279 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_103: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_130)
        mul_440: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_130, sigmoid_103);  convolution_130 = sigmoid_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_280: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg137_1, [1, 384, -1]);  arg137_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_441: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg138_1, 0.07450538873672485);  arg138_1 = None
        view_281: "f32[384]" = torch.ops.aten.view.default(mul_441, [-1]);  mul_441 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_93 = torch.ops.aten.var_mean.correction(view_280, [0, 2], correction = 0, keepdim = True)
        getitem_186: "f32[1, 384, 1]" = var_mean_93[0]
        getitem_187: "f32[1, 384, 1]" = var_mean_93[1];  var_mean_93 = None
        add_112: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_93: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_93: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_280, getitem_187);  view_280 = getitem_187 = None
        mul_442: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        unsqueeze_93: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_281, -1);  view_281 = None
        mul_443: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_93);  mul_442 = unsqueeze_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_282: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_443, [384, 64, 3, 3]);  mul_443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_131: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_440, view_282, arg139_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_440 = view_282 = arg139_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_104: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_131)
        mul_444: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_131, sigmoid_104);  convolution_131 = sigmoid_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_283: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg140_1, [1, 384, -1]);  arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_445: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg141_1, 0.07450538873672485);  arg141_1 = None
        view_284: "f32[384]" = torch.ops.aten.view.default(mul_445, [-1]);  mul_445 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_94 = torch.ops.aten.var_mean.correction(view_283, [0, 2], correction = 0, keepdim = True)
        getitem_188: "f32[1, 384, 1]" = var_mean_94[0]
        getitem_189: "f32[1, 384, 1]" = var_mean_94[1];  var_mean_94 = None
        add_113: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_94: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_94: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_283, getitem_189);  view_283 = getitem_189 = None
        mul_446: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        unsqueeze_94: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_284, -1);  view_284 = None
        mul_447: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_446, unsqueeze_94);  mul_446 = unsqueeze_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_285: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_447, [384, 64, 3, 3]);  mul_447 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_132: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_444, view_285, arg142_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_444 = view_285 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_105: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_132)
        mul_448: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_132, sigmoid_105);  convolution_132 = sigmoid_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_286: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg143_1, [1, 1536, -1]);  arg143_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_449: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg144_1, 0.09125009274634042);  arg144_1 = None
        view_287: "f32[1536]" = torch.ops.aten.view.default(mul_449, [-1]);  mul_449 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_95 = torch.ops.aten.var_mean.correction(view_286, [0, 2], correction = 0, keepdim = True)
        getitem_190: "f32[1, 1536, 1]" = var_mean_95[0]
        getitem_191: "f32[1, 1536, 1]" = var_mean_95[1];  var_mean_95 = None
        add_114: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_95: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_95: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_286, getitem_191);  view_286 = getitem_191 = None
        mul_450: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        unsqueeze_95: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_287, -1);  view_287 = None
        mul_451: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_450, unsqueeze_95);  mul_450 = unsqueeze_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_288: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_451, [1536, 384, 1, 1]);  mul_451 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_133: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_448, view_288, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_448 = view_288 = arg145_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_20: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_133, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_134: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_20, arg146_1, arg147_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg146_1 = arg147_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_19: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_134);  convolution_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_135: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_19, arg148_1, arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_19 = arg148_1 = arg149_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_106: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_135);  convolution_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_452: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_133, sigmoid_106);  convolution_133 = sigmoid_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_453: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_452, 2.0);  mul_452 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_454: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_453, 0.2);  mul_453 = None
        add_115: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_454, add_110);  mul_454 = add_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_107: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_115)
        mul_455: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_115, sigmoid_107);  sigmoid_107 = None
        mul_456: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_455, 0.9128709291752768);  mul_455 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_289: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg150_1, [1, 384, -1]);  arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_457: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg151_1, 0.04562504637317021);  arg151_1 = None
        view_290: "f32[384]" = torch.ops.aten.view.default(mul_457, [-1]);  mul_457 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_96 = torch.ops.aten.var_mean.correction(view_289, [0, 2], correction = 0, keepdim = True)
        getitem_192: "f32[1, 384, 1]" = var_mean_96[0]
        getitem_193: "f32[1, 384, 1]" = var_mean_96[1];  var_mean_96 = None
        add_116: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_96: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_96: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_289, getitem_193);  view_289 = getitem_193 = None
        mul_458: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        unsqueeze_96: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_290, -1);  view_290 = None
        mul_459: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_458, unsqueeze_96);  mul_458 = unsqueeze_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_291: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_459, [384, 1536, 1, 1]);  mul_459 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_136: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_456, view_291, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_456 = view_291 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_108: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_136)
        mul_460: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_136, sigmoid_108);  convolution_136 = sigmoid_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_292: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg153_1, [1, 384, -1]);  arg153_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_461: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg154_1, 0.07450538873672485);  arg154_1 = None
        view_293: "f32[384]" = torch.ops.aten.view.default(mul_461, [-1]);  mul_461 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_97 = torch.ops.aten.var_mean.correction(view_292, [0, 2], correction = 0, keepdim = True)
        getitem_194: "f32[1, 384, 1]" = var_mean_97[0]
        getitem_195: "f32[1, 384, 1]" = var_mean_97[1];  var_mean_97 = None
        add_117: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_97: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_97: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_292, getitem_195);  view_292 = getitem_195 = None
        mul_462: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        unsqueeze_97: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_293, -1);  view_293 = None
        mul_463: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_97);  mul_462 = unsqueeze_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_294: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_463, [384, 64, 3, 3]);  mul_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_137: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_460, view_294, arg155_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_460 = view_294 = arg155_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_109: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_137)
        mul_464: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_137, sigmoid_109);  convolution_137 = sigmoid_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_295: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg156_1, [1, 384, -1]);  arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_465: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg157_1, 0.07450538873672485);  arg157_1 = None
        view_296: "f32[384]" = torch.ops.aten.view.default(mul_465, [-1]);  mul_465 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_98 = torch.ops.aten.var_mean.correction(view_295, [0, 2], correction = 0, keepdim = True)
        getitem_196: "f32[1, 384, 1]" = var_mean_98[0]
        getitem_197: "f32[1, 384, 1]" = var_mean_98[1];  var_mean_98 = None
        add_118: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_98: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_98: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_295, getitem_197);  view_295 = getitem_197 = None
        mul_466: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = rsqrt_98 = None
        unsqueeze_98: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_296, -1);  view_296 = None
        mul_467: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_466, unsqueeze_98);  mul_466 = unsqueeze_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_297: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_467, [384, 64, 3, 3]);  mul_467 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_138: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_464, view_297, arg158_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_464 = view_297 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_110: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_138)
        mul_468: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_138, sigmoid_110);  convolution_138 = sigmoid_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_298: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg159_1, [1, 1536, -1]);  arg159_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_469: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg160_1, 0.09125009274634042);  arg160_1 = None
        view_299: "f32[1536]" = torch.ops.aten.view.default(mul_469, [-1]);  mul_469 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_99 = torch.ops.aten.var_mean.correction(view_298, [0, 2], correction = 0, keepdim = True)
        getitem_198: "f32[1, 1536, 1]" = var_mean_99[0]
        getitem_199: "f32[1, 1536, 1]" = var_mean_99[1];  var_mean_99 = None
        add_119: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_99: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_99: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_298, getitem_199);  view_298 = getitem_199 = None
        mul_470: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = rsqrt_99 = None
        unsqueeze_99: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_299, -1);  view_299 = None
        mul_471: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_99);  mul_470 = unsqueeze_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_300: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_471, [1536, 384, 1, 1]);  mul_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_139: "f32[8, 1536, 18, 18]" = torch.ops.aten.convolution.default(mul_468, view_300, arg161_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_468 = view_300 = arg161_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_21: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_139, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_140: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_21, arg162_1, arg163_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg162_1 = arg163_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_20: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_140);  convolution_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_141: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_20, arg164_1, arg165_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = arg164_1 = arg165_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_111: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_141);  convolution_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_472: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_139, sigmoid_111);  convolution_139 = sigmoid_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_473: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_472, 2.0);  mul_472 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_474: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_473, 0.2);  mul_473 = None
        add_120: "f32[8, 1536, 18, 18]" = torch.ops.aten.add.Tensor(mul_474, add_115);  mul_474 = add_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_112: "f32[8, 1536, 18, 18]" = torch.ops.aten.sigmoid.default(add_120)
        mul_475: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(add_120, sigmoid_112);  add_120 = sigmoid_112 = None
        mul_476: "f32[8, 1536, 18, 18]" = torch.ops.aten.mul.Tensor(mul_475, 0.8980265101338745);  mul_475 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:101 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_5: "f32[8, 1536, 9, 9]" = torch.ops.aten.avg_pool2d.default(mul_476, [2, 2], [2, 2], [0, 0], True, False)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_301: "f32[1, 1536, 1536]" = torch.ops.aten.view.default(arg166_1, [1, 1536, -1]);  arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_477: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg167_1, 0.04562504637317021);  arg167_1 = None
        view_302: "f32[1536]" = torch.ops.aten.view.default(mul_477, [-1]);  mul_477 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_100 = torch.ops.aten.var_mean.correction(view_301, [0, 2], correction = 0, keepdim = True)
        getitem_200: "f32[1, 1536, 1]" = var_mean_100[0]
        getitem_201: "f32[1, 1536, 1]" = var_mean_100[1];  var_mean_100 = None
        add_121: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_100: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        sub_100: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(view_301, getitem_201);  view_301 = getitem_201 = None
        mul_478: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = rsqrt_100 = None
        unsqueeze_100: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_302, -1);  view_302 = None
        mul_479: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_478, unsqueeze_100);  mul_478 = unsqueeze_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_303: "f32[1536, 1536, 1, 1]" = torch.ops.aten.view.default(mul_479, [1536, 1536, 1, 1]);  mul_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_142: "f32[8, 1536, 9, 9]" = torch.ops.aten.convolution.default(avg_pool2d_5, view_303, arg168_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_5 = view_303 = arg168_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_304: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg169_1, [1, 384, -1]);  arg169_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_480: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg170_1, 0.04562504637317021);  arg170_1 = None
        view_305: "f32[384]" = torch.ops.aten.view.default(mul_480, [-1]);  mul_480 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_101 = torch.ops.aten.var_mean.correction(view_304, [0, 2], correction = 0, keepdim = True)
        getitem_202: "f32[1, 384, 1]" = var_mean_101[0]
        getitem_203: "f32[1, 384, 1]" = var_mean_101[1];  var_mean_101 = None
        add_122: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-05);  getitem_202 = None
        rsqrt_101: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_101: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_304, getitem_203);  view_304 = getitem_203 = None
        mul_481: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = rsqrt_101 = None
        unsqueeze_101: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_305, -1);  view_305 = None
        mul_482: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_101);  mul_481 = unsqueeze_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_306: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_482, [384, 1536, 1, 1]);  mul_482 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_143: "f32[8, 384, 18, 18]" = torch.ops.aten.convolution.default(mul_476, view_306, arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_476 = view_306 = arg171_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_113: "f32[8, 384, 18, 18]" = torch.ops.aten.sigmoid.default(convolution_143)
        mul_483: "f32[8, 384, 18, 18]" = torch.ops.aten.mul.Tensor(convolution_143, sigmoid_113);  convolution_143 = sigmoid_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_307: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg172_1, [1, 384, -1]);  arg172_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_484: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg173_1, 0.07450538873672485);  arg173_1 = None
        view_308: "f32[384]" = torch.ops.aten.view.default(mul_484, [-1]);  mul_484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_102 = torch.ops.aten.var_mean.correction(view_307, [0, 2], correction = 0, keepdim = True)
        getitem_204: "f32[1, 384, 1]" = var_mean_102[0]
        getitem_205: "f32[1, 384, 1]" = var_mean_102[1];  var_mean_102 = None
        add_123: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_102: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_102: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_307, getitem_205);  view_307 = getitem_205 = None
        mul_485: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = rsqrt_102 = None
        unsqueeze_102: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_308, -1);  view_308 = None
        mul_486: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_485, unsqueeze_102);  mul_485 = unsqueeze_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_309: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_486, [384, 64, 3, 3]);  mul_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_144: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_483, view_309, arg174_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  mul_483 = view_309 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_114: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_144)
        mul_487: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_144, sigmoid_114);  convolution_144 = sigmoid_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_310: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg175_1, [1, 384, -1]);  arg175_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_488: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg176_1, 0.07450538873672485);  arg176_1 = None
        view_311: "f32[384]" = torch.ops.aten.view.default(mul_488, [-1]);  mul_488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_103 = torch.ops.aten.var_mean.correction(view_310, [0, 2], correction = 0, keepdim = True)
        getitem_206: "f32[1, 384, 1]" = var_mean_103[0]
        getitem_207: "f32[1, 384, 1]" = var_mean_103[1];  var_mean_103 = None
        add_124: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_103: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        sub_103: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_310, getitem_207);  view_310 = getitem_207 = None
        mul_489: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = rsqrt_103 = None
        unsqueeze_103: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_311, -1);  view_311 = None
        mul_490: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_489, unsqueeze_103);  mul_489 = unsqueeze_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_312: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_490, [384, 64, 3, 3]);  mul_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_145: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_487, view_312, arg177_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_487 = view_312 = arg177_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_115: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_145)
        mul_491: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_145, sigmoid_115);  convolution_145 = sigmoid_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_313: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg178_1, [1, 1536, -1]);  arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_492: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg179_1, 0.09125009274634042);  arg179_1 = None
        view_314: "f32[1536]" = torch.ops.aten.view.default(mul_492, [-1]);  mul_492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_104 = torch.ops.aten.var_mean.correction(view_313, [0, 2], correction = 0, keepdim = True)
        getitem_208: "f32[1, 1536, 1]" = var_mean_104[0]
        getitem_209: "f32[1, 1536, 1]" = var_mean_104[1];  var_mean_104 = None
        add_125: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_104: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        sub_104: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_313, getitem_209);  view_313 = getitem_209 = None
        mul_493: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = rsqrt_104 = None
        unsqueeze_104: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_314, -1);  view_314 = None
        mul_494: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_104);  mul_493 = unsqueeze_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_315: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_494, [1536, 384, 1, 1]);  mul_494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_146: "f32[8, 1536, 9, 9]" = torch.ops.aten.convolution.default(mul_491, view_315, arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_491 = view_315 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_22: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_146, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_147: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_22, arg181_1, arg182_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg181_1 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_21: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_147);  convolution_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_148: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_21, arg183_1, arg184_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg183_1 = arg184_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_116: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_148);  convolution_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_495: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_146, sigmoid_116);  convolution_146 = sigmoid_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_496: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_495, 2.0);  mul_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_497: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_496, 0.2);  mul_496 = None
        add_126: "f32[8, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_497, convolution_142);  mul_497 = convolution_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_117: "f32[8, 1536, 9, 9]" = torch.ops.aten.sigmoid.default(add_126)
        mul_498: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(add_126, sigmoid_117);  sigmoid_117 = None
        mul_499: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_498, 0.9805806756909201);  mul_498 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_316: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg185_1, [1, 384, -1]);  arg185_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_500: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg186_1, 0.04562504637317021);  arg186_1 = None
        view_317: "f32[384]" = torch.ops.aten.view.default(mul_500, [-1]);  mul_500 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_105 = torch.ops.aten.var_mean.correction(view_316, [0, 2], correction = 0, keepdim = True)
        getitem_210: "f32[1, 384, 1]" = var_mean_105[0]
        getitem_211: "f32[1, 384, 1]" = var_mean_105[1];  var_mean_105 = None
        add_127: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_105: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_105: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_316, getitem_211);  view_316 = getitem_211 = None
        mul_501: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = rsqrt_105 = None
        unsqueeze_105: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_317, -1);  view_317 = None
        mul_502: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_501, unsqueeze_105);  mul_501 = unsqueeze_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_318: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_502, [384, 1536, 1, 1]);  mul_502 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_149: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_499, view_318, arg187_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_499 = view_318 = arg187_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_118: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_149)
        mul_503: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_149, sigmoid_118);  convolution_149 = sigmoid_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_319: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg188_1, [1, 384, -1]);  arg188_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_504: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg189_1, 0.07450538873672485);  arg189_1 = None
        view_320: "f32[384]" = torch.ops.aten.view.default(mul_504, [-1]);  mul_504 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_106 = torch.ops.aten.var_mean.correction(view_319, [0, 2], correction = 0, keepdim = True)
        getitem_212: "f32[1, 384, 1]" = var_mean_106[0]
        getitem_213: "f32[1, 384, 1]" = var_mean_106[1];  var_mean_106 = None
        add_128: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_106: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_106: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_319, getitem_213);  view_319 = getitem_213 = None
        mul_505: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = rsqrt_106 = None
        unsqueeze_106: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_320, -1);  view_320 = None
        mul_506: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_106);  mul_505 = unsqueeze_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_321: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_506, [384, 64, 3, 3]);  mul_506 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_150: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_503, view_321, arg190_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_503 = view_321 = arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_119: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_150)
        mul_507: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_150, sigmoid_119);  convolution_150 = sigmoid_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_322: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg191_1, [1, 384, -1]);  arg191_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_508: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg192_1, 0.07450538873672485);  arg192_1 = None
        view_323: "f32[384]" = torch.ops.aten.view.default(mul_508, [-1]);  mul_508 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_107 = torch.ops.aten.var_mean.correction(view_322, [0, 2], correction = 0, keepdim = True)
        getitem_214: "f32[1, 384, 1]" = var_mean_107[0]
        getitem_215: "f32[1, 384, 1]" = var_mean_107[1];  var_mean_107 = None
        add_129: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-05);  getitem_214 = None
        rsqrt_107: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_107: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_322, getitem_215);  view_322 = getitem_215 = None
        mul_509: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = rsqrt_107 = None
        unsqueeze_107: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_323, -1);  view_323 = None
        mul_510: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_509, unsqueeze_107);  mul_509 = unsqueeze_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_324: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_510, [384, 64, 3, 3]);  mul_510 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_151: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_507, view_324, arg193_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_507 = view_324 = arg193_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_120: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_151)
        mul_511: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_151, sigmoid_120);  convolution_151 = sigmoid_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_325: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg194_1, [1, 1536, -1]);  arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_512: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg195_1, 0.09125009274634042);  arg195_1 = None
        view_326: "f32[1536]" = torch.ops.aten.view.default(mul_512, [-1]);  mul_512 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_108 = torch.ops.aten.var_mean.correction(view_325, [0, 2], correction = 0, keepdim = True)
        getitem_216: "f32[1, 1536, 1]" = var_mean_108[0]
        getitem_217: "f32[1, 1536, 1]" = var_mean_108[1];  var_mean_108 = None
        add_130: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_108: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_108: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_325, getitem_217);  view_325 = getitem_217 = None
        mul_513: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = rsqrt_108 = None
        unsqueeze_108: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_326, -1);  view_326 = None
        mul_514: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_513, unsqueeze_108);  mul_513 = unsqueeze_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_327: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_514, [1536, 384, 1, 1]);  mul_514 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_152: "f32[8, 1536, 9, 9]" = torch.ops.aten.convolution.default(mul_511, view_327, arg196_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_511 = view_327 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_23: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_152, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_153: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_23, arg197_1, arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg197_1 = arg198_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_22: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_153);  convolution_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_154: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_22, arg199_1, arg200_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_22 = arg199_1 = arg200_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_121: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_154);  convolution_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_515: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_152, sigmoid_121);  convolution_152 = sigmoid_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_516: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_515, 2.0);  mul_515 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_517: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_516, 0.2);  mul_516 = None
        add_131: "f32[8, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_517, add_126);  mul_517 = add_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:178 in forward, code: out = self.act1(x) * self.beta
        sigmoid_122: "f32[8, 1536, 9, 9]" = torch.ops.aten.sigmoid.default(add_131)
        mul_518: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(add_131, sigmoid_122);  sigmoid_122 = None
        mul_519: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_518, 0.9622504486493761);  mul_518 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_328: "f32[1, 384, 1536]" = torch.ops.aten.view.default(arg201_1, [1, 384, -1]);  arg201_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_520: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg202_1, 0.04562504637317021);  arg202_1 = None
        view_329: "f32[384]" = torch.ops.aten.view.default(mul_520, [-1]);  mul_520 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_109 = torch.ops.aten.var_mean.correction(view_328, [0, 2], correction = 0, keepdim = True)
        getitem_218: "f32[1, 384, 1]" = var_mean_109[0]
        getitem_219: "f32[1, 384, 1]" = var_mean_109[1];  var_mean_109 = None
        add_132: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_109: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_109: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_328, getitem_219);  view_328 = getitem_219 = None
        mul_521: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = rsqrt_109 = None
        unsqueeze_109: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_329, -1);  view_329 = None
        mul_522: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_521, unsqueeze_109);  mul_521 = unsqueeze_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_330: "f32[384, 1536, 1, 1]" = torch.ops.aten.view.default(mul_522, [384, 1536, 1, 1]);  mul_522 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_155: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_519, view_330, arg203_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_519 = view_330 = arg203_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:187 in forward, code: out = self.conv2(self.act2(out))
        sigmoid_123: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_155)
        mul_523: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_155, sigmoid_123);  convolution_155 = sigmoid_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_331: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg204_1, [1, 384, -1]);  arg204_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_524: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg205_1, 0.07450538873672485);  arg205_1 = None
        view_332: "f32[384]" = torch.ops.aten.view.default(mul_524, [-1]);  mul_524 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_110 = torch.ops.aten.var_mean.correction(view_331, [0, 2], correction = 0, keepdim = True)
        getitem_220: "f32[1, 384, 1]" = var_mean_110[0]
        getitem_221: "f32[1, 384, 1]" = var_mean_110[1];  var_mean_110 = None
        add_133: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_110: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_110: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_331, getitem_221);  view_331 = getitem_221 = None
        mul_525: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = rsqrt_110 = None
        unsqueeze_110: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_332, -1);  view_332 = None
        mul_526: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_110);  mul_525 = unsqueeze_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_333: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_526, [384, 64, 3, 3]);  mul_526 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_156: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_523, view_333, arg206_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_523 = view_333 = arg206_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:189 in forward, code: out = self.conv2b(self.act2b(out))
        sigmoid_124: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_156)
        mul_527: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_156, sigmoid_124);  convolution_156 = sigmoid_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_334: "f32[1, 384, 576]" = torch.ops.aten.view.default(arg207_1, [1, 384, -1]);  arg207_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_528: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg208_1, 0.07450538873672485);  arg208_1 = None
        view_335: "f32[384]" = torch.ops.aten.view.default(mul_528, [-1]);  mul_528 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_111 = torch.ops.aten.var_mean.correction(view_334, [0, 2], correction = 0, keepdim = True)
        getitem_222: "f32[1, 384, 1]" = var_mean_111[0]
        getitem_223: "f32[1, 384, 1]" = var_mean_111[1];  var_mean_111 = None
        add_134: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_111: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_111: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(view_334, getitem_223);  view_334 = getitem_223 = None
        mul_529: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = rsqrt_111 = None
        unsqueeze_111: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(view_335, -1);  view_335 = None
        mul_530: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_529, unsqueeze_111);  mul_529 = unsqueeze_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_336: "f32[384, 64, 3, 3]" = torch.ops.aten.view.default(mul_530, [384, 64, 3, 3]);  mul_530 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_157: "f32[8, 384, 9, 9]" = torch.ops.aten.convolution.default(mul_527, view_336, arg209_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_527 = view_336 = arg209_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:192 in forward, code: out = self.conv3(self.act3(out))
        sigmoid_125: "f32[8, 384, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_157)
        mul_531: "f32[8, 384, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_157, sigmoid_125);  convolution_157 = sigmoid_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_337: "f32[1, 1536, 384]" = torch.ops.aten.view.default(arg210_1, [1, 1536, -1]);  arg210_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_532: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg211_1, 0.09125009274634042);  arg211_1 = None
        view_338: "f32[1536]" = torch.ops.aten.view.default(mul_532, [-1]);  mul_532 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_112 = torch.ops.aten.var_mean.correction(view_337, [0, 2], correction = 0, keepdim = True)
        getitem_224: "f32[1, 1536, 1]" = var_mean_112[0]
        getitem_225: "f32[1, 1536, 1]" = var_mean_112[1];  var_mean_112 = None
        add_135: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_112: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        sub_112: "f32[1, 1536, 384]" = torch.ops.aten.sub.Tensor(view_337, getitem_225);  view_337 = getitem_225 = None
        mul_533: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = rsqrt_112 = None
        unsqueeze_112: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(view_338, -1);  view_338 = None
        mul_534: "f32[1, 1536, 384]" = torch.ops.aten.mul.Tensor(mul_533, unsqueeze_112);  mul_533 = unsqueeze_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_339: "f32[1536, 384, 1, 1]" = torch.ops.aten.view.default(mul_534, [1536, 384, 1, 1]);  mul_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_158: "f32[8, 1536, 9, 9]" = torch.ops.aten.convolution.default(mul_531, view_339, arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_531 = view_339 = arg212_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:42 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_24: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_158, [2, 3], True)
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:46 in forward, code: x_se = self.fc1(x_se)
        convolution_159: "f32[8, 384, 1, 1]" = torch.ops.aten.convolution.default(mean_24, arg213_1, arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg213_1 = arg214_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:47 in forward, code: x_se = self.act(self.bn(x_se))
        relu_23: "f32[8, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_159);  convolution_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:48 in forward, code: x_se = self.fc2(x_se)
        convolution_160: "f32[8, 1536, 1, 1]" = torch.ops.aten.convolution.default(relu_23, arg215_1, arg216_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg215_1 = arg216_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_126: "f32[8, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_160);  convolution_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:49 in forward, code: return x * self.gate(x_se)
        mul_535: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_158, sigmoid_126);  convolution_158 = sigmoid_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:194 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_536: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_535, 2.0);  mul_535 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:199 in forward, code: out = out * self.alpha + shortcut
        mul_537: "f32[8, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_536, 0.2);  mul_536 = None
        add_136: "f32[8, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_537, add_131);  mul_537 = add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:99 in forward, code: self.weight.reshape(1, self.out_channels, -1), None, None,
        view_340: "f32[1, 2304, 1536]" = torch.ops.aten.view.default(arg217_1, [1, 2304, -1]);  arg217_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:100 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_538: "f32[2304, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg218_1, 0.04562504637317021);  arg218_1 = None
        view_341: "f32[2304]" = torch.ops.aten.view.default(mul_538, [-1]);  mul_538 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:98 in forward, code: weight = F.batch_norm(
        var_mean_113 = torch.ops.aten.var_mean.correction(view_340, [0, 2], correction = 0, keepdim = True)
        getitem_226: "f32[1, 2304, 1]" = var_mean_113[0]
        getitem_227: "f32[1, 2304, 1]" = var_mean_113[1];  var_mean_113 = None
        add_137: "f32[1, 2304, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_113: "f32[1, 2304, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_113: "f32[1, 2304, 1536]" = torch.ops.aten.sub.Tensor(view_340, getitem_227);  view_340 = getitem_227 = None
        mul_539: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = rsqrt_113 = None
        unsqueeze_113: "f32[2304, 1]" = torch.ops.aten.unsqueeze.default(view_341, -1);  view_341 = None
        mul_540: "f32[1, 2304, 1536]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_113);  mul_539 = unsqueeze_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:101 in forward, code: training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        view_342: "f32[2304, 1536, 1, 1]" = torch.ops.aten.view.default(mul_540, [2304, 1536, 1, 1]);  mul_540 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/std_conv.py:102 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convolution_161: "f32[8, 2304, 9, 9]" = torch.ops.aten.convolution.default(add_136, view_342, arg219_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_136 = view_342 = arg219_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/models/nfnet.py:446 in forward_features, code: x = self.final_act(x)
        sigmoid_127: "f32[8, 2304, 9, 9]" = torch.ops.aten.sigmoid.default(convolution_161)
        mul_541: "f32[8, 2304, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_161, sigmoid_127);  convolution_161 = sigmoid_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_25: "f32[8, 2304, 1, 1]" = torch.ops.aten.mean.dim(mul_541, [-1, -2], True);  mul_541 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_343: "f32[8, 2304]" = torch.ops.aten.view.default(mean_25, [8, 2304]);  mean_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-benchmarks/lib/python3.12/site-packages/timm/layers/classifier.py:132 in forward, code: x = self.fc(x)
        permute_1: "f32[2304, 1000]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg221_1, view_343, permute_1);  arg221_1 = view_343 = permute_1 = None
        return (addmm_1,)
        